
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass

RNG = np.random.default_rng(42)
torch.manual_seed(42)

MUON_MASS_MEV = 105.7
X0_REF_CM = 1.76

print("fixed seed so that each time pipeline produces same result")

@dataclass
class Grid:
  nx: int
  ny: int
  nz: int
  voxel_cm: float
  @property
  def half_x(self): return self.nx * self.voxel_cm / 2
  @property
  def half_y(self): return self.ny * self.voxel_cm / 2
  @property
  def half_z(self): return self.nz * self.voxel_cm / 2

  def layer_z_centers(self):
    edges = np.linspace(-self.half_z, self.half_z, self.nz+1)
    return 0.5 * (edges [:-1] + edges[1:])
grid = Grid(nx=16, ny =16, nz =12, voxel_cm = 2.5)
print (f"Grid:{grid.nx}x{grid.ny}x{grid.nz} voxels,{grid.voxel_cm} cm each")


# the housing: a dense block (density 0.75) with a thin air border
true_density = np.zeros((grid.nz, grid.ny, grid.nx))
true_density[1:-1, 1:-1, 1:-1] = 0.75

# carve out a void (density 0.05) off-center - the defect we want the pipeline to notice

true_density = np.zeros((grid.nz, grid.ny, grid.nx))
true_density[1:-1, 1:-1, 1:-1] = 0.75

# carve out a void (density 0.05) off-center the defect we want the pipeline to notice

z_index, y_index, x_index = np.meshgrid(
    np.arange(grid.nz), np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
void_center = np.array([grid.nz * 0.4, grid.ny * 0.65, grid.nx * 0.35])
distance_to_void_center = np.sqrt(
    (z_index - void_center[0]) ** 2 +
    (y_index - void_center[1]) ** 2 +
    (x_index - void_center[2]) ** 2)
true_density[distance_to_void_center < 2.24] = 0.05

print(f"Housing voxels: {np.count_nonzero(true_density > 0.5)}")
print(f"Void voxels:    {np.count_nonzero((true_density > 0.01) & (true_density < 0.5))}")

plt.figure(figsize=(4, 4))
plt.imshow(true_density[int(grid.nz * 0.4)], vmin=0, vmax=0.9, cmap="viridis")
plt.title("True density, one z-slice")
plt.colorbar()
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/dense_block.png")
plt.pause(0.3)




########### Mathemaatical formulas to use in the stream later


def muon_beta(momentum_mev):
  energy = np.sqrt(momentum_mev ** 2 + MUON_MASS_MEV ** 2)
  return momentum_mev / energy


def highland_theta0(momentum_mev, path_length_cm, radiation_length_cm):
  beta = muon_beta(momentum_mev)
  ratio = np.clip(path_length_cm / radiation_length_cm, 1e-12, None)
  theta0 = (13.6 / (beta * momentum_mev)) * np.sqrt(ratio)
  log_correction = 1.0 + 0.038 * np.log(np.clip(ratio, 1e-6, None))
  return theta0 * log_correction


def density_to_radiation_length(density_fraction, eps=1e-3):
  return X0_REF_CM / (density_fraction + eps)


def momentum_from_beta(beta):
  beta = np.clip(beta, 1e-3, 1 - 1e-7)
  return MUON_MASS_MEV * beta / np.sqrt(1 - beta ** 2)


def build_local_axes(forward_direction):
  """Two unit vectors perpendicular to forward_direction and to each
  other - a local coordinate frame for measuring 'how much did this one
  muon scatter relative to where it was already heading'."""
  n = len(forward_direction)
  reference = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
  overlap = np.sum(reference * forward_direction, axis=1, keepdims=True)
  side_u = reference - overlap * forward_direction
  side_u_length = np.linalg.norm(side_u, axis=1, keepdims=True)
  too_short = side_u_length[:, 0] < 1e-6
  if too_short.any():
    reference2 = np.array([0.0, 1.0, 0.0])
    overlap2 = np.sum(reference2 * forward_direction[too_short], axis=1, keepdims=True)
    side_u[too_short] = reference2 - overlap2 * forward_direction[too_short]
    side_u_length = np.linalg.norm(side_u, axis=1, keepdims=True)
  side_u = side_u / side_u_length
  side_v = np.cross(forward_direction, side_u)
  return side_u, side_v


print("5 physics formulas defined.")


###### ##### scattering anlge dependence on momentum

# how theta0 showed by highland theta depends on momentum and density
print(f"{'p [MeV]':>10}{'d=0.05':>10}{'d=0.3':>10}{'d=0.75':>10}{'d=1.0':>10}")
for p in [300, 600, 1000, 2000]:
    row = f"{p:10.0f}"
    for d in [0.05, 0.3, 0.75, 1.0]:
        X0 = density_to_radiation_length(np.array([d]))
        theta0 = highland_theta0(np.array([float(p)]), 2.5, X0)[0]
        row += f"{theta0*1000:10.2f}"
    print(row)

plt.figure(figsize=(5, 3.5))
p_curve = np.linspace(200, 2500, 200)
for d in [0.05, 0.3, 0.75, 1.0]:
    X0 = density_to_radiation_length(np.array([d]))
    plt.plot(p_curve, highland_theta0(p_curve, 2.5, X0) * 1000, label=f"density={d}")
plt.xlabel("momentum [MeV]"); plt.ylabel("theta0 [mrad]"); plt.yscale("log")
plt.title(r"Scattering angle ($\theta$) vs momentum"); plt.legend()
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/highland_theta_dependence_momentum.png")
plt.pause(0.5)


##### Generate a batch of new muon events

### detector plan choices

TOP_PLANES_Z = (45.0, 35.0, 25.0)
BOTTOM_PLANES_Z = (-25.0, -35.0, -45.0)
ALL_PLANES_Z = TOP_PLANES_Z + BOTTOM_PLANES_Z


### Noise model

DETECTOR_RESOLUTION_CM = 0.05
BACKGROUND_HIT_PROBABILITY = 0.35
BACKGROUND_HIT_SPREAD_CM = 12.0

#### TOF beta resolution

TOF_BETA_RESOLUTION = 1e-4


###3 Hits range

MAX_HITS_PER_EVENT = 2 * len(ALL_PLANES_Z)

N_EVENTS = 10000


# --- momentum for every event, plus a noisy time-of-flight-based guess ---
momentum_mev = RNG.uniform(300, 5000, size=N_EVENTS)
true_beta = muon_beta(momentum_mev)
tof_beta_measurement = np.clip(true_beta + RNG.normal(0, TOF_BETA_RESOLUTION, N_EVENTS),
                                1e-3, 1 - 1e-7)

print(f"{N_EVENTS} events, momentum {momentum_mev.min():.0f}-{momentum_mev.max():.0f} MeV")
plt.figure(figsize=(4, 3))
plt.hist(momentum_mev, bins=40)
plt.title("Momentum spectrum"); plt.xlabel("MeV")
plt.show(block=False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/hits_range.png")
plt.pause(0.5)




# --- entry position and direction: a 20-degree-wide beam cone ---
entry_xy = RNG.uniform(-grid.half_x * 0.6, grid.half_x * 0.6, size=(N_EVENTS, 2))
divergence_rad = np.deg2rad(20.0)
angle_x = RNG.normal(0, divergence_rad, N_EVENTS)
angle_y = RNG.normal(0, divergence_rad, N_EVENTS)


#entry_direction = np.stack([
#    np.sin(angle_x), np.sin(angle_y), -np.cos(angle_x) * np.cos(angle_y)], axis=1)


entry_direction = np.stack([
    -np.sin(angle_x),
    np.cos(angle_x) * np.sin(angle_y),
    -np.cos(angle_x) * np.cos(angle_y),
], axis = 1)



norms = np.linalg.norm(entry_direction, axis=1)
print("entry_direction norm range:", norms.min(), "to", norms.max(), "(should read 1.0 to 1.0 now)")


print("Entry direction z-component range:", entry_direction[:, 2].min(),
      "to", entry_direction[:, 2].max(),
      "(should be negative - beam points down; occasionally a rare >90-degree "
      "sampled angle flips this for one event out of thousands, a known edge "
      "case of a Gaussian-sampled wide beam cone, not something worth guarding "
      "against for a demo dataset this size)")
plt.figure(figsize=(4, 4))
plt.scatter(entry_xy[:, 0], entry_xy[:, 1], s=2, alpha=0.3)
plt.title("Entry positions on the beam face"); plt.gca().set_aspect("equal")
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/postions_on_each_face_scatter.png")
plt.pause(0.5)

plt.figure(figsize=(4,4))
plt.hist2d(entry_xy[:,0],entry_xy[:, 1], bins=25)
plt.title("Entry positions, binned")
plt.colorbar()
plt.gca().set_aspect("equal")
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/postions_on_each_face_2D.png")
plt.pause(0.5)



# --- true scattering angle: sum the Highland contribution from every
# z-layer each straight-line path crosses, using the REAL density field ---
z_centers = grid.layer_z_centers()
total_theta0_squared = np.zeros(N_EVENTS)

for z in z_centers:
    distance_along_ray = (z - TOP_PLANES_Z[0]) / entry_direction[:, 2]
    x = entry_xy[:, 0] + distance_along_ray * entry_direction[:, 0]
    y = entry_xy[:, 1] + distance_along_ray * entry_direction[:, 1]

    voxel_x = np.floor((x + grid.half_x) / grid.voxel_cm).astype(int)
    voxel_y = np.floor((y + grid.half_y) / grid.voxel_cm).astype(int)
    inside_grid = (voxel_x >= 0) & (voxel_x < grid.nx) & (voxel_y >= 0) & (voxel_y < grid.ny)

    layer_index = int(np.argmin(np.abs(z_centers - z)))
    density_here = np.zeros(N_EVENTS)
    density_here[inside_grid] = true_density[layer_index, voxel_y[inside_grid], voxel_x[inside_grid]]

    radiation_length = density_to_radiation_length(np.clip(density_here, 0, None))
    theta0_this_layer = highland_theta0(momentum_mev, grid.voxel_cm, radiation_length)
    total_theta0_squared += np.where(inside_grid, theta0_this_layer ** 2, 0.0)

true_theta0 = np.sqrt(total_theta0_squared)
print(f"True scattering angle: mean {true_theta0.mean()*100000:.2f} mrad, "
      f"max {true_theta0.max()*100000:.1f} mrad")
plt.figure(figsize=(4, 3))
plt.hist(true_theta0 * 100000, bins=40)
plt.title("True scattering angle"); plt.xlabel("mrad")
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/scaterring_angle_true.png")
plt.pause(0.5)




# --- apply the scattering kick, in the frame local to each muon ---
kick_u = RNG.normal(0, true_theta0)
kick_v = RNG.normal(0, true_theta0)
side_u, side_v = build_local_axes(entry_direction)
exit_direction = entry_direction + kick_u[:, None] * side_u + kick_v[:, None] * side_v
exit_direction = exit_direction / np.linalg.norm(exit_direction, axis=1, keepdims=True)

angle_between = np.degrees(np.arccos(np.clip(
    np.sum(entry_direction * exit_direction, axis=1), -1, 1)))
print(f"Entry-to-exit deflection: mean {angle_between.mean():.3f} deg, "
      f"max {angle_between.max():.2f} deg (tiny compared to the 20-deg beam divergence)")




# --- project entry/exit tracks onto the 6 detector planes, add smearing ---
distance_to_exit = (z_centers[-1] - TOP_PLANES_Z[0]) / entry_direction[:, 2]
exit_xy = entry_xy + distance_to_exit[:, None] * entry_direction[:, :2]
exit_z = z_centers[-1]

true_hit_xy = {}
for plane_z in TOP_PLANES_Z:
    distance = (plane_z - TOP_PLANES_Z[0]) / entry_direction[:, 2]
    xy = entry_xy + distance[:, None] * entry_direction[:, :2]
    true_hit_xy[plane_z] = xy + RNG.normal(0, DETECTOR_RESOLUTION_CM, xy.shape)
for plane_z in BOTTOM_PLANES_Z:
    distance = (plane_z - exit_z) / exit_direction[:, 2]
    xy = exit_xy + distance[:, None] * exit_direction[:, :2]
    true_hit_xy[plane_z] = xy + RNG.normal(0, DETECTOR_RESOLUTION_CM, xy.shape)

print("Computed the true hit position on each of the 6 planes for all events.")
print("Example, event 0:")
for plane_z in ALL_PLANES_Z:
    print(f"  z={plane_z:6.1f} cm -> hit at {np.round(true_hit_xy[plane_z][0], 3)}")



# --- build the padded hit arrays: 1 real hit + sometimes 1 background
# hit per plane, shuffled so the real hit isn't always first ---
hit_xyz = np.zeros((N_EVENTS, MAX_HITS_PER_EVENT, 3))
hit_is_top_plane = np.zeros((N_EVENTS, MAX_HITS_PER_EVENT))
hit_is_real = np.zeros((N_EVENTS, MAX_HITS_PER_EVENT))
hit_is_used = np.zeros((N_EVENTS, MAX_HITS_PER_EVENT), dtype=bool)

for event_i in range(N_EVENTS):
    next_free_slot = 0
    for plane_z in ALL_PLANES_Z:
        is_top = 1.0 if plane_z in TOP_PLANES_Z else 0.0

        candidates_xy = [true_hit_xy[plane_z][event_i]]
        candidates_is_real = [1]
        if RNG.uniform() < BACKGROUND_HIT_PROBABILITY:
            noise_xy = true_hit_xy[plane_z][event_i] + RNG.normal(0, BACKGROUND_HIT_SPREAD_CM, 2)
            candidates_xy.append(noise_xy)
            candidates_is_real.append(0)

        for candidate_index in RNG.permutation(len(candidates_xy)):
            slot = next_free_slot
            hit_xyz[event_i, slot] = [candidates_xy[candidate_index][0],
                                       candidates_xy[candidate_index][1], plane_z]
            hit_is_top_plane[event_i, slot] = is_top
            hit_is_real[event_i, slot] = candidates_is_real[candidate_index]
            hit_is_used[event_i, slot] = True
            next_free_slot += 1

hits_per_event = hit_is_used.sum(axis=1)
print(f"Hits per event: min {hits_per_event.min():.0f}, max {hits_per_event.max():.0f}, "
      f"mean {hits_per_event.mean():.2f} (6 planes, some with an extra background hit)")

plt.figure(figsize=(4, 3))
plt.hist(hits_per_event, bins=np.arange(5.5, 13))
plt.title("Hit candidates per event"); plt.xlabel("count")
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/hits_candidate_per_event.png")
plt.pause(0.5)



xyz_normalized = hit_xyz.copy()
xyz_normalized[..., 0] /= 20.0   # x in units of the grid half-width
xyz_normalized[..., 1] /= 20.0
xyz_normalized[..., 2] /= 45.0   # z in units of the farthest plane distance

MOMENTUM_SCALE_MEV = 1000.0
tof_momentum_guess = momentum_from_beta(tof_beta_measurement) / MOMENTUM_SCALE_MEV
true_momentum_scaled = momentum_mev / MOMENTUM_SCALE_MEV

hit_features_tensor = torch.tensor(
    np.concatenate([xyz_normalized, hit_is_top_plane[..., None]], axis=-1), dtype=torch.float32)
hit_is_used_tensor = torch.tensor(hit_is_used, dtype=torch.bool)
hit_is_real_tensor = torch.tensor(hit_is_real, dtype=torch.float32)
tof_momentum_tensor = torch.tensor(tof_momentum_guess, dtype=torch.float32).unsqueeze(-1)
true_momentum_tensor = torch.tensor(true_momentum_scaled, dtype=torch.float32)

print("hit_features_tensor shape:", tuple(hit_features_tensor.shape),
      "= (events, max_hits, [x, y, z, is_top_plane])")
print("hit_is_used_tensor shape: ", tuple(hit_is_used_tensor.shape))



###### Graph neural Network Definition

class GraphAttentionBlock(nn.Module):
    """Every hit in an event attends to every other hit in that same
    event. Plain matmul + transpose, no einsum."""

    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.to_query = nn.Linear(feature_dim, feature_dim)
        self.to_key = nn.Linear(feature_dim, feature_dim)
        self.to_value = nn.Linear(feature_dim, feature_dim)
        self.combine = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feature_dim))
        self.norm_after_attention = nn.LayerNorm(feature_dim)
        self.norm_after_combine = nn.LayerNorm(feature_dim)

    def forward(self, node_features, node_is_real):
        query = self.to_query(node_features)
        key = self.to_key(node_features)
        value = self.to_value(node_features)

        attention_scores = (query @ key.transpose(1, 2)) / (query.shape[-1] ** 0.5)
        both_real = node_is_real.unsqueeze(2) & node_is_real.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~both_real, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1) * both_real

        messages = attention_weights @ value
        node_features = self.norm_after_attention(node_features + messages)
        combined = torch.cat([node_features, messages], dim=-1)
        node_features = self.norm_after_combine(node_features + self.combine(combined))
        return node_features


class TrackGNN(nn.Module):
    """Embed hits -> 2 attention blocks -> noise-rejection head +
    momentum (mean, log-variance) head."""

    def __init__(self, input_dim=4, feature_dim=48):
        super().__init__()
        self.embed_hit = nn.Sequential(
            nn.Linear(input_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))
        self.attention_block_1 = GraphAttentionBlock(feature_dim)
        self.attention_block_2 = GraphAttentionBlock(feature_dim)
        self.node_head = nn.Linear(feature_dim, 1)
        self.momentum_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, hit_features, hit_is_used, tof_momentum_guess):
        node_features = self.embed_hit(hit_features)
        node_features = self.attention_block_1(node_features, hit_is_used)
        node_features = self.attention_block_2(node_features, hit_is_used)

        node_logit = self.node_head(node_features).squeeze(-1).masked_fill(~hit_is_used, -1e9)

        signal_probability = torch.sigmoid(node_logit) * hit_is_used
        pooling_weight = signal_probability / (signal_probability.sum(dim=1, keepdim=True) + 1e-6)
        pooled_features = (pooling_weight.unsqueeze(-1) * node_features).sum(dim=1)

        mean_and_log_var = self.momentum_head(torch.cat([pooled_features, tof_momentum_guess], dim=-1))
        return node_logit, mean_and_log_var[:, 0], mean_and_log_var[:, 1]


def gaussian_negative_log_likelihood(predicted_mean, predicted_log_var, true_value):
    predicted_log_var = torch.clamp(predicted_log_var, -4, 12)
    return 0.5 * (predicted_log_var + (true_value - predicted_mean) ** 2 / torch.exp(predicted_log_var))

print("Model classes defined.")


# demo: an untrained forward pass, just to see the shapes flow through
model = TrackGNN()
sample_node_logit, sample_mean, sample_log_var = model(
    hit_features_tensor[:4], hit_is_used_tensor[:4], tof_momentum_tensor[:4])
print("node_logit shape:", tuple(sample_node_logit.shape), "(one score per hit candidate)")
print("momentum mean (untrained, meaningless yet):", sample_mean.detach().numpy())




#### Training the model

n_validation = int(N_EVENTS * 0.1)
shuffled_indices = torch.randperm(N_EVENTS)
validation_indices = shuffled_indices[:n_validation]
training_indices = shuffled_indices[n_validation:]

model = TrackGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

train_loss_history = []
val_loss_history = []
val_accuracy_history = []
val_mae_history = []

EPOCHS = 60
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    model.train()
    epoch_indices = training_indices[torch.randperm(len(training_indices))]
    total_loss = 0.0

    for batch_start in range(0, len(epoch_indices), BATCH_SIZE):
        batch = epoch_indices[batch_start: batch_start + BATCH_SIZE]

        node_logit, mom_mean, mom_log_var = model(
            hit_features_tensor[batch], hit_is_used_tensor[batch], tof_momentum_tensor[batch])

        used = hit_is_used_tensor[batch]
        noise_loss = F.binary_cross_entropy_with_logits(node_logit[used], hit_is_real_tensor[batch][used])
        mom_loss = gaussian_negative_log_likelihood(
            mom_mean, mom_log_var, true_momentum_tensor[batch]).mean()
        loss = noise_loss + mom_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)

    train_loss_history.append(total_loss / len(training_indices))

    model.eval()
    with torch.no_grad():
        node_logit, mom_mean, mom_log_var = model(
            hit_features_tensor[validation_indices], hit_is_used_tensor[validation_indices],
            tof_momentum_tensor[validation_indices])
        used = hit_is_used_tensor[validation_indices]
        labels = hit_is_real_tensor[validation_indices]
        true_mom = true_momentum_tensor[validation_indices]

        noise_loss = F.binary_cross_entropy_with_logits(node_logit[used], labels[used])
        mom_loss = gaussian_negative_log_likelihood(mom_mean, mom_log_var, true_mom).mean()
        val_loss_history.append((noise_loss + mom_loss).item())

        predicted_labels = (torch.sigmoid(node_logit[used]) > 0.5).float()
        val_accuracy_history.append((predicted_labels == labels[used]).float().mean().item())
        val_mae_history.append((mom_mean - true_mom).abs().mean().item() * MOMENTUM_SCALE_MEV)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d}  train_loss={train_loss_history[-1]:.4f}  "
              f"val_loss={val_loss_history[-1]:.4f}  "
              f"noise_accuracy={val_accuracy_history[-1]:.4f}  "
              f"momentum_MAE={val_mae_history[-1]:.1f} MeV")

print("\nTraining done.")



fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].plot(train_loss_history, label="train")
axes[0].plot(val_loss_history, label="validation")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(val_accuracy_history)
axes[1].set_title("Validation noise-rejection accuracy"); axes[1].set_ylim(0.5, 1)
axes[2].plot(val_mae_history)
axes[2].set_title("Validation momentum MAE [MeV]")
fig.tight_layout()
plt.show(block = False)
plt.savefig("/Users/binishbatool/PycharmProjects/pythonProject/gnn_track_finding/validation_gnn.png")