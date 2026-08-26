
import numpy as np
import torch
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

