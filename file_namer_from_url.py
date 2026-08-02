
import requests
from bs4 import BeautifulSoup


def print_code(doc_url):

    response = requests.get(doc_url)
    response.raise_for_status()


    soup = BeautifulSoup(response.text, "html.parser")


    rows = soup.find_all("tr")
    points = []

    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        x_str = cells[0].get_text().strip()
        char = cells[1].get_text()
        y_str = cells[2].get_text().strip()

        if not x_str or not y_str:
            continue

        x = int(x_str)
        y = int(y_str)
        points.append((x, y, char))

    if not points:
        print("No data found")
        return

    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for x, y, char in points:
        grid[y][x] = char

    for row in grid:
        print("".join(row))



url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
print_code(url)