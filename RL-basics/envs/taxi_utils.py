import numpy as np
import gymnasium as gym
from gymnasium import spaces
from os import path
import matplotlib as mpl

mpl.rcParams['animation.embed_limit'] = 500.0

# ---------------------------------------------------------------------------
# Map and constants
# ---------------------------------------------------------------------------
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

WINDOW_SIZE = (550 * 1.5, 350 * 1.5)
LOCS_COLORS = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# ---------------------------------------------------------------------------
# External Render Cache and Helpers
# ---------------------------------------------------------------------------
_RENDER_CACHE = {
    "window": None,
    "assets_loaded": False,
    "taxi_imgs": None,
    "passenger_img": None,
    "destination_img": None,
    "median_horiz": None,
    "median_vert": None,
    "background_img": None,
    "taxi_orientation": 0,
}

def _get_surf_loc(map_loc, cell_size):
    return ((map_loc[1] * 2 + 1) * cell_size[0], (map_loc[0] + 1) * cell_size[1])

def render_taxi_frame(state, desc, lastaction, window_size, locs, locs_colors):
    try:
        import pygame
    except ImportError as e:
        raise DependencyNotInstalled('pygame is not installed') from e

    if _RENDER_CACHE["window"] is None:
        pygame.init()
        pygame.display.set_caption("Taxi")
        _RENDER_CACHE["window"] = pygame.Surface(window_size)

    window = _RENDER_CACHE["window"]
    cell_size = (window_size[0] / desc.shape[1], window_size[1] / desc.shape[0])

    if not _RENDER_CACHE["assets_loaded"]:
        here = path.join(path.dirname(__file__), '../imgs/custom_taxi')
        load = lambda name: pygame.transform.scale(pygame.image.load(path.join(here, name)), cell_size)

        _RENDER_CACHE["taxi_imgs"] = [load(n) for n in ("cab_front.png", "cab_rear.png", "cab_right.png", "cab_left.png")]
        _RENDER_CACHE["passenger_img"] = load("passenger.png")
        dest = load("uni.png")
        dest.set_alpha(170)
        _RENDER_CACHE["destination_img"] = dest
        _RENDER_CACHE["median_horiz"] = [load(n) for n in ("gridworld_median_left.png", "gridworld_median_horiz.png", "gridworld_median_right.png")]
        _RENDER_CACHE["median_vert"] = [load(n) for n in ("gridworld_median_top.png", "gridworld_median_vert.png", "gridworld_median_bottom.png")]
        _RENDER_CACHE["background_img"] = load("taxi_background.png")
        _RENDER_CACHE["assets_loaded"] = True

    for y in range(desc.shape[0]):
        for x in range(desc.shape[1]):
            cell = (x * cell_size[0], y * cell_size[1])
            window.blit(_RENDER_CACHE["background_img"], cell)
            ch = desc[y][x]
            if ch == b"|":
                if y == 0 or desc[y - 1][x] != b"|": window.blit(_RENDER_CACHE["median_vert"][0], cell)
                elif y == desc.shape[0] - 1 or desc[y + 1][x] != b"|": window.blit(_RENDER_CACHE["median_vert"][2], cell)
                else: window.blit(_RENDER_CACHE["median_vert"][1], cell)
            elif ch == b"-":
                if x == 0 or desc[y][x - 1] != b"-": window.blit(_RENDER_CACHE["median_horiz"][0], cell)
                elif x == desc.shape[1] - 1 or desc[y][x + 1] != b"-": window.blit(_RENDER_CACHE["median_horiz"][2], cell)
                else: window.blit(_RENDER_CACHE["median_horiz"][1], cell)

    for loc_cell, color in zip(locs, locs_colors):
        color_cell = pygame.Surface(cell_size)
        color_cell.set_alpha(128)
        color_cell.fill(color)
        sx, sy = _get_surf_loc(loc_cell, cell_size)
        window.blit(color_cell, (sx, sy + 10))

    taxi_row, taxi_col = state["row"], state["col"]
    pass_idx, dest_idx = state["pass_idx"], state["dest_idx"]

    if pass_idx < 4:
        window.blit(_RENDER_CACHE["passenger_img"], _get_surf_loc(locs[pass_idx], cell_size))

    if lastaction in (0, 1, 2, 3):
        _RENDER_CACHE["taxi_orientation"] = lastaction

    taxi_location = _get_surf_loc((taxi_row, taxi_col), cell_size)
    dest_loc = _get_surf_loc(locs[dest_idx], cell_size) if dest_idx < 4 else None
    taxi_img_to_draw = _RENDER_CACHE["taxi_imgs"][_RENDER_CACHE["taxi_orientation"]]

    if dest_loc is None:
        window.blit(taxi_img_to_draw, taxi_location)
    else:
        dest_y_adjusted = dest_loc[1] - cell_size[1] // 2
        if dest_loc[1] <= taxi_location[1]:
            window.blit(_RENDER_CACHE["destination_img"], (dest_loc[0], dest_y_adjusted))
            window.blit(taxi_img_to_draw, taxi_location)
        else:
            window.blit(taxi_img_to_draw, taxi_location)
            window.blit(_RENDER_CACHE["destination_img"], (dest_loc[0], dest_y_adjusted))

    return np.transpose(np.array(pygame.surfarray.pixels3d(window)), axes=(1, 0, 2))

def close_taxi_render():
    if _RENDER_CACHE["window"] is None:
        return
    import pygame
    pygame.display.quit()
    pygame.quit()
    _RENDER_CACHE["window"] = None
    _RENDER_CACHE["assets_loaded"] = False

# ---------------------------------------------------------------------------
# High-Level One-Liner Wrapper
# ---------------------------------------------------------------------------
def render_taxi(state, lastaction):
    """Exposes a clean interface to the student class, mapping tuples and actions."""
    if state is None:
        return None
        
    # Maps streamlined actions (0:UP, 1:RIGHT, 2:DOWN, 3:LEFT) 
    # back to legacy orientation indices (0:SOUTH, 1:NORTH, 2:EAST, 3:WEST)
    action_renderer_map = {0: 1, 1: 2, 2: 0, 3: 3}
    mapped_lastaction = action_renderer_map.get(lastaction, lastaction)
    
    # Converts internal tuple state format to the dict required by the legacy code
    render_state_dict = {
        "row": state[0],
        "col": state[1],
        "pass_idx": state[2],
        "dest_idx": state[3]
    }
    
    return render_taxi_frame(
        state=render_state_dict, desc=np.asarray(MAP, dtype="c"), lastaction=mapped_lastaction,
        window_size=WINDOW_SIZE, locs=LOCS, locs_colors=LOCS_COLORS
    )