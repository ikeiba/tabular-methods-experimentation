"""Text rendering for Chrome Dino."""

from __future__ import annotations

from typing import Sequence, Any


def render_ansi(
    *,
    screen_width: float,
    dino_x: float,
    dino_y: float,
    dino_width: float,
    dino_height: float,
    is_ducking: bool,
    obstacles: Sequence[Any],
    elapsed_steps: int,
    game_speed: float,
) -> str:
    """Build an 80x10 ASCII view from the continuous environment state."""
    cols = 80
    rows = 10
    scale_x = screen_width / cols
    scale_y = 20.0  # Each row represents 20 pixels of height

    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    # Draw ground on the bottom row
    for c in range(cols):
        grid[rows - 1][c] = "_"

    # Draw obstacles ('#' for cactus, 'V' for bird)
    for obs in obstacles:
        c_start = max(0, int(obs.x / scale_x))
        c_end = min(cols - 1, int((obs.x + obs.width) / scale_x))
        r_bottom = rows - 1 - int(obs.y / scale_y)
        r_top = rows - 1 - int((obs.y + obs.height) / scale_y)

        char = "#" if obs.kind == "cactus" else "V"
        for r in range(max(0, r_top), min(rows, r_bottom + 1)):
            for c in range(c_start, c_end + 1):
                if 0 <= c < cols:
                    grid[r][c] = char

    # Draw dinosaur ('d' when ducking, 'D' when standing/jumping)
    d_start = max(0, int(dino_x / scale_x))
    d_end = min(cols - 1, int((dino_x + dino_width) / scale_x))
    d_bottom = rows - 1 - int(dino_y / scale_y)
    d_top = rows - 1 - int((dino_y + dino_height) / scale_y)

    dino_char = "d" if is_ducking else "D"
    for r in range(max(0, d_top), min(rows, d_bottom + 1)):
        for c in range(d_start, d_end + 1):
            if 0 <= c < cols:
                grid[r][c] = dino_char

    header = f"Score: {elapsed_steps:05d} | Speed: {game_speed:.1f}"
    body = "\n".join("".join(row) for row in grid)
    return f"{header}\n{body}\n"