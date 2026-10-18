"""PyGame visualization for the Chrome Dino environment."""

from __future__ import annotations

from typing import Sequence, Any

import pygame


class PygameRenderer:
    GROUND_OFFSET = 50  # Distance in pixels from the bottom of the window to the ground
    DAY_NIGHT_INTERVAL = 700  # Steps before toggling day/night mode

    # Day palette
    DAY_BG_COLOR = (247, 247, 247)
    DAY_GROUND_COLOR = (83, 83, 83)
    DAY_TEXT_COLOR = (83, 83, 83)

    # Night palette
    NIGHT_BG_COLOR = (32, 33, 36)
    NIGHT_GROUND_COLOR = (170, 170, 170)
    NIGHT_TEXT_COLOR = (220, 220, 220)

    # Entity colors
    DINO_COLOR = (60, 130, 230)
    DINO_DUCK_COLOR = (40, 95, 180)
    CACTUS_COLOR = (45, 150, 75)
    BIRD_COLOR = (215, 85, 50)

    def __init__(self, width: int, height: int, fps: int):
        pygame.init()
        self.width = width
        self.height = height
        self.ground_y = height - self.GROUND_OFFSET
        self.fps = fps

        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chrome Dino - RL Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.closed = False

    def render(
        self,
        *,
        dino_x: float,
        dino_y: float,
        dino_width: float,
        dino_height: float,
        is_ducking: bool,
        obstacles: Sequence[Any],
        elapsed_steps: int,
        game_speed: float,
    ) -> None:
        if self.closed:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Determine Day or Night cycle based on elapsed_steps (tiempo_juego)
        is_night = (elapsed_steps // self.DAY_NIGHT_INTERVAL) % 2 == 1
        bg_color = self.NIGHT_BG_COLOR if is_night else self.DAY_BG_COLOR
        fg_color = self.NIGHT_GROUND_COLOR if is_night else self.DAY_GROUND_COLOR
        text_color = self.NIGHT_TEXT_COLOR if is_night else self.DAY_TEXT_COLOR

        self.window.fill(bg_color)
        self._draw_ground(fg_color)
        self._draw_obstacles(obstacles, bg_color)
        self._draw_dino(dino_x, dino_y, dino_width, dino_height, is_ducking)
        self._draw_hud(elapsed_steps, game_speed, is_night, text_color)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _to_screen_y(self, physics_y: float, obj_height: float) -> int:
        """Convert bottom-left physics Y to top-left Pygame screen Y."""
        return int(self.ground_y - physics_y - obj_height)

    def _draw_ground(self, color: tuple[int, int, int]) -> None:
        pygame.draw.line(
            self.window, color, (0, self.ground_y), (self.width, self.ground_y), width=2
        )

    def _draw_dino(
        self, x: float, y: float, width: float, height: float, is_ducking: bool
    ) -> None:
        screen_y = self._to_screen_y(y, height)
        rect = pygame.Rect(int(x), screen_y, int(width), int(height))
        color = self.DINO_DUCK_COLOR if is_ducking else self.DINO_COLOR
        pygame.draw.rect(self.window, color, rect, border_radius=5)

        # Draw a small white eye on the front-top of the dinosaur
        eye_x = int(x + width - 10)
        eye_y = screen_y + 8
        pygame.draw.circle(self.window, (255, 255, 255), (eye_x, eye_y), 3)

    def _draw_obstacles(
        self, obstacles: Sequence[Any], bg_color: tuple[int, int, int]
    ) -> None:
        for obs in obstacles:
            if obs.x > self.width:
                continue

            screen_y = self._to_screen_y(obs.y, obs.height)
            rect = pygame.Rect(int(obs.x), screen_y, int(obs.width), int(obs.height))

            if obs.kind == "cactus":
                pygame.draw.rect(self.window, self.CACTUS_COLOR, rect, border_radius=4)
                # Draw subtle vertical separators if 2 or 3 cacti are grouped together
                single_w = 25
                num_cacti = max(1, round(obs.width / single_w))
                for i in range(1, num_cacti):
                    sep_x = int(obs.x + i * single_w)
                    pygame.draw.line(
                        self.window,
                        bg_color,
                        (sep_x, screen_y),
                        (sep_x, screen_y + int(obs.height)),
                        width=2,
                    )
            else:
                # Bird obstacle
                pygame.draw.rect(self.window, self.BIRD_COLOR, rect, border_radius=6)
                # Draw a small beak on the left side of the bird
                beak_points = [
                    (int(obs.x), screen_y + int(obs.height // 2)),
                    (int(obs.x) - 8, screen_y + int(obs.height // 2) + 4),
                    (int(obs.x), screen_y + int(obs.height // 2) + 8),
                ]
                pygame.draw.polygon(self.window, self.BIRD_COLOR, beak_points)

    def _draw_hud(
        self,
        elapsed_steps: int,
        game_speed: float,
        is_night: bool,
        text_color: tuple[int, int, int],
    ) -> None:
        cycle_label = "NIGHT" if is_night else "DAY"
        hud_text = f"SCORE: {elapsed_steps:05d}   SPEED: {game_speed:4.1f}   MODE: {cycle_label}"
        surface = self.font.render(hud_text, True, text_color)
        self.window.blit(surface, (self.width - surface.get_width() - 20, 15))

    def close(self) -> None:
        if not self.closed:
            pygame.display.quit()
            self.closed = True