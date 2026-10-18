"""Play the Chrome Dino environment manually using the keyboard."""

import pygame

from dino_game.dino_env import DinoEnv


def main() -> None:
    env = DinoEnv(render_mode="human", max_steps=5000)
    env.reset()
    env.render()  # Initialize the PyGame window

    running = True
    while running:
        # Read currently held keys for responsive continuous controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = DinoEnv.JUMP
        elif keys[pygame.K_DOWN]:
            action = DinoEnv.DUCK
        else:
            action = DinoEnv.NOOP

        _, _, terminated, truncated, _ = env.step(action)
        env.render()

        # Stop if the user closed the PyGame window
        if env._renderer is None or env._renderer.closed:
            break

        if terminated or truncated:
            print(f"Fin del episodio | Pasos sobrevividos: {env._elapsed_steps}")
            pygame.time.delay(600)
            env.reset()

    env.close()


if __name__ == "__main__":
    main()