"""Watch a random agent using the ANSI or PyGame renderer."""

import argparse

from dino_game.dino_env import DinoEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="use PyGame")
    args = parser.parse_args()

    render_mode = "human" if args.human else "ansi"
    env = DinoEnv(render_mode=render_mode, renderer="pygame", max_steps=500)
    env.reset(seed=0)
    render(env)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        render(env)

    env.close()


def render(env: DinoEnv) -> None:
    """Render the environment and print the result only in ANSI mode."""
    rendering = env.render()
    if rendering is not None:
        print(rendering)


if __name__ == "__main__":
    main()