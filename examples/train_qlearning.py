"""Train PushBox with the course's tabular Q-learning implementation."""

import argparse
import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deustorl.common import (
    EpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
    evaluate_policy,
    max_policy,
)
from deustorl.qlearning_fixed import QLearning
from dino_game.dino_env import DinoEnv


def evaluate_and_save_scores(env, q_table, n_steps: int, seed: int, csv_path: Path) -> None:
    """Append the ten final episode scores without updating the trained Q-table."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).isoformat()
    rewards = []
    episode_lengths = []
    with csv_path.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["run_id", "n_steps", "seed", "episode", "reward", "steps"]
        )
        if target.tell() == 0:
            writer.writeheader()
        for episode in range(1, 11):
            print(f"Final evaluation episode {episode}/10")
            reward, steps = evaluate_policy(
                env, q_table, max_policy, n_episodes=1, verbose=True
            )
            writer.writerow({
                "run_id": run_id,
                "n_steps": n_steps,
                "seed": seed,
                "episode": episode,
                "reward": reward,
                "steps": int(steps),
            })
            target.flush()
            rewards.append(reward)
            episode_lengths.append(steps)

    print(f"Average reward per episode: {sum(rewards) / len(rewards):.4f}")
    print(f"Average steps per episode: {sum(episode_lengths) / len(episode_lengths):.4f}")
    print(f"Final evaluation scores saved to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="use PyGame")
    args = parser.parse_args()

    seed = 3
    n_steps = 2_000_000
    random.seed(seed)
    np.random.seed(seed)

    training_env = DinoEnv()
    algorithm = QLearning(training_env)
    exploration_policy = LinearDecayEpsilonGreedyPolicy(
        initial_epsilon=1.0, min_epsilon=0.01, decay_steps=1_500_000
    )

    training_env.reset(seed=seed)

    # Fast evaluation environment for periodic Tensorboard logging during training
    eval_env = DinoEnv()

    print("Training Q-learning algorithm...")
    algorithm.learn(
        exploration_policy,
        n_steps=n_steps,
        discount_rate=0.99,
        lr=0.1,
        eval_env=eval_env,
        eval_freq=100_000,
        eval_episodes=5,
    )

    eval_env.close()
    training_env.close()

    print("Evaluating learned policy...")
    render_mode = "human" if args.human else "ansi"
    evaluation_env = DinoEnv(render_mode=render_mode, renderer="pygame")
    csv_path = (
        Path(__file__).resolve().parent.parent
        / "analisis" / "results" / "last10scores" / "scores.csv"
    )
    try:
        evaluate_and_save_scores(evaluation_env, algorithm.q_table, n_steps, seed, csv_path)
    finally:
        evaluation_env.close()


if __name__ == "__main__":
    main()
