"""Train PushBox with the course's tabular Q-learning implementation."""

import argparse
import random

import numpy as np

from deustorl.common import (
    EpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
    evaluate_policy,
    max_policy,
)
from deustorl.qlearning_fixed import QLearning
from dino_game.dino_env import DinoEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="use PyGame")
    args = parser.parse_args()

    seed = 3
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
        n_steps=2_000_000,
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
    evaluate_policy(evaluation_env, algorithm.q_table, max_policy, n_episodes=10, verbose=True)

    evaluation_env.close()


if __name__ == "__main__":
    main()
