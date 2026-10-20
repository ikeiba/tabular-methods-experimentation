"""Full hyperparameter optimization and environment comparison for Chrome Dino.

Based on lab2/experiments_2/experiment_2.10.py.
Compares Sarsa, Expected Sarsa, and Q-Learning (using the _fixed implementations)
across two distinct environments using Optuna:
  1. Constant Speed (Easier, stationary, 1,620 states)
  2. Variable Speed (Harder, dynamic, 4,860 states)"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import optuna
from optuna.samplers import TPESampler

from deustorl.common import (
    EpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
    evaluate_policy,
    max_policy,
)
from deustorl.expected_sarsa_fixed import ExpectedSarsa
from deustorl.qlearning_fixed import QLearning
from deustorl.sarsa_fixed import Sarsa
from dino_game.dino_env import DinoEnv


def create_algorithm(algo_name: str, env: DinoEnv) -> Sarsa | ExpectedSarsa | QLearning:
    """Instantiate the requested RL algorithm from the fixed modules."""
    if algo_name == "sarsa":
        return Sarsa(env)
    elif algo_name == "esarsa":
        return ExpectedSarsa(env)
    elif algo_name == "qlearning":
        return QLearning(env)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def make_objective(constant_speed: bool, n_steps: int, seed: int):
    """Factory to create an Optuna objective function for a given speed mode."""

    def objective(trial: optuna.Trial) -> float:
        algo_name = trial.suggest_categorical("algo_name", ["sarsa", "esarsa", "qlearning"])
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
        lr_decay = trial.suggest_float("learning_rate_decay", 0.90, 1.0, step=0.01)
        lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay", [100, 500, 1_000, 5_000])
        discount_rate = trial.suggest_float("discount_rate", 0.85, 0.999, step=0.01)
        min_epsilon = trial.suggest_float("min_epsilon", 0.005, 0.05, step=0.005)
        decay_ratio = trial.suggest_float("decay_ratio", 0.50, 0.85, step=0.05)

        env = DinoEnv(constant_speed=constant_speed)
        env.reset(seed=seed + trial.number)

        eval_env = DinoEnv(constant_speed=constant_speed)

        algo = create_algorithm(algo_name, env)
        decay_steps = int(n_steps * decay_ratio)
        policy = LinearDecayEpsilonGreedyPolicy(
            initial_epsilon=1.0, min_epsilon=min_epsilon, decay_steps=decay_steps
        )

        eval_freq = max(10_000, n_steps // 10)  # Evaluate 10 times across training
        algo.learn(
            policy,
            n_steps=n_steps,
            discount_rate=discount_rate,
            lr=lr,
            lrdecay=lr_decay,
            n_episodes_decay=lr_episodes_decay,
            verbose=False,
            eval_env=eval_env,
            eval_freq=eval_freq,
            eval_episodes=5,
            tb_log_dir=os.path.join(
                "logs", "Full_experiment_optuna_trial", f"{trial.number}_{algo_name}"
            ),
        )

        avg_reward, avg_steps = evaluate_policy(
            eval_env, algo.q_table, max_policy, n_episodes=20, verbose=False
        )
        eval_env.close()
        env.close()

        trial.set_user_attr("avg_steps", avg_steps)
        trial.set_user_attr("num_states", int(env.observation_space.n))
        return float(avg_reward)

    return objective


def run_study(
    study_name: str,
    constant_speed: bool,
    n_trials: int,
    n_steps: int,
    seed: int,
    storage_file: str,
) -> optuna.Study:
    """Run an independent Optuna study for one environment mode."""
    env_mode_label = (
        "CONSTANT SPEED (Stationary, 1,620 states)"
        if constant_speed
        else "VARIABLE SPEED (Dynamic, 4,860 states)"
    )

    print("\n" + "=" * 70)
    print(f"STARTING STUDY: {study_name}")
    print(f"Environment Mode: {env_mode_label}")
    print(f"Trials: {n_trials} | Steps per trial: {n_steps:,}")
    print("=" * 70)

    study_dir = os.path.join("optuna", study_name)
    os.makedirs(study_dir, exist_ok=True)

    tpe_sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        sampler=tpe_sampler,
        direction="maximize",
        study_name=study_name,
        storage=storage_file,
        load_if_exists=True,
    )

    objective_fn = make_objective(constant_speed=constant_speed, n_steps=n_steps, seed=seed)
    study.optimize(objective_fn, n_trials=n_trials)

    best_trial = study.best_trial
    print("\n" + "-" * 70)
    print(f"STUDY FINISHED: {study_name}")
    print(f"Best Trial #{best_trial.number} | Best Avg Reward: {best_trial.value:.2f}")
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    print("-" * 70)

    # Save best trial data to JSON
    best_trial_data = {
        "study_name": study_name,
        "constant_speed": constant_speed,
        "trial_number": best_trial.number,
        "best_value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }
    json_path = os.path.join(study_dir, "best_trial.json")
    with open(json_path, "w") as f:
        json.dump(best_trial_data, f, indent=4)
    print(f"Saved best trial to: {json_path}")

    # Generate visualization figures safely
    visualizations = [
        ("optimization_history.html", optuna.visualization.plot_optimization_history),
        ("contour.html", optuna.visualization.plot_contour),
        ("slice.html", optuna.visualization.plot_slice),
        ("param_importances.html", optuna.visualization.plot_param_importances),
    ]

    for filename, plot_func in visualizations:
        try:
            fig = plot_func(study)
            target_path = os.path.join(study_dir, filename)
            fig.write_html(target_path)
            print(f"Saved plot: {target_path}")
        except Exception as e:
            print(f"Could not generate plot {filename}: {e}")

    return study


def print_comparison_summary(
    studies: dict[str, optuna.Study],
    output_path: str = "optuna/comparison_summary.json",
) -> None:
    """Print and save a comparative table summarizing the findings across studies."""
    print("\n" + "=" * 80)
    print("COMPARATIVE STUDY SUMMARY: CONSTANT SPEED vs VARIABLE SPEED")
    print("=" * 80)
    print(
        f"{'Study':<22} | {'Algorithm':<10} | {'Reward':<8} | {'Steps':<7} | "
        f"{'LR':<8} | {'Gamma':<6} | {'Eps':<6}"
    )
    print("-" * 80)

    summary_data = {}

    for name, study in studies.items():
        best = study.best_trial
        algo = best.params.get("algo_name", "N/A")
        lr = best.params.get("learning_rate", 0.0)
        gamma = best.params.get("discount_rate", 0.0)
        eps = best.params.get("min_epsilon", best.params.get("epsilon", 0.0))
        steps = best.user_attrs.get("avg_steps", 0.0)

        print(
            f"{name:<22} | {algo:<10} | {best.value:<8.2f} | {steps:<7.1f} | "
            f"{lr:<8.4f} | {gamma:<6.3f} | {eps:<6.2f}"
        )

        summary_data[name] = {
            "best_trial_number": best.number,
            "best_reward": best.value,
            "best_params": best.params,
            "user_attrs": best.user_attrs,
        }

    print("=" * 80 + "\n")

    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Comparison summary saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full hyperparameter optimization and comparison for Chrome Dino RL."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "constant", "variable"],
        default="both",
        help="Which environment study to run: 'constant' (1620 states), 'variable' (4860 states), or 'both' (default)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna trials per study (default: 15)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10_000_000,
        help="Training steps per trial (default: 500,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs("optuna", exist_ok=True)
    storage_file = "sqlite:///optuna/optuna.db"

    studies_run = {}

    # Study 1: Constant Speed (Easy / Stationary)
    if args.mode in ["both", "constant"]:
        study_constant = run_study(
            study_name="dino_constant_speed",
            constant_speed=True,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
            seed=args.seed,
            storage_file=storage_file,
        )
        studies_run["dino_constant_speed"] = study_constant

    # Study 2: Variable Speed (Hard / Non-stationary)
    if args.mode in ["both", "variable"]:
        study_variable = run_study(
            study_name="dino_variable_speed",
            constant_speed=False,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
            seed=args.seed + 100,
            storage_file=storage_file,
        )
        studies_run["dino_variable_speed"] = study_variable

    # If both were executed, print and save the comparative summary
    if len(studies_run) > 1:
        print_comparison_summary(studies_run)


if __name__ == "__main__":
    main()
