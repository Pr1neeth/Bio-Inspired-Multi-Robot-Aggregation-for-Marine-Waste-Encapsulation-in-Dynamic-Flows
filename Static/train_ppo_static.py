#!/usr/bin/env python3
import os
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from env_static import OceanTrashEnv, OceanPhysicsConfig  # <-- change if your file is named differently


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------
def make_env(grid_size=(16, 16), circle_radius=3):
    """Factory for a single training/eval environment (no rendering)."""
    def _thunk():
        env = OceanTrashEnv(
            grid_size=grid_size,
            n_trash=1,
            n_robots=6,
            max_steps=400,
            render_mode=None,          # no rendering during training
            spawn_prob=0.0,
            placement="center",
            wrap=False,
            grid_lines=False,
            fps=10,
            frozen=False,
            circle_radius=circle_radius,
            physics=OceanPhysicsConfig(enabled=False),
        )
        return env
    return _thunk


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train(args):
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    grid_size = (args.grid_h, args.grid_w)
    circle_radius = args.circle_radius

    # ---------------- Vectorized envs ----------------
    # Training env
    train_env = make_vec_env(
        make_env(grid_size=grid_size, circle_radius=circle_radius),
        n_envs=args.vec,
        seed=args.seed,
    )
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,   # keep reward scale as-is
        clip_obs=10.0,
    )

    # Evaluation env (shares normalization stats with training env)
    eval_env = make_vec_env(
        make_env(grid_size=grid_size, circle_radius=circle_radius),
        n_envs=1,
        seed=args.seed + 100,
    )
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    # Share obs normalization stats
    eval_env.obs_rms = train_env.obs_rms

    # ---------------- PPO model ----------------
    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,             # per env → total batch = n_steps * n_envs
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda" if args.cuda else "auto",
    )

    # ---------------- Evaluation callback ----------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=max(1, args.eval_freq // args.vec),  # in env steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(
        f"Training PPO for {args.timesteps} timesteps "
        f"with {args.vec} parallel envs (seed={args.seed})..."
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # ---------------- Save model + normalization ----------------
    model_path = os.path.join(log_dir, "ppo_ocean_circle_final")
    vecnorm_path = os.path.join(log_dir, "vec_normalize.pkl")

    model.save(model_path)
    train_env.save(vecnorm_path)

    print(f"\nSaved final model to: {model_path}")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")
    print(f"Best eval model (if any) in: {os.path.join(log_dir, 'best_model')}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--vec", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--log-dir", type=str, default="runs/ppo_ocean_circle",
                        help="Directory for logs/models")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="Eval frequency in env steps")
    parser.add_argument("--cuda", action="store_true",
                        help="Force use of CUDA if available")

    # Env-specific
    parser.add_argument("--grid-h", type=int, default=16,
                        help="Grid height")
    parser.add_argument("--grid-w", type=int, default=16,
                        help="Grid width")
    parser.add_argument("--circle-radius", type=int, default=3,
                        help="Target ring radius around trash")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
