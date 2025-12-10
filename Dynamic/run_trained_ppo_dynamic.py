#!/usr/bin/env python3
import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_dynamic import OceanTrashEnv, OceanPhysicsConfig


def make_render_env(grid_size=(16, 16), circle_radius=3):
    """Single env with rendering enabled (for visualization)."""
    def _thunk():
        env = OceanTrashEnv(
            grid_size=grid_size,
            n_trash=1,
            n_robots=6,
            max_steps=400,
            render_mode="human",          # 👈 render on screen
            spawn_prob=0.0,
            placement="center",
            wrap=False,
            grid_lines=True,
            fps=10,
            frozen=False,
            circle_radius=circle_radius,
            physics=OceanPhysicsConfig(enabled=True),
        )
        return env
    return _thunk


def run(args):
    log_dir = args.log_dir

    # Paths must match what you used in training
    model_path = os.path.join(log_dir, "ppo_ocean_circle_final.zip")
    vecnorm_path = os.path.join(log_dir, "vec_normalize.pkl")

    if args.best:
        # optionally use best eval model
        model_path = os.path.join(log_dir, "best_model", "best_model.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize stats not found at: {vecnorm_path}")

    grid_size = (args.grid_h, args.grid_w)
    circle_radius = args.circle_radius

    # -------- 1) Create a single rendering env, vectorized --------
    base_env = DummyVecEnv([make_render_env(grid_size=grid_size, circle_radius=circle_radius)])

    # -------- 2) Load VecNormalize wrapper with saved stats --------
    # This recreates the same normalization used during training.
    vec_env = VecNormalize.load(vecnorm_path, base_env)
    vec_env.training = False         # very important: eval mode
    vec_env.norm_reward = False

    # -------- 3) Load the PPO model --------
    model = PPO.load(model_path, env=vec_env, device="auto")
    print(f"Loaded model from: {model_path}")
    print(f"Loaded VecNormalize stats from: {vecnorm_path}")

    # -------- 4) Run episodes forever (until Ctrl+C) --------
    try:
        while True:
            obs = vec_env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            step = 0

            while not (done or truncated):
                # deterministic=True for cleaner behavior
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)

                # VecEnv returns arrays; single env means index 0
                done = bool(dones[0])
                truncated = bool(infos[0].get("TimeLimit.truncated", False))
                episode_reward += float(reward[0])
                step += 1

            # pull some debug info from the underlying env
            info = infos[0]
            circle_formed = info.get("circle_formed", False)
            mean_radial_error = info.get("mean_radial_error", None)

            msg = f"Episode finished in {step} steps. Return = {episode_reward:.3f}"
            if mean_radial_error is not None:
                msg += f", mean_radial_error = {mean_radial_error:.3f}"
            msg += f", circle_formed = {circle_formed}"
            print(msg)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # unwrap and close underlying env
        vec_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="runs/ppo_ocean_circle",
                        help="Log/model directory (same as training)")
    parser.add_argument("--grid-h", type=int, default=16,
                        help="Grid height (must match training)")
    parser.add_argument("--grid-w", type=int, default=16,
                        help="Grid width (must match training)")
    parser.add_argument("--circle-radius", type=int, default=3,
                        help="Circle radius (must match training)")
    parser.add_argument("--best", action="store_true",
                        help="Use best_model/best_model.zip instead of final")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
