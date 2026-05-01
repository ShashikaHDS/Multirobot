"""
Curriculum trainer for the v8 generalized rendezvous policy.

Stages progressively increase map size and robot count. The same policy weights
carry across stages because the observation shapes are fixed (global=coarse_size,
local=crop_size) regardless of underlying map size.

Usage:
    python train_v8_cnn.py --n-envs 4 --steps-per-stage 500000

The curriculum runs once end-to-end. To resume from a saved model:
    python train_v8_cnn.py --resume models_v8/<run>/final_stage1.zip --start-stage 2
"""

import os
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env_v8_generalized import GeneralizedRendezvousEnv, EnvConfig
from feature_extractor_v8 import MultiRobotCNNExtractor


# max_robots is fixed at 6 in every stage so the observation/action shapes
# stay constant -> the policy weights carry across stages. The `max_active`
# knob controls how many of those 6 slots are actually used per episode.
CURRICULUM = [
    dict(map_rows=10, map_cols=10, max_robots=6, min_robots=2, max_active=4,
         max_steps=200, num_clusters=3, cluster_size_range=(2, 5)),
    dict(map_rows=20, map_cols=20, max_robots=6, min_robots=2, max_active=6,
         max_steps=400, num_clusters=5, cluster_size_range=(2, 10)),
    dict(map_rows=30, map_cols=30, max_robots=6, min_robots=3, max_active=6,
         max_steps=600, num_clusters=7, cluster_size_range=(3, 12)),
    dict(map_rows=40, map_cols=40, max_robots=6, min_robots=3, max_active=6,
         max_steps=800, num_clusters=10, cluster_size_range=(3, 15)),
]


def build_cfg(stage_dict, cells_per_robot):
    base = EnvConfig(cells_per_robot=cells_per_robot)
    for k, v in stage_dict.items():
        setattr(base, k, v)
    return base


def make_env(cfg, seed):
    def _thunk():
        env = GeneralizedRendezvousEnv(cfg)
        env.seed(seed)
        return env
    return _thunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--steps-per-stage", type=int, default=1_000_000)
    p.add_argument("--save-dir", default="models_v8")
    p.add_argument("--log-dir", default="logs_v8")
    p.add_argument("--cells-per-robot", type=int, default=2)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=9e-5)
    p.add_argument("--n-steps", type=int, default=2048,
                   help="PPO rollout length per env.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--features-dim", type=int, default=512)
    p.add_argument("--robot-embed-dim", type=int, default=256)
    p.add_argument("--resume", default=None,
                   help="Path to saved PPO .zip to resume from.")
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.run_name or f"v8_{int(time.time())}"
    save_root = os.path.join(args.save_dir, run_name)
    os.makedirs(save_root, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=MultiRobotCNNExtractor,
        features_extractor_kwargs=dict(
            features_dim=args.features_dim,
            robot_embed_dim=args.robot_embed_dim,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = None
    for stage_idx, stage in enumerate(CURRICULUM):
        if stage_idx < args.start_stage:
            continue

        cfg = build_cfg(stage, args.cells_per_robot)
        env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
        env = env_cls([make_env(cfg, seed=stage_idx * 100 + i)
                       for i in range(max(1, args.n_envs))])

        if model is None:
            if args.resume:
                print(f"Resuming from {args.resume}")
                model = PPO.load(args.resume, env=env, device="auto",
                                 tensorboard_log=os.path.join(args.log_dir, run_name))
            else:
                model = PPO(
                    "MultiInputPolicy", env,
                    verbose=1,
                    tensorboard_log=os.path.join(args.log_dir, run_name),
                    ent_coef=args.ent_coef,
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    policy_kwargs=policy_kwargs,
                    device="auto",
                )
        else:
            model.set_env(env)

        ckpt = CheckpointCallback(
            save_freq=max(50_000 // max(1, args.n_envs), 1),
            save_path=save_root,
            name_prefix=f"stage{stage_idx}",
        )
        print(f"\n=== Stage {stage_idx}: {stage} ===")
        model.learn(
            total_timesteps=args.steps_per_stage,
            callback=ckpt,
            reset_num_timesteps=False,
            tb_log_name=f"stage{stage_idx}",
        )
        final_path = os.path.join(save_root, f"final_stage{stage_idx}")
        model.save(final_path)
        print(f"Saved {final_path}.zip")
        env.close()

    print("\nCurriculum complete.")


if __name__ == "__main__":
    main()
