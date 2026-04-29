from stable_baselines3 import PPO
from rl_mltirobot_pygame import RectangleReductionEnv
import time
import os


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = RectangleReductionEnv(rows=20, cols=20, num_clusters=5, cluster_size_range=(21, 30), min_distance=5)
env.reset()
#'MultiInputPolicy'
#model = PPO("MultiInputPolicy", env ,verbose=1, tensorboard_log=logdir,device="cuda")
model = PPO.load("D:\MNEED\Thesis\RL\rectangle_reduction_model_01",env)
TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")


