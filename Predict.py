from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from v5_env import RectangleReductionEnv
import time
import os
import numpy as np
import random
from map_gen_v4 import MapGen

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
grid_maps = MapGen()
num_robots = 6
field_of_view = 1
grid_map_call = grid_maps.generate_connected_clusters_map(rows=20, cols=20, num_clusters=5, cluster_size_range=(2, 10), min_distance=3)
grid_map = grid_map_call.copy()
free_cells = np.argwhere(grid_map == 0)
random_indices = random.sample(range(len(free_cells)), num_robots)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])
env = RectangleReductionEnv(grid_map, num_robots, robot_positions=robot_positions, field_of_view=1)
env.reset()

#'MultiInputPolicy'
#model = PPO("MultiInputPolicy", env ,verbose=1, tensorboard_log=logdir,device="cuda")
#model = PPO.load("C:\\Users\\Dell-A1\\Desktop\\RL_V2.0\\models\\1736994507\\130000",env)

model = PPO.load("C:/Users/Dell-A1/Desktop/new_models/6robots", env = env)
#model = PPO.load("C:/Users/Dell-A1/Desktop/RL_V2.0/models/1736997665/140000", env = env)
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#print("yes",mean_reward, std_reward)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
#for i in range(1000):
#    action, _states = model.predict(obs, deterministic=True)
#    obs, rewards, dones, info = vec_env.step(action)
    
while True:
    
    #model.learn(total_timesteps=100000)
	model.learn(total_timesteps=100000)

