import gym
#from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import random
import pygame
import os
import time
from map_gen_v4 import MapGen
import matplotlib



pygame.init()

bg_color = (255,255,255)
grid_color = pygame.Color("grey")
obs_color = (0,0,0)
robot_color = (255,0,0)

class RectangleReductionEnv(gym.Env):
    def __init__(self, grid_map, num_robots, robot_positions, field_of_view):
        super(RectangleReductionEnv, self).__init__()

        # Initialize environment variables
        self.grid_map = np.array(grid_map)
        self.num_robots = num_robots
        self.robot_positions = np.array(robot_positions)
        self.prev_positions = np.zeros((num_robots,2))
        self.field_of_view = field_of_view
        
        #Define the map array for training
        self.maps = [[],[],[],[]]

        # Cumulative reward for the episode
        self.cumulative_reward = 0  # Initialize cumulative reward

        # Map dimensions
        self.grid_size = self.grid_map.shape
        #print(type(self.grid_size),len(self.grid_size))

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_robots)

        # Observation space: Known map and robot positions
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(self.grid_size), shape=(self.num_robots, 2), dtype=np.int32),
        })

        # Initialize known map and rectangle area
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()
        self.count=0
    def reset(self):
        """Reset environment to initial state."""
        self.grid_map = grid_map

        print("Training Map Size:", self.grid_map.shape)
        '''''
        #no random postion for each
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()
        self.count+=1
        print("count : "+ str(self.count))
        return self._get_observation() #optional
        '''    
        # Reset cumulative reward for the new episode
        self.cumulative_reward = 0

        #random robot positions for each reset
        free_cells = np.argwhere(self.grid_map == 0)
        #random_indices = random.sample(range(len(free_cells)), self.num_robots)
        #self.robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])
        self.robot_positions=robot_positions
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()
        return self._get_observation()

    def step(self, actions):
        """Execute actions and update the environment."""
        rewards = 0
        new_positions = []
        cell_size = 20
        width = self.grid_size[1] * cell_size
        height = self.grid_size[0] * cell_size
        screen = pygame.display.set_mode((width, height))
        screen.fill(bg_color)

        # Draw grid
        for y in range(height // cell_size):
            for x in range(width // cell_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, grid_color, rect, 1)

        for i, action in enumerate(actions):
            #print("i: "+str(i)+", action: "+str(action)+", actions: "+ str(actions))
            
            new_position, previous_position = self._move_robot(self.robot_positions[i], action)
            #print(new_position)

            # Collision with other robots
            if tuple(new_position) in new_positions:
                rewards -= 2  # Penalize for collision
                new_position = self.robot_positions[i]  # Revert position
                
            # Collision with obstacles
            elif self.grid_map[new_position[0]][new_position[1]] != 0:
                #print(new_position,self.robot_positions[i],self.grid_map[new_position[0]][new_position[1]])

                rewards -= 1
                new_position = self.robot_positions[i]
               
            else:
                new_positions.append(tuple(new_position))

            self.robot_positions[i] = new_position
            self._update_known_map(self.robot_positions[i])
            # Draw robot
            rect = pygame.Rect(self.robot_positions[i][1] * cell_size, self.robot_positions[i][0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, robot_color, rect)

        # Draw explored map
        obs = np.argwhere(self.known_map > 0)
        for data in obs:
            rect = pygame.Rect(data[1] * cell_size, data[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, obs_color, rect)

        pygame.display.update()

        # Calculate the new rectangle area
        new_area = self.calculate_rectangle_area()
        
        # Reward calculation
        if new_area <= 9:  # Target area reached (3x3 rectangle)
            rewards += 30
            
        elif new_area < self.max_rectangle_area:
            rewards += 10  # Reward for reducing the area
        else:
            rewards -= 1  # Penalize for increasing the area
        
        self.max_rectangle_area = new_area
        self.cumulative_reward += rewards
        #print(self.cumulative_reward)
        # Check if optimal rectangle is achieved
        done = new_area <= 9
        if done:
            print("---------------------Goal reached!-----------------------")
        #time.sleep(1)
        return self._get_observation(), rewards, done, {}
    
    def _move_robot(self, position, action):
        """Move robot according to action."""
        x, y = position
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:  # Right
            y += 1
        # Ensure robots do not move into obstacles
        return np.array([x, y]),position


    def calculate_rectangle_area(self):
        """Calculate the rectangle area spanned by all robots."""
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        width = max(x_coords) - min(x_coords) + 1
        #if width==0:
          #  width=1
        
        height = max(y_coords) - min(y_coords) + 1
        #if height==0:
           # height=1
        return (width * height)

    def _get_observation(self):
        """Return the current known map and robot positions."""
        return {
            "known_map": self.known_map.copy(),
            "robot_positions": self.robot_positions.copy(),
        }
    

    def _update_known_map(self, position):
        """Update known map based on robot's field of view."""
        x, y = position
        for dx in range(-self.field_of_view, self.field_of_view + 1):
            for dy in range(-self.field_of_view, self.field_of_view + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    #print(nx,ny, self.known_map, self.grid_map)
                    self.known_map[nx][ny] = self.grid_map[nx][ny]

#grid_maps= MapGen()

    

grid_map=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

'''''
grid_maps = [[
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
],[
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]]

'''
num_robots=4
field_of_view=1
#rows=20, cols=20, num_clusters=5, cluster_size_range=(21, 30), min_distance=5
#grid_map = grid_maps.generate_connected_clusters_map(rows=20, cols=20, num_clusters=5, cluster_size_range=(21, 30), min_distance=5)

grid_map = np.array(grid_map)

robot_positions = np.array([[0, 0], [33, 33],[0,33],[33,0]])
print(robot_positions)
env = RectangleReductionEnv(grid_map, num_robots, robot_positions=robot_positions, field_of_view=1)

# Test the environment
obs = env.reset()

for _ in range(10):
    actions = env.action_space.sample()
    obs, reward, done, info = env.step(actions)
    print(f"Step Reward: {reward}, Done: {done}")
    if done:
        print("Optimal rectangle reached!")
        break

# Train using Stable Baselines3
#vec_env = make_env(lambda: env, n_envs=4)

print("Training started")
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
model = PPO("MultiInputPolicy", env, verbose=1,tensorboard_log=logdir,device="cuda")
model.learn(total_timesteps=10000)

# Save the model
#model.save("loading/rectangle_reduction_model_PPO_02c")
#print("Model saved successfully!")