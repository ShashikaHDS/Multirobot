import gym
import numpy as np
from stable_baselines3 import PPO,A2C
import random
import pygame
import os
import time
from map_gen_v4 import MapGen

pygame.init()

# Colors
bg_color = (255, 255, 255)
grid_color = pygame.Color("grey")
obs_color = (0, 0, 0)
robot_color = (255, 0, 0)

class RectangleReductionEnv(gym.Env):
    def __init__(self, grid_map, num_robots, robot_positions, field_of_view):
        super(RectangleReductionEnv, self).__init__()
        
        # Initialize environment variables
        self.grid_map = np.array(grid_map)
        self.num_robots = num_robots
        self.robot_positions = np.array(robot_positions)
        self.prev_positions = np.zeros((num_robots, 2))
        self.field_of_view = field_of_view
        self.step_count=0
        self.reward_contribution=[0,0,0,0,0,0,0]


        # Define the map array for training
        self.maps = [[], [], [], []]

        # Initialize cumulative reward
        self.cumulative_reward = 0  

        # Map dimensions
        self.grid_size = self.grid_map.shape

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_robots)


        

        #visited cells
        self.visit_map = np.zeros_like(self.grid_map)

        # Observation space: Known map and robot positions
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(self.grid_size), shape=(self.num_robots, 2), dtype=np.int32),
        })

        # Initialize known map and rectangle area
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()

    def no_of_robots(self):
        num_robots=(3,4)
        num_robots=np.random.randint(num_robots[0], num_robots[1])
        return num_robots

    def reset(self):
        self.reward_contribution=[0,0,0,0,0,0,0]
        #changing no of robots
        self.step_count=0
        #print("No of Robots 1: "+str(self.num_robots))
        """Reset environment to initial state."""
        self.grid_map = grid_maps.generate_connected_clusters_map(
            rows=20, cols=20, num_clusters=3, cluster_size_range=(5, 20), min_distance=5)
        #num_robots=(4,6)
        #num_robots_new=np.random.randint(num_robots[0], num_robots[1] + 1)

        #self.num_robots=num_robots_new

        self.cumulative_reward = 0
        #print("no of new robots 2: "+ str(num_robots))
        # Random robot positions for each reset
        free_cells = np.argwhere(self.grid_map == 0)
        random_indices = random.sample(range(len(free_cells)), self.num_robots)
        self.robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()
        return self._get_observation()

    def step(self, actions):
        self.step_count+=1
        #print("step count: "+str(self.step_count))
        """Execute actions and update the environment."""
        reward = 0
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
                reward -= 2  # Penalize for collision
                new_position = self.robot_positions[i]  # Revert position
                
            # Collision with obstacles
            elif self.grid_map[new_position[0]][new_position[1]] != 0:
                #print(new_position,self.robot_positions[i],self.grid_map[new_position[0]][new_position[1]])

                reward -=30
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

        if self.step_count<=300000:
        # Reward calculation

            # Target area reached (3x3 rectangle)
            if new_area <= 16:  
                reward += 20000
                self.reward_contribution[0]+=20000

            elif new_area <= 25:
                reward += 4000
                self.reward_contribution[1]+=4000      

            # Reward for reducing rectangle area
            elif self.max_rectangle_area<=16 and new_area>16:
                reward-=100
                self.reward_contribution[2]-=100  

            elif new_area < self.max_rectangle_area:
                reward +=(200+ 5 * (self.max_rectangle_area - new_area))  # Reward for convergence
                self.reward_contribution[3]+=(200+ 5 * (self.max_rectangle_area - new_area) )

            elif new_area > self.max_rectangle_area:
                reward -= (2000+(5 * abs(self.max_rectangle_area - new_area)))  # Penalty for moving away
                self.reward_contribution[4]-= (200+(2 * abs(self.max_rectangle_area - new_area)))

            # Reward for proximity to goal

            
            # Discourage spreading out
            spread = max(self.robot_positions[:, 0]) - min(self.robot_positions[:, 0]) + max(self.robot_positions[:, 1]) - min(
                self.robot_positions[:, 1])
            if spread > self.max_rectangle_area:
                reward -= 100
                self.reward_contribution[5]-= 100

            # Update visit map and penalize revisits
            for i, position in enumerate(self.robot_positions):
                self.visit_map[position[0], position[1]] += 1
                if(self.visit_map[position[0],position[1]]>4):
                    revisit_penalty = -self.visit_map[position[0], position[1]]
                    reward += (int(revisit_penalty*0.1))
                    self.reward_contribution[6]+= (int(revisit_penalty*0.1))
           

            done = new_area <= 16
            if done:
                 
                print("---------------------Goal reached!-----------------------")
                print(self.reward_contribution)
                print("cumulative_reward : " +str(self.cumulative_reward))
                print("Step count for the episode : "+str(self.step_count))
        
        else:
            print(" Limit exeded................. !")
            reward-=5
            done=True

        self.max_rectangle_area = new_area
        self.cumulative_reward += reward
        #print(self.cumulative_reward)
        # Check if optimal rectangle is achieved

        #time.sleep(1)
        return self._get_observation(), reward, done, {}

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
        return np.array([x, y]), position

    def calculate_rectangle_area(self):
        """Calculate the rectangle area spanned by all robots."""
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        width = max(x_coords) - min(x_coords) + 1
        height = max(y_coords) - min(y_coords) + 1
        return width * height

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
                    self.known_map[nx][ny] = self.grid_map[nx][ny]

grid_maps = MapGen()

#no of robots
#num_robots=(3,4)
#num_robots=np.random.randint(num_robots[0], num_robots[1])

num_robots = 4


##############
field_of_view = 1
grid_map = grid_maps.generate_connected_clusters_map(rows=20, cols=20, num_clusters=5, cluster_size_range=(21, 30), min_distance=5)
grid_map = np.array(grid_map)
free_cells = np.argwhere(grid_map == 0)
random_indices = random.sample(range(len(free_cells)), num_robots)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

env = RectangleReductionEnv(grid_map, num_robots, robot_positions=robot_positions, field_of_view=1)

# Training setup
print("Initializing environment...")
obs = env.reset()

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

print("Training started...")
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda",learning_rate=3e-4)
model.learn(total_timesteps=2000000)

# Save the model
model.save("loading/rectangle_reduction_model_PPO_v2_12_28_6_14")
print("Model saved successfully!")
