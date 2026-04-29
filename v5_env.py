import gym
import numpy as np
from stable_baselines3 import PPO
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
        self.prev_positions = self.robot_positions.copy()
        self.field_of_view = field_of_view
        self.step_count = 0
        self.map_use_count = 0
        self.max_map_uses = 3                               #no of maps per epi
        self.reward_contribution = [0, 0, 0, 0, 0, 0, 0]

        # Distance tracking
        self.distances_traveled = np.zeros(num_robots)  # Initialize distances for all robots

        self.steps_taken = np.zeros(num_robots, dtype=int)  # Tracks valid steps for each robot

        # Map dimensions
        self.grid_size = self.grid_map.shape

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_robots)

        # Visited cells
        self.visit_map = np.zeros_like(self.grid_map)

        # Observation space
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(self.grid_size), shape=(self.num_robots, 2), dtype=np.int32),
        })

        # Known map and rectangle area
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()
        self.cumulative_reward = 0

    def reset(self):
        self.reward_contribution = [0, 0, 0, 0, 0, 0, 0]
        self.step_count = 0

        """if self.map_use_count >= self.max_map_uses:
            self.map_use_count = 0
            #print(".....New_Map......")
            self.grid_map = grid_maps.generate_connected_clusters_map(
                rows=20, cols=20, num_clusters=5, cluster_size_range=(2, 5), min_distance=3)
        else:
            #print("Used map")
            self.map_use_count += 1"""

        self.grid_map = grid_maps.generate_connected_clusters_map(
                rows=20, cols=20, num_clusters=5, cluster_size_range=(2, 10), min_distance=3)

        free_cells = np.argwhere(self.grid_map == 0)
        random_indices = random.sample(range(len(free_cells)), self.num_robots)
        self.robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])
        self.prev_positions = self.robot_positions.copy()
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area,free_status,bounds = self.calculate_rectangle_area()
        self.cumulative_reward = 0

        # Reset distances traveled
        self.distances_traveled = np.zeros(self.num_robots)

        self.steps_taken = np.zeros(self.num_robots, dtype=int)  # Reset steps counter

        return self._get_observation()

    def step(self, actions):
        if self.step_count==0:
            pass
            #print("00000000000000000000000000000000000000000000000000000000000000000000000000000000")
            #print(f"Initial positions of robots: {self.robot_positions}")
        
        
        self.step_count += 1
        
        #print(f"step count : {self.step_count}")
        reward = 0
        new_positions = []
        cell_size = 20
        width = self.grid_size[1] * cell_size
        height = self.grid_size[0] * cell_size
        screen = pygame.display.set_mode((width, height))
        screen.fill(bg_color)
        

        valid_move=True
        self_collide = False
        obs_collide = False
        smt_wrong = False

        
        
        for y in range(height // cell_size):
            for x in range(width // cell_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, grid_color, rect, 1)
        
        

        for i, action in enumerate(actions):                                                                 
            new_position, pre_pos = self._move_robot(self.robot_positions[i], action)


            if tuple(new_position) in new_positions:                              #each other
                reward = -5
                new_position = self.robot_positions[i].copy()
                valid_move=False
                self_collide = True
                if tuple(new_position) in new_positions:
                    smt_wrong = True



            elif self.grid_map[new_position[0]][new_position[1]] != 0:            #obstacles
                reward = -5
                new_position = self.robot_positions[i].copy()
                obs_collide = True
                

            elif np.array_equal(new_position, self.robot_positions[i]):           #No movement
                pass

            if valid_move:                                         #Increment only if the move is valid
                new_positions.append(tuple(new_position))

            

            # Calculate distance traveled
            distance = np.abs(new_position - self.robot_positions[i]).sum()       # Manhattan distance
            self.distances_traveled[i] += distance            
            
            self.robot_positions[i] = new_position
            self._update_known_map(self.robot_positions[i])

            rect = pygame.Rect(self.robot_positions[i][1] * cell_size, self.robot_positions[i][0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, robot_color, rect)

        
        #pygame display
        obs = np.argwhere(self.known_map > 0)

        
        for data in obs:
            rect = pygame.Rect(data[1] * cell_size, data[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, obs_color, rect)

        

        new_area,free_status,square_bounds = self.calculate_rectangle_area()

        self.draw_square(screen, square_bounds, cell_size)
        pygame.display.update()

        new_min_x = min(self.robot_positions[:, 0])
        new_max_x = max(self.robot_positions[:, 0])
        new_min_y = min(self.robot_positions[:, 1])
        new_max_y = max(self.robot_positions[:, 1])

        prev_min_x = min(self.prev_positions[:, 0])
        prev_max_x = max(self.prev_positions[:, 0])
        prev_min_y = min(self.prev_positions[:, 1])
        prev_max_y = max(self.prev_positions[:, 1])

        center_previous=self.calculate_rectangle_centre_point(prev_max_x,prev_min_x,prev_max_y,prev_min_y)
        center_new=self.calculate_rectangle_centre_point(prev_max_x,prev_min_x,prev_max_y,prev_min_y)
        if self.step_count==1:
            pass

        '''
        print(f"robot_positions {self.robot_positions}")
        print(f"prev_positions {self.prev_positions}")
        print("----------------------")
        print(new_min_x, new_max_x)
        print(prev_min_x, prev_max_x)
        print(new_min_y, new_max_y)
        print(prev_min_y, prev_max_y)
        '''

        if new_area <= 16 and free_status:

            #print("goal", self.robot_positions, self_collide, obs_collide, smt_wrong)                                                  #converge
            print("---Goal---")
            reward = 1000
            #time.sleep(1)
            

        elif new_area < self.max_rectangle_area:
            reward = 20 + self.max_rectangle_area - new_area
        elif new_area >= self.max_rectangle_area:
            reward = -0.5
        
        else:
            pass
            #print(f"Distances traveled by each robot: {self.distances_traveled}")
            #print(f"Total steps taken by each robot in this episode: {self.steps_taken}")

        done = new_area <= 16 and free_status
       
        self.max_rectangle_area = new_area
        self.cumulative_reward += reward
        self.prev_positions = self.robot_positions.copy()

        return self._get_observation(), reward, done, {}
    def is_square_free(self, min_x, max_x, min_y, max_y):
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self.grid_map[x, y] != 0:  # 0 represents free cells
                    return False
        return True

    def _move_robot(self, position, action):
        x, y = position.copy()
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:
            y += 1
        return np.array([x, y]), position
    
    def draw_square(self, screen, bounds, cell_size):
        min_x, max_x, min_y, max_y = bounds
        rect_x = min_y * cell_size  # Convert grid coordinates to pixel coordinates
        rect_y = min_x * cell_size
        rect_width = (max_y - min_y + 1) * cell_size
        rect_height = (max_x - min_x + 1) * cell_size
        square_color = (0, 255, 0)  # Green for valid square

        pygame.draw.rect(screen, square_color, (rect_x, rect_y, rect_width, rect_height), 2)  # Thickness = 2


    def calculate_rectangle_area(self):
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Ensure it's a square by enforcing equal side lengths
        side_length = max(max_x - min_x, max_y - min_y) + 1
        square_min_x = min_x
        square_max_x = square_min_x + side_length - 1
        square_min_y = min_y
        square_max_y = square_min_y + side_length - 1

        # Check if all cells inside the square are free
        free = True
        if square_max_x >= self.grid_size[0] or square_max_y >= self.grid_size[1]:
            free = False
        else:
            for x in range(square_min_x, square_max_x + 1):
                for y in range(square_min_y, square_max_y + 1):
                    if self.grid_map[x, y] != 0:  # Non-free cell
                        free  = False

        # Valid square area
        return side_length ** 2, free , (square_min_x, square_max_x, square_min_y, square_max_y)

    def calculate_rectangle_centre_point(self,prev_max_x,prev_min_x,prev_max_y,prev_min_y):
        x_center=(prev_max_x-prev_min_x)/2
        y_center=(prev_max_y-prev_min_y)/2
        return [x_center,y_center]


    def _get_observation(self):
        return {
            "known_map": self.known_map.copy(),
            "robot_positions": self.robot_positions.copy(),
        }

    def _update_known_map(self, position):
        x, y = position
        for dx in range(-self.field_of_view, self.field_of_view + 1):
            for dy in range(-self.field_of_view, self.field_of_view + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_map[nx, ny] = self.grid_map[nx, ny]

grid_maps = MapGen()

'''
num_robots = 4
field_of_view = 1
grid_map_call = grid_maps.generate_connected_clusters_map(rows=20, cols=20, num_clusters=5, cluster_size_range=(2, 10), min_distance=3)
grid_map = grid_map_call.copy()
free_cells = np.argwhere(grid_map == 0)
random_indices = random.sample(range(len(free_cells)), num_robots)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

env = RectangleReductionEnv(grid_map, num_robots, robot_positions=robot_positions, field_of_view=1)

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

print("Training started...")
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir,ent_coef=0.05, device="cuda", learning_rate=3e-4)
TIMESTEPS = 10000
iters=0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    #model.save("loading/rectangle_reduction_model_PPO_2025_01_15_v5.0")
    print("Model saved successfully!")
'''