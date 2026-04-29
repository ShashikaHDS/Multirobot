import gym
import numpy as np
import random
import pygame
import os
import time
from stable_baselines3 import PPO
from matplotlib import pyplot as plt

pygame.init()

# Colors for visualization
bg_color = (255, 255, 255)
grid_color = pygame.Color("grey")
obs_color = (0, 0, 0)
robot_color = (255, 0, 0)


# MapGen class for dynamic map generation
class MapGen:
    @classmethod
    def generate_connected_clusters_map(cls, rows, cols, num_clusters, cluster_size_range, min_distance):
        grid_map = np.zeros((rows, cols), dtype=int)
        cluster_centers = []

        def is_far_enough(new_center, centers):
            for center in centers:
                if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
                    return False
            return True

        while len(cluster_centers) < num_clusters:
            center_x = np.random.randint(0, rows)
            center_y = np.random.randint(0, cols)
            if is_far_enough((center_x, center_y), cluster_centers):
                cluster_centers.append((center_x, center_y))

        for center_x, center_y in cluster_centers:
            num_ones = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1

            while len(cluster_cells) < num_ones:
                current_x, current_y = random.choice(cluster_cells)
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_x, new_y = current_x + direction[0], current_y + direction[1]

                if 0 <= new_x < rows and 0 <= new_y < cols and grid_map[new_x, new_y] == 0:
                    grid_map[new_x, new_y] = 1
                    cluster_cells.append((new_x, new_y))

        return grid_map


# RectangleReductionEnv Class
class RectangleReductionEnv(gym.Env):
    def __init__(self, rows, cols, num_clusters, cluster_size_range, min_distance, num_robots, field_of_view):
        super(RectangleReductionEnv, self).__init__()

        # Map generation parameters
        self.rows = rows
        self.cols = cols
        self.num_clusters = num_clusters
        self.cluster_size_range = cluster_size_range
        self.min_distance = min_distance

        # Robot parameters
        self.num_robots = num_robots
        self.field_of_view = field_of_view

        # Action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_robots)
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=(rows, cols), dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(rows, cols), shape=(self.num_robots, 2), dtype=np.int32),
        })

        self.grid_map = None
        self.known_map = None
        self.robot_positions = None
        self.max_rectangle_area = None

        # Initialize Pygame display
        self.cell_size = 20
        self.screen = pygame.display.set_mode((cols * self.cell_size, rows * self.cell_size))
        pygame.display.set_caption("Rectangle Reduction Environment")

    def reset(self):
        """Reset environment with a new random map."""
        self.grid_map = MapGen.generate_connected_clusters_map(
            self.rows, self.cols, self.num_clusters, self.cluster_size_range, self.min_distance
        )
        self.known_map = np.zeros_like(self.grid_map)

        free_cells = np.argwhere(self.grid_map == 0)
        random_indices = np.random.choice(len(free_cells), self.num_robots, replace=False)
        self.robot_positions = free_cells[random_indices]

        self.max_rectangle_area = self.calculate_rectangle_area()
        self._render()
        return self._get_observation()

    def step(self, actions):
        """Execute actions and update environment."""
        rewards = 0
        for i, action in enumerate(actions):
            self.robot_positions[i], _ = self._move_robot(self.robot_positions[i], action)
            self._update_known_map(self.robot_positions[i])

        new_area = self.calculate_rectangle_area()
        if new_area <= 9:
            rewards += 10
            done = True
        elif new_area < self.max_rectangle_area:
            rewards += 1
        else:
            rewards -= 1

        self.max_rectangle_area = new_area
        done = new_area <= 9
        self._render()
        return self._get_observation(), rewards, done, {}

    def _move_robot(self, position, action):
        x, y = position
        if action == 0 and x > 0: x -= 1
        if action == 1 and x < self.rows - 1: x += 1
        if action == 2 and y > 0: y -= 1
        if action == 3 and y < self.cols - 1: y += 1
        if self.grid_map[x, y] == 1: return position, position
        return np.array([x, y]), position

    def calculate_rectangle_area(self):
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        width = max(x_coords) - min(x_coords) + 1
        height = max(y_coords) - min(y_coords) + 1
        return width * height

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
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    self.known_map[nx][ny] = self.grid_map[nx][ny]

    def _render(self):
        """Render the environment using Pygame."""
        self.screen.fill(bg_color)
        for x in range(self.rows):
            for y in range(self.cols):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self.grid_map[x, y] == 1:
                    pygame.draw.rect(self.screen, obs_color, rect)
                else:
                    pygame.draw.rect(self.screen, grid_color, rect, 1)

        for robot_pos in self.robot_positions:
            rect = pygame.Rect(robot_pos[1] * self.cell_size, robot_pos[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, robot_color, rect)

        pygame.display.update()


# Main Code
if __name__ == "__main__":
    rows, cols = 20, 20
    num_clusters = 5
    cluster_size_range = (10, 50)
    min_distance = 5
    num_robots = 4
    field_of_view = 1

    env = RectangleReductionEnv(rows, cols, num_clusters, cluster_size_range, min_distance, num_robots, field_of_view)

    # Run the environment
    obs = env.reset()
    done = False
    while not done:
        actions = env.action_space.sample()
        obs, reward, done, _ = env.step(actions)
        print(f"Step Reward: {reward}, Done: {done}")

    print("Simulation Complete")
    pygame.quit()
