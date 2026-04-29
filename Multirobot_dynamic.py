import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RectangleReductionEnv(gym.Env):
    def __init__(self, grid_map, num_robots, robot_positions, field_of_view, num_dynamic_obstacles):
        super(RectangleReductionEnv, self).__init__()

        # Initialize environment variables
        self.grid_map = np.array(grid_map)
        self.num_robots = num_robots
        self.robot_positions = np.array(robot_positions)
        self.field_of_view = field_of_view
        self.num_dynamic_obstacles = num_dynamic_obstacles

        # Map dimensions
        self.grid_size = self.grid_map.shape

        # Initialize dynamic obstacles
        self.dynamic_obstacles = self._initialize_dynamic_obstacles()

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = spaces.MultiDiscrete([5] * self.num_robots)

        # Observation space: Known map and robot positions
        self.observation_space = spaces.Dict({
            "known_map": spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": spaces.Box(0, max(self.grid_size), shape=(self.num_robots, 2), dtype=np.int32),
        })

        # Initialize known map and rectangle area
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()

    def _initialize_dynamic_obstacles(self):
        """Initialize dynamic obstacles with random positions and directions."""
        obstacles = []
        for _ in range(self.num_dynamic_obstacles):
            while True:
                position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
                if self.grid_map[tuple(position)] == 0:  # Ensure obstacle starts on a free cell
                    break
            direction = np.random.choice([-1, 1], size=2)
            obstacles.append({"position": position, "direction": direction})
        return obstacles

    def reset(self):
        """Reset environment to initial state."""
        self.known_map = np.zeros_like(self.grid_map)
        self.max_rectangle_area = self.calculate_rectangle_area()
        self.dynamic_obstacles = self._initialize_dynamic_obstacles()
        return self._get_observation()

    def step(self, actions):
        """Execute actions and update the environment."""
        rewards = 0
        for i, action in enumerate(actions):
            self.robot_positions[i] = self._move_robot(self.robot_positions[i], action)
            self._update_known_map(self.robot_positions[i])

        # Update dynamic obstacles
        self._move_dynamic_obstacles()

        # Calculate the new rectangle area
        new_area = self.calculate_rectangle_area()

        # Reward calculation
        if new_area == 9:  # Target area reached (3x3 rectangle)
            rewards += 10
        elif new_area < self.max_rectangle_area:
            rewards += 1  # Reward for reducing the area
        else:
            rewards -= 1  # Penalize for increasing the area

        self.max_rectangle_area = new_area

        # Check if optimal rectangle is achieved
        done = new_area == 9

        return self._get_observation(), rewards, done, {}

    def _get_observation(self):
        """Return the current known map and robot positions."""
        return {
            "known_map": self.known_map.copy(),
            "robot_positions": self.robot_positions.copy(),
        }

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
        return np.array([x, y]) if self.grid_map[x, y] == 0 else position

    def _update_known_map(self, position):
        """Update known map based on robot's field of view."""
        x, y = position
        for dx in range(-self.field_of_view, self.field_of_view + 1):
            for dy in range(-self.field_of_view, self.field_of_view + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_map[nx, ny] = self.grid_map[nx, ny]

    def _move_dynamic_obstacles(self):
        """Move dynamic obstacles and update their positions."""
        for obstacle in self.dynamic_obstacles:
            pos = obstacle["position"]
            direction = obstacle["direction"]

            # Calculate new position
            new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]

            # Check if new position is valid
            if not (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]) or self.grid_map[new_x, new_y] != 0:
                # Reverse direction
                obstacle["direction"] = -obstacle["direction"]
            else:
                # Update position
                obstacle["position"] = [new_x, new_y]

    def calculate_rectangle_area(self):
        """Calculate the rectangle area spanned by all robots."""
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        width = max(x_coords) - min(x_coords) + 1
        height = max(y_coords) - min(y_coords) + 1
        return width * height


# Test Environment
grid_map = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

robot_positions = [[0, 0], [4, 4]]

env = RectangleReductionEnv(grid_map, num_robots=2, robot_positions=robot_positions, field_of_view=1, num_dynamic_obstacles=3)

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
vec_env = make_vec_env(lambda: env, n_envs=4)
model = PPO("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("rectangle_reduction_with_dynamic_obstacles")
