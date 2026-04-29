#Multi agent RL

from pettingzoo.utils.env import ParallelEnv
import gym
import numpy as np
import random
from ray import tune
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env


# Map Generation
class MapGen:
    @staticmethod
    def generate_connected_clusters_map(rows, cols, num_clusters, cluster_size_range, min_distance):
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
            cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1

            while len(cluster_cells) < cluster_size:
                current_x, current_y = random.choice(cluster_cells)
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(directions)

                for direction in directions:
                    new_x = current_x + direction[0]
                    new_y = current_y + direction[1]

                    if 0 <= new_x < rows and 0 <= new_y < cols and grid_map[new_x, new_y] == 0:
                        grid_map[new_x, new_y] = 1
                        cluster_cells.append((new_x, new_y))
                        break
        return grid_map


# Multi-Agent Environment
class RectangleReductionPettingZooEnv(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_map, num_robots, field_of_view):
        super().__init__()
        self.grid_map = grid_map
        self.num_robots = num_robots
        self.robot_positions = self._initialize_robot_positions()
        self.field_of_view = field_of_view

        self.agents = [f"robot_{i}" for i in range(self.num_robots)]
        self.action_spaces = {agent: gym.spaces.Discrete(5) for agent in self.agents}
        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
                "robot_position": gym.spaces.Box(0, max(self.grid_map.shape), shape=(2,), dtype=np.int32),
            }) for agent in self.agents
        }

        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.done_agents = set()

    def _initialize_robot_positions(self):
        free_cells = np.argwhere(self.grid_map == 0)
        selected_indices = random.sample(range(len(free_cells)), self.num_robots)
        return [tuple(free_cells[i]) for i in selected_indices]

    def reset(self):
        self.robot_positions = self._initialize_robot_positions()
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.done_agents = set()
        return {agent: self._get_observation(agent) for agent in self.agents}

    def step(self, actions):
        rewards = {}
        dones = {}
        infos = {}
        for agent, action in actions.items():
            rewards[agent] = self._move_robot(agent, action)
            dones[agent] = False
            infos[agent] = {}

        # Check if all agents have reached the goal
        all_done = all(dones.values())
        if all_done:
            for agent in self.agents:
                dones[agent] = True
        return (
            {agent: self._get_observation(agent) for agent in self.agents},
            rewards,
            dones,
            infos,
        )

    def _get_observation(self, agent):
        idx = int(agent.split("_")[1])
        return {
            "known_map": self.grid_map.copy(),
            "robot_position": np.array(self.robot_positions[idx]),
        }

    def _move_robot(self, agent, action):
        idx = int(agent.split("_")[1])
        x, y = self.robot_positions[idx]

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_map.shape[0] - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_map.shape[1] - 1:
            y += 1

        if self.grid_map[x, y] == 1:
            return -10  # Collision with obstacle

        self.robot_positions[idx] = (x, y)
        return 1  # Reward for moving


# Register the environment for RLlib
def env_creator(_):
    map_gen = MapGen()
    test_map = map_gen.generate_connected_clusters_map(
        rows=20, cols=20, num_clusters=5, cluster_size_range=(5, 15), min_distance=3
    )
    return RectangleReductionPettingZooEnv(test_map, num_robots=4, field_of_view=1)


# Main Training Code
if __name__ == "__main__":
    from ray import init

    init(ignore_reinit_error=True)

    # Register environment
    register_env("rectangle_reduction", env_creator)

    # Define PPO configuration for RLlib
    config = {
        "env": "rectangle_reduction",
        "framework": "torch",
        "multiagent": {
            "policies": {
                f"robot_{i}": (None, gym.spaces.Dict({
                    "known_map": gym.spaces.Box(0, 1, shape=(20, 20), dtype=np.uint8),
                    "robot_position": gym.spaces.Box(0, 20, shape=(2,), dtype=np.int32),
                }), gym.spaces.Discrete(5), {})
                for i in range(4)
            },
            "policy_mapping_fn": lambda agent_id: agent_id,  # Each agent gets its own policy
        },
        "lr": 3e-4,
        "num_workers": 1,
        "train_batch_size": 2000,
    }

    # Train the model
    print("Starting training...")
    analysis = tune.run(
        ppo.PPOTrainer,
        config=config,
        stop={"training_iteration": 100},
        local_dir="models",
        checkpoint_at_end=True,
    )

    # Save the trained model
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
    )
    if checkpoints:
        best_checkpoint = checkpoints[0][0]
        print(f"Best checkpoint saved at: {best_checkpoint}")
