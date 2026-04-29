import numpy as np
import pygame
from stable_baselines3 import PPO
from rl_mltirobot_pygame import RectangleReductionEnv  # Import your custom environment

# Function to run the trained model
def run_model(model_path, rows=20, cols=20, num_clusters=5, cluster_size_range=(10, 50), min_distance=5, num_robots=4, field_of_view=1):
    """
    Load and run a trained PPO model in the Rectangle Reduction Environment.

    Args:
        model_path (str): Path to the saved model file.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        num_clusters (int): Number of obstacle clusters.
        cluster_size_range (tuple): Size range of clusters.
        min_distance (int): Minimum distance between clusters.
        num_robots (int): Number of robots.
        field_of_view (int): Robot's field of view.
    """
    # Initialize the environment
    env = RectangleReductionEnv(
        rows=rows, cols=cols, 
        num_clusters=num_clusters, 
        cluster_size_range=cluster_size_range, 
        min_distance=min_distance, 
        num_robots=num_robots, 
        field_of_view=field_of_view
    )

    # Load the trained model
    print("Loading the trained model...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")

    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0

    print("Running the model...")

    # Run the model
    while not done:
        action, _ = model.predict(obs)  # Predict actions using the model
        obs, reward, done, _ = env.step(action)  # Take a step in the environment
        total_reward += reward
        print(f"Step Reward: {reward}, Total Reward: {total_reward}, Done: {done}")

    print("Model execution complete!")
    print(f"Total reward achieved: {total_reward}")
    pygame.quit()

# Main script to run the model
if __name__ == "__main__":
    # Path to the saved model
    model_path = "rectangle_reduction_model.zip"

    # Run the model
    run_model(model_path)
