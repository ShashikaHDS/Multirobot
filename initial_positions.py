import random
import numpy as np
import image_converter

def get_random_positions(binary_grid, num_positions):
    """
    Generate random positions for robots in the free space of the map.
    Args:
        binary_grid (np.ndarray): The binary map where 1=obstacle, 0=free space.
        num_positions (int): Number of random positions to generate.
    Returns:
        positions (list of tuples): List of (row, col) positions.
    """
    # Find all free cells (value = 0)
    free_cells = [(r, c) for r in range(binary_grid.shape[0]) 
                          for c in range(binary_grid.shape[1]) if binary_grid[r, c] == 0]

    if len(free_cells) < num_positions:
        raise ValueError("Not enough free space to position all robots!")

    # Randomly sample positions from free cells without replacement
    positions = random.sample(free_cells, num_positions)
    return positions

# Load the binary grid (replace with your grid array)
binary_grid = np.array(image_converter.ImgConvert.binary_grid)

# Generate 5 random positions
num_positions = 5
try:
    random_positions = get_random_positions(binary_grid, num_positions)
    print("Random Initial Positions for Robots (avoiding obstacles):")
    for i, pos in enumerate(random_positions):
        print(f"Robot {i+1}: {pos}")
except ValueError as e:
    print(e)
