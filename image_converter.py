import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImgConvert():
    def image_to_binary_grid(image_path, grid_size=20):
        """
        Convert an image with grid cells into a binary array.
        Args:
            image_path (str): Path to the image file.
            grid_size (int): Size of each grid cell in pixels (default 20).
        Returns:
            binary_grid (np.ndarray): 2D array where 1=obstacle (green), 0=free space.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found. Check the file path.")

        # Convert the image to HSV (easier for color thresholding)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define green color range in HSV
        lower_green = np.array([35, 50, 50])   # Lower bound for green
        upper_green = np.array([85, 255, 255])  # Upper bound for green

        # Create a mask for green areas
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Get image dimensions
        height, width = mask.shape

        # Calculate the number of rows and columns based on grid size
        rows = height // grid_size
        cols = width // grid_size

        # Initialize a binary grid
        binary_grid = np.zeros((rows, cols), dtype=int)

        # Process each grid cell
        for i in range(rows):
            for j in range(cols):
                # Crop the grid cell
                cell = mask[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]
                # If there are enough non-zero (green) pixels in the cell, mark it as 1
                if np.sum(cell) > 0:
                    binary_grid[i, j] = 1

        return binary_grid

    def save_grid_as_2d_array(binary_grid, output_file):
        """
        Save the binary grid as a 2D array format to a text file.
        Args:
            binary_grid (np.ndarray): Binary grid array.
            output_file (str): Path to the output text file.
        """
        with open(output_file, 'w') as f:
            f.write("binary_grid = [\n")
            for row in binary_grid:
                f.write(f"    {row.tolist()},\n")
            f.write("]")
        print(f"2D binary grid saved to {output_file}")

    # Path to the image file
    image_path = "D:\MNEED\Thesis\RL\Maps\map1.png"  # Replace with your image path
    output_file = "binary_grid_2d_array.txt"  # Output file to save the grid

    # Convert image to binary grid
    binary_grid = image_to_binary_grid(image_path, grid_size=20)
    
    # Save the grid as a 2D array
    save_grid_as_2d_array(binary_grid, output_file)

    # Print the binary grid (optional)
    #print("Binary Grid Representation:")
    #print(binary_grid)

    # Optional: Display the binary grid
    #plt.imshow(binary_grid, cmap="Greys", origin="upper")
    #plt.title("Binary Grid Representation")
    #plt.show()
