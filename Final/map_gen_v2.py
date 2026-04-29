import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

class MapGen:
    @classmethod
    def generate_connected_clusters_map(cls, rows, cols, num_clusters, cluster_size_range, min_distance):
        """
        Generate a map with connected clusters of obstacles.

        Args:
            rows (int): Number of rows in the map.
            cols (int): Number of columns in the map.
            num_clusters (int): Number of clusters to generate.
            cluster_size_range (tuple): Range (min_size, max_size) for the number of '1's per cluster.
            min_distance (int): Minimum distance between cluster centers.

        Returns:
            np.ndarray: A 2D grid map with clusters of obstacles marked as 1.
        """
        grid_map = np.zeros((rows, cols), dtype=int)
        cluster_centers = []

        def is_far_enough(new_center, centers):
            """Check if the new center is sufficiently far from existing centers."""
            for center in centers:
                if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
                    return False
            return True

        # Step 1: Generate cluster centers
        while len(cluster_centers) < num_clusters:
            center_x = np.random.randint(0, rows)
            center_y = np.random.randint(0, cols)
            if is_far_enough((center_x, center_y), cluster_centers):
                cluster_centers.append((center_x, center_y))

        # Step 2: Create connected clusters
        for center_x, center_y in cluster_centers:
            num_ones = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            cluster_cells = [(center_x, center_y)]  # Start with cluster center
            grid_map[center_x, center_y] = 1

            while len(cluster_cells) < num_ones:
                current_x, current_y = random.choice(cluster_cells)
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_x, new_y = current_x + direction[0], current_y + direction[1]

                if 0 <= new_x < rows and 0 <= new_y < cols and grid_map[new_x, new_y] == 0:
                    grid_map[new_x, new_y] = 1
                    cluster_cells.append((new_x, new_y))

        # Step 3: Smoothen edges using morphological operations
        grid_map_smoothed = cls.smoothen_clusters(grid_map)

        return grid_map_smoothed

    @staticmethod
    def smoothen_clusters(grid_map):
        """
        Smoothen the cluster edges and fill holes using morphological operations.
        Args:
            grid_map (np.ndarray): The raw grid map with jagged clusters.
        Returns:
            np.ndarray: Smoothed and filled grid map.
        """
        # Convert grid to binary image (uint8 format for OpenCV)
        grid_binary = (grid_map * 255).astype(np.uint8)

        # Define a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Apply closing to fill holes inside clusters
        grid_closed = cv2.morphologyEx(grid_binary, cv2.MORPH_CLOSE, kernel)

        # Apply dilation and erosion (optional: closing again) to smooth edges
        grid_smoothed = cv2.morphologyEx(grid_closed, cv2.MORPH_OPEN, kernel)

        # Convert back to binary grid (0 and 1)
        return (grid_smoothed > 0).astype(int)

# Main code
if __name__ == "__main__":
    # Generate a map
    map_generator = MapGen()
    connected_clusters_map = map_generator.generate_connected_clusters_map(
        rows=100, cols=100, num_clusters=5, cluster_size_range=(25, 750), min_distance=10
    )
    print("Generated Map:")
    print(connected_clusters_map)

    # Visualize the smoothed map
    plt.imshow(connected_clusters_map, cmap="Greys", origin="upper")
    plt.title("Smoothed Connected Clusters Map")
    plt.show()
