import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
class MapGen():
    def generate_connected_clusters_map(rows, cols, num_clusters, cluster_size_range, min_distance):
        """
        Generate a map with connected clusters of obstacles.
        Args:
            rows: Number of rows in the map.
            cols: Number of columns in the map.
            num_clusters: Number of clusters to generate.
            cluster_size_range: Tuple (min_size, max_size) for the number of '1's per cluster.
            min_distance: Minimum distance between cluster centers.
        Returns:
            grid_map: A grid map with connected clusters of obstacles.
        """
        grid_map = np.zeros((rows, cols), dtype=int)
        cluster_centers = []

        # Function to check if a new center is sufficiently far from existing ones
        def is_far_enough(new_center, centers):
            for center in centers:
                if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
                    return False
            return True

        # Generate cluster centers
        while len(cluster_centers) < num_clusters:
            center_x = np.random.randint(0, rows)
            center_y = np.random.randint(0, cols)
            if is_far_enough((center_x, center_y), cluster_centers):
                cluster_centers.append((center_x, center_y))

        # Create connected clusters
        for center_x, center_y in cluster_centers:
            num_ones = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            cluster_cells = [(center_x, center_y)]  # Start with the cluster center
            grid_map[center_x, center_y] = 1

            # Expand the cluster until the required size is reached
            while len(cluster_cells) < num_ones:
                # Randomly pick a cell from the current cluster
                current_x, current_y = random.choice(cluster_cells)
                
                # Randomly select a direction to expand
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # Right, Left, Down, Up
                new_x, new_y = current_x + direction[0], current_y + direction[1]

                # Ensure the new cell is within grid boundaries and not already occupied
                if 0 <= new_x < rows and 0 <= new_y < cols and grid_map[new_x, new_y] == 0:
                    grid_map[new_x, new_y] = 1
                    cluster_cells.append((new_x, new_y))

        return grid_map

    # Generate a map with 5 clusters, sizes varying between 3 and 7, minimum distance 3
    connected_clusters_map = generate_connected_clusters_map(
        rows=100, cols=100, num_clusters=5, cluster_size_range=(25, 750), min_distance=10
    )
    print(connected_clusters_map)

    plt.imshow(connected_clusters_map, cmap="Greys", origin="upper")
    plt.title("Connected Clusters Map")
    plt.show()
