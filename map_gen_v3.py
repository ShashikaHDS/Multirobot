import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

class MapGen:

    @classmethod
    def generate_connected_clusters_map(cls, rows, cols, num_clusters, cluster_size_range, min_distance):
        """
        Generate a map with multiple clusters, avoiding unreachable spaces.

        Args:
            rows (int): Number of rows in the map.
            cols (int): Number of columns in the map.
            num_clusters (int): Number of clusters to generate.
            cluster_size_range (tuple): Range (min_size, max_size) for the number of '1's per cluster.
            min_distance (int): Minimum distance between cluster centers.

        Returns:
            np.ndarray: A 2D grid map with clusters marked as 1.
        """
        grid_map = np.zeros((rows, cols), dtype=int)
        cluster_centers = []
        occupied_mask = np.zeros((rows, cols), dtype=bool)

        def is_far_enough(new_center, centers):
            """Check if the new center is sufficiently far from existing centers."""
            for center in centers:
                if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
                    return False
            return True

        def is_valid_growth(new_x, new_y):
            """Check if the growth does not block paths and remains valid."""
            if not (0 <= new_x < rows and 0 <= new_y < cols) or grid_map[new_x, new_y] == 1:
                return False
            return True

        def is_fully_connected():
            """Check if all open spaces in the map are connected to the borders."""
            visited = np.zeros_like(grid_map, dtype=bool)
            queue = []

            # Start flood-fill from the first empty border cell
            for r in range(rows):
                if grid_map[r][0] == 0:
                    queue.append((r, 0))
                    break
            for r in range(rows):
                if grid_map[r][cols - 1] == 0:
                    queue.append((r, cols - 1))
                    break
            for c in range(cols):
                if grid_map[0][c] == 0:
                    queue.append((0, c))
                    break
            for c in range(cols):
                if grid_map[rows - 1][c] == 0:
                    queue.append((rows - 1, c))
                    break

            # Perform BFS to mark all connected empty spaces
            while queue:
                x, y = queue.pop(0)
                if visited[x, y]:
                    continue
                visited[x, y] = True
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid_map[nx, ny] == 0:
                        queue.append((nx, ny))

            # Check if any open cell is not connected
            for r in range(rows):
                for c in range(cols):
                    if grid_map[r][c] == 0 and not visited[r, c]:
                        return False

            return True

        # Step 1: Generate distinct cluster centers
        while len(cluster_centers) < num_clusters:
            center_x = np.random.randint(0, rows)
            center_y = np.random.randint(0, cols)
            if is_far_enough((center_x, center_y), cluster_centers):
                cluster_centers.append((center_x, center_y))

        # Step 2: Grow clusters with connectivity checks
        for center_x, center_y in cluster_centers:
            cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)

            print(cluster_size_range[0],cluster_size_range[1])
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1
            occupied_mask[center_x, center_y] = True

            while len(cluster_cells) < cluster_size:
                current_x, current_y = random.choice(cluster_cells)
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(directions)  # Randomize growth direction

                for direction in directions:
                    new_x = current_x + direction[0]
                    new_y = current_y + direction[1]

                    # Validate growth and ensure connectivity
                    if is_valid_growth(new_x, new_y):
                        grid_map[new_x, new_y] = 1
                        cluster_cells.append((new_x, new_y))
                        occupied_mask[new_x, new_y] = True
                        break

            # After each cluster is placed, verify full connectivity
            if not is_fully_connected():
                # Undo the last cluster placement if it blocks connectivity
                for x, y in cluster_cells:
                    grid_map[x, y] = 0

        return grid_map




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


# Main code for testing
if __name__ == "__main__":
    # Generate a map with a 2-cell margin from the borders
    map_generator = MapGen()
    connected_clusters_map = map_generator.generate_connected_clusters_map(
        rows=20, cols=20, num_clusters=5, cluster_size_range=(5, 15), min_distance=5
    )
    print("Generated Map:")
    print(connected_clusters_map)

    # Visualize the map
    plt.imshow(connected_clusters_map, cmap="Greys", origin="upper")
    plt.title("Connected Clusters Map with 2-Grid-Cell Border Gap")
    plt.show()
