import numpy as np
import matplotlib.pyplot as plt
import random


class MapGen:
    @classmethod
    def generate_connected_clusters_map(cls, rows, cols, num_clusters, cluster_size_range, min_distance):
        """
        Generate a map with multiple clusters, ensuring constraints like free cell connectivity.

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

        def is_far_enough(new_center, centers):
            """Check if the new center is sufficiently far from existing centers."""
            for center in centers:
                if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
                    return False
            return True

        def is_valid_growth(new_x, new_y):
            """Check if the new position is valid for cluster growth."""
            # Ensure the position is within bounds and not already occupied
            if not (1 <= new_x < rows - 1 and 1 <= new_y < cols - 1):
                return False
            return grid_map[new_x, new_y] == 0

        def grow_cluster(center_x, center_y, cluster_size):
            """Grow a cluster starting from the given center."""
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1

            while len(cluster_cells) < cluster_size:
                current_x, current_y = random.choice(cluster_cells)
                # Allow diagonal growth
                directions = [
                    (0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                random.shuffle(directions)

                for dx, dy in directions:
                    new_x, new_y = current_x + dx, current_y + dy
                    if is_valid_growth(new_x, new_y):
                        grid_map[new_x, new_y] = 1
                        cluster_cells.append((new_x, new_y))
                        break

        def ensure_full_connectivity():
            """Ensure all free cells are connected."""
            visited = np.zeros_like(grid_map, dtype=bool)

            def flood_fill(x, y):
                """Perform a flood-fill algorithm."""
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if visited[cx, cy]:
                        continue
                    visited[cx, cy] = True
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid_map[nx, ny] == 0:
                            stack.append((nx, ny))

            # Find the first free cell to start the flood-fill
            free_cells = np.argwhere(grid_map == 0)
            if len(free_cells) == 0:
                return

            first_free_cell = free_cells[0]
            flood_fill(first_free_cell[0], first_free_cell[1])

            # Check for unvisited free cells and convert them to obstacles
            for r in range(rows):
                for c in range(cols):
                    if grid_map[r, c] == 0 and not visited[r, c]:
                        grid_map[r, c] = 1  # Convert isolated free cells to obstacles

        # Step 1: Generate distinct cluster centers
        while len(cluster_centers) < num_clusters:
            center_x = np.random.randint(1, rows - 1)
            center_y = np.random.randint(1, cols - 1)
            if is_far_enough((center_x, center_y), cluster_centers):
                cluster_centers.append((center_x, center_y))

        # Step 2: Grow clusters
        for center_x, center_y in cluster_centers:
            cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            grow_cluster(center_x, center_y, cluster_size)

        # Step 3: Ensure all free cells are connected
        ensure_full_connectivity()

        return grid_map


# Main code for testing
if __name__ == "__main__":
    map_generator = MapGen()
    connected_clusters_map = map_generator.generate_connected_clusters_map(
        rows=20, cols=20, num_clusters=3, cluster_size_range=(3, 4), min_distance=5
    )
    print("Generated Map:")
    print(connected_clusters_map)

    # Visualize the map
    plt.imshow(connected_clusters_map, cmap="Greys", origin="upper")
    plt.title("Connected Clusters Map")
    plt.show()
