import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

class MapGen:
    @classmethod
    def generate_connected_clusters_map(cls, rows, cols, num_clusters, cluster_size_range, min_distance):
        """
        Generate a map with multiple clusters, ensuring:
        - Connectivity of free cells.
        - Minimum distance between clusters.
        - All clusters and obstacles follow the constraints.
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
            """Check if the growth is valid."""
            if not (0 <= new_x < rows and 0 <= new_y < cols) or grid_map[new_x, new_y] == 1:
                return False
            # Ensure no diagonal adjacency with other clusters
            neighbors = [
                (new_x - 1, new_y - 1), (new_x - 1, new_y + 1),
                (new_x + 1, new_y - 1), (new_x + 1, new_y + 1)
            ]
            if any(0 <= nx < rows and 0 <= ny < cols and grid_map[nx, ny] == 1 for nx, ny in neighbors):
                return False
            return True

        def ensure_free_cell_connectivity():
            """Ensure all free cells are connected."""
            visited = np.zeros_like(grid_map, dtype=bool)

            # Perform BFS from the first free cell
            def bfs(start):
                queue = [start]
                visited[start] = True
                while queue:
                    x, y = queue.pop(0)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid_map[nx, ny] == 0:
                            visited[nx, ny] = True
                            queue.append((nx, ny))

            # Start BFS
            for x in range(rows):
                for y in range(cols):
                    if grid_map[x, y] == 0:
                        bfs((x, y))
                        break

            # Convert disconnected free cells into obstacles
            for x in range(rows):
                for y in range(cols):
                    if grid_map[x, y] == 0 and not visited[x, y]:
                        grid_map[x, y] = 1

        # Step 1: Generate cluster centers
        while len(cluster_centers) < num_clusters:
            center_x = random.randint(1, rows - 2)
            center_y = random.randint(1, cols - 2)
            if is_far_enough((center_x, center_y), cluster_centers):
                cluster_centers.append((center_x, center_y))

        # Step 2: Grow clusters
        for center_x, center_y in cluster_centers:
            cluster_size = random.randint(*cluster_size_range)
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1

            while len(cluster_cells) < cluster_size:
                current_x, current_y = random.choice(cluster_cells)
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(directions)

                for dx, dy in directions:
                    new_x = current_x + dx
                    new_y = current_y + dy

                    if is_valid_growth(new_x, new_y):
                        grid_map[new_x, new_y] = 1
                        cluster_cells.append((new_x, new_y))
                        break

        # Step 3: Ensure all free cells are connected
        ensure_free_cell_connectivity()

        return grid_map

    @staticmethod
    def smoothen_clusters(grid_map):
        """
        Smoothen the cluster edges and fill holes using morphological operations.
        """
        grid_binary = (grid_map * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        grid_closed = cv2.morphologyEx(grid_binary, cv2.MORPH_CLOSE, kernel)
        grid_smoothed = cv2.morphologyEx(grid_closed, cv2.MORPH_OPEN, kernel)
        return (grid_smoothed > 0).astype(int)

# Main code for testing
if __name__ == "__main__":
    rows, cols = 20, 20
    num_clusters = 5
    cluster_size_range = (5, 15)
    min_distance = 5

    map_generator = MapGen()
    generated_map = map_generator.generate_connected_clusters_map(rows, cols, num_clusters, cluster_size_range, min_distance)

    print("Generated Map:")
    print(generated_map)

    plt.imshow(generated_map, cmap="Greys", origin="upper")
    plt.title("Generated Map with Connected Free Cells")
    plt.show()
