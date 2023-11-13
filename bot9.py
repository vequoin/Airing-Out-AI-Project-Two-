import math
import random
from GameManagerProbability import GameManager_Probability

class Bot:
    # ... other methods and attributes ...

    def __init__(self, ship, alpha, gameManager):
        self.ship = ship
        self.alpha = alpha
        self.gameManager = gameManager
        self.steps_since_last_sense = 0
        self.sense_frequency = 5  # Sensing every 5 steps as an example

    def should_sense(self):
        # Sense every 'sense_frequency' steps
        if self.steps_since_last_sense >= self.sense_frequency:
            self.steps_since_last_sense = 0  # Reset counter after sensing
            return True
        return False
    

    def sense_and_update(self):
        # Activate the sensing mechanism
        beep_heard = self.sense_multiple()

        # Update the probability grid based on whether a beep was heard
        if beep_heard:
            self.update_probabilities_based_on_beep()
        else:
            self.update_probabilities_no_beep()

        # Additional logic, if necessary

    def sense_multiple(self):
        # Assuming this method is already implemented to return True if a beep is heard
        pass

    def update_probabilities_no_beep(self):
        # Logic to update probabilities when no beep is heard
        # This would typically decrease the probability in areas closer to the bot
        pass



    def update_probabilities_based_on_beep(self):
        for cell in self.gameManager.probability_grid.keys():
            distance = self.gameManager.calculate_distance(self.position, cell)
            beep_probability = self.gameManager.calculate_probability(distance)

            # Increase probability based on proximity and beep probability
            self.probability_grid[cell] = min(1, self.probability_grid[cell] + beep_probability)

        self.normalize_probabilities()

    def update_probabilities_no_beep(self):
        for cell in self.probability_grid.keys():
            distance = self.calculate_distance(self.position, cell)
            beep_probability = self.calculate_beep_probability(distance)

            # Decrease probability based on proximity and lack of beep
            self.probability_grid[cell] *= (1 - beep_probability)

        self.normalize_probabilities()


    def calculate_beep_probability(self, distance):
        # Calculate the probability of detecting a beep based on distance
        # This could be a function that decreases with distance
        return math.exp(-self.alpha * (distance - 1))

    def normalize_probabilities(self):
        total_prob = sum(self.probability_grid.values())
        for cell in self.probability_grid:
            self.probability_grid[cell] /= total_prob


    def find_clusters(self):
        # Define a threshold to consider a cell as part of a high-probability cluster
        threshold = self.calculate_cluster_threshold()

        # Find high-probability cells
        high_prob_cells = [cell for cell, prob in self.probability_grid.items() if prob >= threshold]

        # Group these cells into clusters
        clusters = self.group_into_clusters(high_prob_cells)
        return clusters

    def calculate_cluster_threshold(self):
        # Calculate a threshold value to identify high-probability cells
        # This could be a dynamic value based on the current state of the probability grid
        mean_prob = sum(self.probability_grid.values()) / len(self.probability_grid)
        return mean_prob  # Example: using the mean probability as a threshold


    def group_into_clusters(self, high_prob_cells):
        clusters = []
        visited = set()

        for cell in high_prob_cells:
            if cell not in visited:
                # Start a new cluster
                cluster = [cell]
                visited.add(cell)

                # Find neighboring high-probability cells
                for neighbor in self.get_neighbors(cell):
                    if neighbor in high_prob_cells and neighbor not in visited:
                        cluster.append(neighbor)
                        visited.add(neighbor)

                clusters.append(cluster)

        return clusters

    def get_neighbors(self, cell):
        # Return the neighboring cells of the given cell
        # Implement logic to find adjacent cells
        pass


    def move_to_cluster(self, clusters):
        if not clusters:
            # No clusters found, revert to a different strategy or exploration
            self.explore()
            return

        # Choose the best cluster to move to, based on certain criteria
        target_cluster = self.choose_best_cluster(clusters)

        # Determine the best path to reach the cluster
        path_to_cluster = self.find_path_to_cluster(target_cluster)

        # Move along the path
        self.follow_path(path_to_cluster)

    def choose_best_cluster(self, clusters):
        # Choose the cluster to move towards
        # This can be based on cluster size, total probability, distance, etc.
        # For simplicity, let's choose the nearest cluster
        nearest_cluster = min(clusters, key=lambda cluster: self.calculate_distance_to_cluster(cluster))
        return nearest_cluster

    def calculate_distance_to_cluster(self, cluster):
        # Calculate the distance from the bot to the cluster
        # For simplicity, calculate distance to the closest cell in the cluster
        distances = [self.calculate_distance(self.position, cell) for cell in cluster]
        return min(distances)  # Return the minimum distance

    def find_path_to_cluster(self, cluster):
        # Find the best path to the chosen cluster
        # This could use a pathfinding algorithm like A* or Dijkstra's
        # For simplicity, let's assume a function that does this is available
        path = self.pathfinding_algorithm(self.position, cluster)
        return path

    def follow_path(self, path):
        # Move the bot along the given path
        # This could involve updating the bot's position and handling movement logic
        for step in path:
            self.move(step)  # Assuming a move method is implemented

    def pathfinding_algorithm(self, start, end):
        # Implement or call a pathfinding algorithm
        pass


    import random

class Bot:
    # ... other methods and attributes ...

    def explore(self):
        # Choose the next move based on exploration strategy
        next_move = self.choose_next_exploration_move()

        # Move the bot to the next position
        self.move(next_move)

    def choose_next_exploration_move(self):
        # Get neighboring cells that have not been recently visited
        neighbors = self.get_valid_neighbors()

        # Choose a random neighbor to move to
        return random.choice(neighbors) if neighbors else None

    def get_valid_neighbors(self):
        # Get neighbors of the current position
        neighbors = self.get_neighbors(self.position)

        # Filter out neighbors that have been recently visited
        return [neighbor for neighbor in neighbors if neighbor not in self.recently_visited]

    def get_neighbors(self, position):
        # Implement logic to get neighboring cells of the given position
        pass

    def move(self, position):
        # Move the bot to the specified position and update relevant states
        self.position = position
        self.update_recently_visited(position)
        # Additional movement logic...

    def update_recently_visited(self, position):
        # Update the list/set of recently visited cells
        self.recently_visited.add(position)
        # Optionally, remove the oldest entry if the list/set exceeds a certain size
