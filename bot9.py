from prob_strategies import Strategy3Bot
class Strategy9_Bot(Strategy3Bot):
    # Assuming other necessary methods and attributes are already defined

    def __init__(self, ship,position, alpha, gameManager):
        super(ship, position)
        self.alpha = alpha
        self.gameManager = gameManager
        # Other initializations

    def strategy_nine(self):
        while not self.task_complete():
            if self.alpha > 0.5:
                self.high_alpha_strategy()
            else:
                self.low_alpha_strategy()

    def high_alpha_strategy(self):
        # Sensing more frequently and focusing on high-probability areas
        if self.should_sense():
            self.gameManager.sense_multi()
            self.gameManager.update_probabilities_

        # Identify high-probability clusters
        clusters = self.find_clusters()

        # Move towards the nearest or best high-probability cluster
        self.move_to_cluster(clusters)

    def low_alpha_strategy(self):
        # Sensing less frequently, covering more area
        if self.should_sense():
            self.sense_and_update()

        # Exploration strategy
        self.explore()

    def should_sense(self):
        # Implement logic to decide when to sense
        pass

    def find_clusters(self):
        # Implement clustering logic for high-alpha strategy
        pass

    def move_to_cluster(self, clusters):
        # Implement logic to move towards a cluster
        pass

    def explore(self):
        # Implement exploration logic for low-alpha strategy
        pass

    def task_complete(self):
        if self.gameManager.leaks:
            return False
        return True
    
    
    def find_clusters(self, prob_threshold=0.1, distance_threshold=3):
        # Filter out high-probability cell pairs
        high_prob_pairs = [pair for pair, prob in self.probability_grid.items() if prob >= prob_threshold]

        # Initialize clusters
        clusters = []

        # Function to check if a cell pair is close to any pair in a cluster
        def is_nearby(pair, cluster):
            for cluster_pair in cluster:
                if self.are_pairs_nearby(pair, cluster_pair, distance_threshold):
                    return True
            return False

        # Iterate over each high-probability cell pair
        for pair in high_prob_pairs:
            added_to_cluster = False
            for cluster in clusters:
                if is_nearby(pair, cluster):
                    cluster.append(pair)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                clusters.append([pair])  # Start a new cluster

        return clusters

    def are_pairs_nearby(self, pair1, pair2, distance_threshold):
        # Check if any cell in pair1 is within distance_threshold of any cell in pair2
        for cell1 in pair1:
            for cell2 in pair2:
                if self.calculate_distance(cell1, cell2) <= distance_threshold:
                    return True
        return False

    def calculate_distance(self, cell1, cell2):
        # Manhattan distance
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

