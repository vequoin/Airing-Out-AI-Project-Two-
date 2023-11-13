import csv
from Ship import Ship
import struct
from collections import deque
import random
from gametwo import GameManager_Probability
from GameManager import GameManager
from multiprocessing import Pool


def run_simulation(alpha, run_number):
    # Set up your simulation environment with the given alpha value
    # Run the simulation
    # Return the result (e.g., number of moves, success/failure, etc.)
    game = GameManager_Probability(15, 8, alpha)
    result = game.run_game()
    return alpha, run_number, result


def run_all_simulations(alpha_values, num_simulations_per_alpha):
    tasks = [(alpha, run) for alpha in alpha_values for run in range(num_simulations_per_alpha)]

    with Pool() as pool:
        results = pool.starmap(run_simulation, tasks)

    # Save results to a CSV file
    with open('simulation_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Alpha Value', 'Run Number', 'Result'])  # Header row
        for result in results:
            writer.writerow(result)  # result already contains alpha, run_number, and simulation result

if __name__ == "__main__":
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5,.6, .7,.8, .9, 1]
    num_simulations_per_alpha = 30
    run_all_simulations(alpha_values, num_simulations_per_alpha)
