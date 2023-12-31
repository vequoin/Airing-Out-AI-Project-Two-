from GameManager import GameManager
from gametwo import GameManager_Probability
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 30

def main():
    values_of_alpha = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    averages = []

    for alpha in values_of_alpha:
        total_result = 0
        for i in range(NUM_SIMULATIONS):
            game = GameManager_Probability(30, 8, alpha)
            result = game.run_game()  # run_game should return the number of moves
            total_result += result
        averages.append(total_result / NUM_SIMULATIONS)
        print(f"Average for alpha={alpha}: {averages[-1]}")

    # Plotting the results
    plt.plot(values_of_alpha, averages, marker='o')
    plt.title('Average Number of Moves per Game for Different Values of alpha')
    plt.xlabel('alpha value')
    plt.ylabel('Average number of moves')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
