from GameManagerDeterministic import GameManager
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 30

def main():
    values_of_k = [2, 4, 6, 8, 10, 12, 15, 17, 20, 25]
    averages = []

    for k in values_of_k:
        total_result = 0
        for i in range(3):
            game = GameManager(24, 2, k)
            result = 0
            if game.isleak:
                result = game.run_game()  # run_game should return the number of moves
            total_result += result
        averages.append(total_result/3)
        print(f"Average for k={k}: {averages[-1]}")
        measure = input("Press Enter...")

    # Plotting the results
    plt.plot(values_of_k, averages, marker='o')
    plt.title('Average Number of Moves per Game for Different Values of k')
    plt.xlabel('k value')
    plt.ylabel('Average number of moves')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
