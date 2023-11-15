from GameManagerDeterministic import GameManager
from GameManagerProbability import GameManager_Probability
import matplotlib.pyplot as plt
import time

NUM_SIMULATIONS_ALPHA = 10
NUM_SIMULATIONS_K = 10

def main():
    start_time = time.time()
    values_of_k = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    values_of_alpha = [.05,.1,.15,.2,.25,.3,.35,.4, .45,.5,.55,.6, .7,.8, .9, 1]
    averages_alpha = []
    averages_k = []

    for alpha in values_of_alpha:
        total_result = 0
        for i in range(NUM_SIMULATIONS_ALPHA):
            game = GameManager_Probability(50,4, alpha)
            result = game.run_game()  # run_game should return the number of moves
            total_result += result
        averages_alpha.append(total_result / NUM_SIMULATIONS_ALPHA)
        print(f"Average for alpha={alpha}: {averages_alpha[-1]}")
        
    for k in values_of_k:
       total_result = 0
       for i in range(NUM_SIMULATIONS_K):
           game = GameManager_Probability(50,4, k)
           result = game.run_game()  # run_game should return the number of moves
           total_result += result
       averages_k.append(total_result / NUM_SIMULATIONS_K)
       print(f"Average for alpha={k}: {averages_k[-1]}")

    # Plotting the results
    plt.plot(values_of_alpha, averages_alpha, marker='o')
    plt.title('Average Number of Moves per Game for Different Values of alpha')
    plt.xlabel('alpha value')
    plt.ylabel('Average number of moves')
    plt.grid(True)
    end_time = time.time()  # Record the end time
    print(f"Total execution time: {end_time - start_time} seconds")
    plt.show()
    
    
    plt.plot(values_of_alpha, averages_k, marker='o')
    plt.title('Average Number of Moves per Game for Different Values of k')
    plt.xlabel('k values')
    plt.ylabel('Average number of moves')
    plt.grid(True)
    end_time = time.time()  # Record the end time
    print(f"Total execution time: {end_time - start_time} seconds")
    plt.show()

if __name__ == "__main__":
    main()
