from GameManager import GameManager

NUM_SIMULATIONS = 20

def main():
    # Initialize game with a 5x5 ship, no specific strategy, and k = 2
    
    total_moves = 0
    for i in range(NUM_SIMULATIONS):
        game_manager = GameManager(20, 1, 3)
        curr_moves = game_manager.stretegy_one()
        total_moves += curr_moves
    print(f"Average moves Taken: {total_moves/NUM_SIMULATIONS}")


if __name__ == "__main__":
    main()
