# simulation.py

import numpy as np

def simulate_multiple_combinations(model, X_seed, combo_size, history_set, num_simulations=10):
    """
    Generate multiple unique lottery combinations using a trained model.

    Parameters:
        model (Sequential): Trained prediction model.
        X_seed (np.array): Seed input sequence for prediction.
        combo_size (int): Number of numbers per combination.
        history_set (set): Set of historical combinations.
        num_simulations (int): Number of combinations to generate.

    Returns:
        list: List of unique predicted combinations.
    """
    simulations = []
    pred = model.predict(X_seed.reshape(1, X_seed.shape[0], 1))[0]
    sorted_indices = np.argsort(pred)[::-1]

    i = 0
    while len(simulations) < num_simulations and i + combo_size <= len(sorted_indices):
        combo = tuple(sorted(sorted_indices[i:i+combo_size] + 1))
        if combo not in history_set and combo not in simulations:
            simulations.append(combo)
        i += 1

    return simulations

def print_simulation_results(simulations):
    """
    Print the list of simulated combinations.

    Parameters:
        simulations (list): List of predicted combinations.
    """
    print("🎲 Simulated Lottery Combinations:")
    for i, combo in enumerate(simulations, 1):
        print(f"{i:02d}: {combo}")
