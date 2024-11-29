from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

def initialize_system(L):
    """
    Initialize the Oslo model system.

    Parameters:
        L (int): Length of the system.

    Returns:
        z (np.ndarray): Array of slopes initialized to 0.
        z_threshold (np.ndarray): Array of random threshold slopes (1 or 2).
    """
    z = np.zeros(L, dtype=int)
    z_threshold = np.random.choice([1, 2], size=L)
    return z, z_threshold

def drive_system(z):
    """
    Add a grain to the left-most site.

    Parameters:
        z (np.ndarray): Current slope configuration.
    """
    z[0] += 1

def relax_system(z, z_threshold):
    """
    Relax the system until all slopes are below or equal to their thresholds.

    Parameters:
        z (np.ndarray): Current slope configuration.
        z_threshold (np.ndarray): Current threshold slopes.

    Returns:
        z (np.ndarray): Updated slope configuration.
        z_threshold (np.ndarray): Updated threshold slopes.
        avalanches (int): Total number of topplings during the relaxation.
    """
    L = len(z)
    avalanches = 0

    while np.any(z > z_threshold):
        for i in range(L):
            if z[i] > z_threshold[i]:
                avalanches += 1
                z_threshold[i] = np.random.choice([1, 2])  # Choose new threshold
                if i == 0:
                    z[i] -= 2
                    z[i + 1] += 1
                elif i == L - 1:
                    z[i] -= 1
                    z[i - 1] += 1
                else:
                    z[i] -= 2
                    z[i - 1] += 1
                    z[i + 1] += 1

    return z, z_threshold, avalanches

def oslo_model(L, num_grains):
    """
    Simulate the Oslo model.

    Parameters:
        L (int): Length of the system.
        num_grains (int): Number of grains to drop into the system.

    Returns:
        heights (list): Heights of the system after each grain addition.
        avalanche_sizes (list): Size of each avalanche.
    """
    z, z_threshold = initialize_system(L)
    heights = []
    avalanche_sizes = []

    for _ in range(num_grains):
        drive_system(z)
        z, z_threshold, avalanches = relax_system(z, z_threshold)
        heights.append(np.sum(z))  # Total height of the system
        avalanche_sizes.append(avalanches)

    return heights, avalanche_sizes


# Function to calculate avalanche size probability
def calculate_probability(avalanche_sizes, max_size):
    """
    Calculate the probability P(s, L) for avalanche sizes.

    Parameters:
        avalanche_sizes (list): List of avalanche sizes.
        max_size (int): Maximum size for bins.

    Returns:
        P (np.ndarray): Normalized probabilities.
        s (np.ndarray): Avalanche size bins.
    """
    counts = Counter(avalanche_sizes)
    s = np.arange(1, max_size + 1)
    P = np.array([counts[size] for size in s]) / len(avalanche_sizes)
    return s, P


def task_1(L, num_grains):
    # Run the simulation
    heights, avalanche_sizes = oslo_model(L, num_grains)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot system height over time
    plt.subplot(1, 2, 1)
    plt.plot(range(num_grains), heights, label="System Height")
    plt.xlabel("Number of Grains")
    plt.ylabel("Height")
    plt.title("System Height vs. Number of Grains")
    plt.legend()

    # Plot avalanche sizes
    plt.subplot(1, 2, 2)
    plt.plot(range(num_grains), avalanche_sizes, label="Avalanche Size")
    plt.xlabel("Number of Grains")
    plt.ylabel("Avalanche Size")
    plt.title("Avalanche Size vs. Number of Grains")
    plt.legend()

    plt.tight_layout()
    plt.show()

def task_2(L, num_grains):
    # Run the Oslo model
    heights, avalanche_sizes = oslo_model(L, num_grains)

    # Find s_max for scaling
    s_max = max(avalanche_sizes)

    # Compute scaled avalanche sizes
    scaled_avalanche_sizes = [s / s_max for s in avalanche_sizes]

    # Plot the scaled avalanche size as a function of time
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_grains), scaled_avalanche_sizes, label=r"Scaled Avalanche Size $s/s_{max}$")
    plt.axvline(x=L, color='r', linestyle='--', label='Transient Period End ($t \\approx L$)')
    plt.xlabel("Time $t$ (Grain Additions)")
    plt.ylabel("Scaled Avalanche Size $s/s_{max}$")
    plt.title("Scaled Avalanche Size vs. Time")
    plt.legend()
    plt.grid()
    plt.show()

def task_3(system_sizes, num_grains):
    # Run simulations for different system sizes
    results = {}
    for L in system_sizes:
        _, avalanche_sizes = oslo_model(L, num_grains)
        max_size = max(avalanche_sizes)
        s, P = calculate_probability(avalanche_sizes, max_size)
        results[L] = (s, P)

    # Plot P(s, L) vs. s in log-log scale
    plt.figure(figsize=(10, 6))
    for L in system_sizes:
        s, P = results[L]
        plt.loglog(s, P, label=f"L = {L}")

    plt.xlabel("Avalanche Size $s$")
    plt.ylabel("Probability $P(s, L)$")
    plt.title("Avalanche Size Probability $P(s, L)$ vs. Size $s$")
    plt.legend()
    plt.grid()
    plt.show()

task_1(10, 1000)
task_2(10, 1000)
task_3([64, 128, 256], 50000)
