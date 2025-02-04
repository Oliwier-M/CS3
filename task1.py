from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

def initialize_system(L):
    z = np.zeros(L, dtype=int)
    z_threshold = np.random.choice([1, 2], size=L)
    return z, z_threshold

def drive_system(z):
    z[0] += 1

def relax_system(z, z_threshold):
    L = len(z)
    avalanches = 0

    while np.any(z > z_threshold):
        for i in range(L):
            if z[i] > z_threshold[i]:
                avalanches += 1
                z_threshold[i] = np.random.choice([1, 2])
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
    z, z_threshold = initialize_system(L)
    heights = []
    avalanche_sizes = []

    for _ in range(num_grains):
        drive_system(z)
        z, z_threshold, avalanches = relax_system(z, z_threshold)
        heights.append(np.sum(z))
        avalanche_sizes.append(avalanches)

    return heights, avalanche_sizes


def calculate_probability(avalanche_sizes, max_size):
    counts = Counter(avalanche_sizes)
    s = np.arange(1, max_size + 1)
    P = np.array([counts[size] for size in s]) / len(avalanche_sizes)
    return s, P


def task_1(L, num_grains):
    heights, avalanche_sizes = oslo_model(L, num_grains)

    plt.figure(figsize=(12, 6))

    # system height over time
    plt.subplot(1, 2, 1)
    plt.plot(range(num_grains), heights, label="System Height")
    plt.xlabel("Number of Grains")
    plt.ylabel("Height")
    plt.title("System Height vs. Number of Grains")
    plt.legend()

    # avalanche sizes
    plt.subplot(1, 2, 2)
    plt.plot(range(num_grains), avalanche_sizes, label="Avalanche Size")
    plt.xlabel("Number of Grains")
    plt.ylabel("Avalanche Size")
    plt.title("Avalanche Size vs. Number of Grains")
    plt.legend()

    plt.tight_layout()
    plt.show()

def task_2(L, num_grains):
    heights, avalanche_sizes = oslo_model(L, num_grains)

    # s_max for scaling
    s_max = max(avalanche_sizes)

    # scaled avalanche sizes
    scaled_avalanche_sizes = [s / s_max for s in avalanche_sizes]

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
    results = {}
    for L in system_sizes:
        _, avalanche_sizes = oslo_model(L, num_grains)
        max_size = max(avalanche_sizes)
        s, P = calculate_probability(avalanche_sizes, max_size)
        results[L] = (s, P)

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
