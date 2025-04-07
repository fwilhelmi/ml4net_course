import numpy as np
import matplotlib.pyplot as plt

def get_random_points_in_map(num_points, map_size):
    """Generates random positions in a 2D map."""
    positions = np.random.rand(num_points, 2) * map_size
    return positions

def plot_deployment_and_performance(positions, map_size, throughput, pe, pc, ps):
    """Plots the Wi-Fi deployment."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Deployment subplot
    axs[0, 0].scatter(map_size / 2, map_size / 2, s=150, c='red', marker='s', label='AP device')
    axs[0, 0].scatter(positions[:, 0], positions[:, 1], s=120, c='blue', marker='o', label='STA device')
    axs[0, 0].set_xlabel('X-coordinate [m]', fontsize=14)
    axs[0, 0].set_ylabel('Y-coordinate [m]', fontsize=14)
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].axis([0, map_size, 0, map_size])
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

    # Probability subplot
    probabilities = [pe, pc, ps]
    labels = ['p_e', 'p_c', 'p_s']
    axs[0, 1].bar(labels, probabilities)
    axs[0, 1].set_ylabel('Probability', fontsize=14)
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
    axs[0, 1].legend(['Probability'], fontsize=12)

    # Throughput subplot
    throughput_value = throughput / 1e6
    axs[1, 1].bar(['Throughput'], [throughput_value])
    axs[1, 1].set_ylabel('Throughput [Mbps]', fontsize=14)
    axs[1, 1].set_xticks(['Throughput'])  # Ensure the x-tick is centered
    axs[1, 1].axis([-0.5, 0.5, 0, max(60, throughput_value * 1.2)])  # Adjust x-axis limits
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

    # Remove the empty subplot
    fig.delaxes(axs[1, 0])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("========================================")
    print("        ML4Net 2024-2025        ")
    print("========================================")

    # Scenario parameters
    map_size = 10           # Map size (x axis) in meters
    number_of_stas = 20     # Number of stations (STAs)
    cw = 16                 # Contention Window (CW)
    # Channel access & transmission parameters
    L = 12000       # Bits per packet
    R = 47e6        # Transmission rate (bits per second)
    DIFS = 34e-6    # DIFS duration (seconds)
    SIFS = 16e-6    # SIFS duration (seconds)
    Tack = 40e-6    # Time for transmitting and ACK (seconds)
    Ttx = L / R     # Time for transmitting a data packet (seconds)

    # Assign random positions to the STAs
    positions = get_random_points_in_map(number_of_stas, map_size)

    # Compute network performance
    # - Compute "slot" probabilities
    tau = 2 / (cw + 1)
    pe = (1 - tau) ** number_of_stas
    ps = number_of_stas * tau * (1 - tau) ** (number_of_stas - 1)
    pc = 1 - pe - ps
    # - Compute "slot" durations
    Te = 9e-6  # Duration of an empty slot
    Tc = DIFS + Ttx + 2 * SIFS
    Ts = DIFS + Ttx + SIFS + Tack
    Ttotal = pe * Te + pc * Tc + ps * Ts
    throughput = (ps * L) / Ttotal

    # Plot the deployment
    plot_deployment_and_performance(positions, map_size, throughput, pe, pc, ps)