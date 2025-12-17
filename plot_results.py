import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size=50):
    """
    Smooths the data by calculating the average of the last 'window_size' episodes.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


try:
    per_scores = np.load("results/acrobot_per_scores.npy")
    uniform_scores = np.load("results/acrobot_uniform_scores.npy")
    print(f"Loaded PER scores: {len(per_scores)} episodes")
    print(f"Loaded Uniform scores: {len(uniform_scores)} episodes")
except FileNotFoundError:
    print(
        "Error: Could not find .npy files. Make sure they are in the 'results' folder!"
    )
    exit()


window = 50
per_ma = moving_average(per_scores, window)
uniform_ma = moving_average(uniform_scores, window)


plt.figure(figsize=(10, 6))


plt.plot(
    uniform_ma, label="Standard DQN (Uniform)", color="blue", linewidth=2, alpha=0.8
)

plt.plot(uniform_scores, color="blue", alpha=0.1)


plt.plot(per_ma, label="DQN + PER (SumTree)", color="orange", linewidth=2)

plt.plot(per_scores, color="orange", alpha=0.1)


plt.title("Comparison: Prioritized vs Uniform Experience Replay (Acrobot-v1)")
plt.xlabel("Episodes")
plt.ylabel("Score (Moving Average)")
plt.legend()
plt.grid(True, alpha=0.3)

save_path = "results/comparison_graph_acrobot.png"
plt.savefig(save_path)
print(f"Graph saved to {save_path}")
plt.show()
