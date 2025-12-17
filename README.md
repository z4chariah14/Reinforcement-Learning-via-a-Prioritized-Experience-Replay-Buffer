# Deep Q-Network (DQN) vs. Prioritized Experience Replay (PER)

This project implements a Deep Q-Network (DQN) from scratch and compares two experience replay strategies: **Standard Uniform Replay** and **Prioritized Experience Replay (PER)**.

The goal is to evaluate whether prioritizing high TD-error (more surprising) experiences improves learning speed in classic reinforcement learning environments.

## Project Structure

- `src/sum_tree.py`  
  Custom binary Sum Tree data structure implemented with NumPy, enabling O(log n) sampling.

- `src/buffer.py`  
  Contains both the `StandardReplayBuffer` (deque-based) and the `PrioritizedReplayBuffer` (Sum Tree–based).

- `src/network.py`  
  PyTorch neural network architecture used by the agent.

- `src/agent.py`  
  Agent class responsible for action selection and learning logic.

- `train.py`  
  Main training loop for CartPole and Acrobot environments.

- `plot_results.py`  
  Script to generate performance comparison plots.

## Dependencies

Python 3.10+ is required, along with the following libraries:

```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

## How to Run

1. **Train the agent**

   Open `train.py` and run the training loop. You can switch between PER and the standard replay buffer by setting `use_per=True` or `False` in the code.

   ```bash
   python train.py
   ```

2. **Visualize results**

   After training both versions and saving the results as `.npy` files, saving models as `.pth`, run the plotting script to compare performance:

   ```bash
   python test.py
   ```

   ```bash
   python plot_results.py
   ```

## Results

The agents were evaluated on **CartPole-v1** and **Acrobot-v1**.

Contrary to the initial hypothesis, the **standard DQN with uniform replay** learned faster than the PER-based agent in these environments.

**Explanation:**

- **Dense rewards**  
  Both environments provide feedback at every time step, so there are no rare or critical transitions that benefit strongly from prioritization.

- **Computational overhead**  
  Maintaining the Sum Tree and computing importance-sampling weights introduced additional overhead, slowing early convergence.

These results highlight a practical version of the no free lunch principle: more complex algorithms do not necessarily outperform simpler ones on relatively easy problems.
