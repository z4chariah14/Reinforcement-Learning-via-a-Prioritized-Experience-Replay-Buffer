import numpy as np
from src.sum_tree import SumTree
from collections import deque
import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.2, beta=0.6, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.beta_frames = beta_frames
        self.frame = 0
        self.Tree = SumTree(capacity)
        self.max_priority = 1.0

    def beta_by_frame(self):
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames)

    def add_priorities(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.Tree.add(self.max_priority, experience)
        self.frame += 1

    def update_priorities(self, td_errors, indices):
        for idx, error in zip(indices, td_errors):
            new_priority = (abs(error) + 1e-5) ** self.alpha
            self.Tree.update(idx, new_priority)
            self.max_priority = max(self.max_priority, new_priority)

    def sample(self, batch_size):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = (
            [],
            [],
            [],
            [],
            [],
        )
        indices = np.zeros(batch_size, dtype=int)
        weights = np.zeros(batch_size, dtype=np.float32)

        total_p = self.Tree.totalPriority()
        segment = total_p / batch_size

        for i in range(batch_size):
            s = np.random.uniform(low=segment * i, high=segment * (i + 1))
            #possible fix if s >= self.Tree.totalPriority(): s = self.Tree.totalPriority() - 1e-5
            idx, priority, data = self.Tree.get_leaf(s)
            
            
            
            if data == 0:
                while data == 0:
                    # used a completely random sample instead, rejecting sampling
                    s = np.random.uniform(0, self.Tree.totalPriority())
                    idx, priority, data = self.Tree.get_leaf(s)

            s_data, a, r, s_prime, done_flag = data

            prob = priority / total_p

            weight = (1 / (self.Tree.capacity * prob)) ** self.beta if prob > 0 else 0

            indices[i] = idx
            weights[i] = weight

            batch_states.append(s_data)
            batch_actions.append(a)
            batch_rewards.append(r)
            batch_next_states.append(s_prime)
            batch_dones.append(done_flag)

        self.beta_by_frame()

        max_weight = weights.max()
        if max_weight > 0:
            weights /= max_weight

        return (
            np.array(batch_states),
            np.array(batch_actions),
            np.array(batch_rewards),
            np.array(batch_next_states),
            np.array(batch_dones),
            weights,
            indices,
        )

class StandardReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.frame = 0

    def add_priorities(self, state, action, reward, next_state, done):
        """
        Simple FIFO storage.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.frame += 1 

    def sample(self, batch_size):
        """
        Uniform Random Sampling.
        Returns dummy weights (ones) and indices (zeros) to match PER API.
        """
        batch = random.sample(self.buffer, batch_size)

        # Unpack the data
        state, action, reward, next_state, done = zip(*batch)
        
    
        # Weights are all 1.0 (no importance sampling needed)
        weights = np.ones(batch_size, dtype=np.float32)
        # Indices are not needed for deque, return dummy zeros
        indices = np.zeros(batch_size, dtype=int)

        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done), weights, indices)

    def update_priorities(self, td_errors, indices):
        pass