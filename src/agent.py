import torch
import torch.optim as optim
import numpy as np
import random
from src.buffer import PrioritizedReplayBuffer
from src.network import DQN


class DQNAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=1e-4,
        gamma=0.99,
        batch_size=32,
        buffer_capacity=100000,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.lr = lr

        self.policy_net = DQN(input_dim, output_dim)
        self.target_net = DQN(input_dim, output_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is only for prediction, not training

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.epsilon = 1.0  # fro decay annealing

    def act(self, state):
        p = random.random()
        if p < self.epsilon:
            random_action = random.choice([0, 1])
            return random_action
        else:
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.policy_net(state_t)
            action = q_values.argmax().item()  # return index
            return action

    def learn(self):
        if self.buffer.frame < self.batch_size:
            return
        s, a, r, n_s, done, weights, indices = self.buffer.sample(self.batch_size)
        s_tensor = torch.tensor(s, dtype=torch.float)
        actions_tensor = torch.tensor(a, dtype=torch.long).unsqueeze(1)
        rewards_tensor = torch.tensor(r, dtype=torch.float)
        next_states_tensor = torch.tensor(n_s, dtype=torch.float)
        done_tensor = torch.tensor(done, dtype=torch.float)
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        Q_values = self.policy_net(s_tensor).gather(1, actions_tensor)
        Q_targets = rewards_tensor + self.gamma * self.target_net(
            next_states_tensor
        ).max(1)[0] * (1 - done_tensor)
        TD_error = Q_targets.unsqueeze(1) - Q_values
        self.buffer.update_priorities(TD_error, indices)
        Loss = torch.mean((TD_error) ** 2 * weights_tensor)
        Loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        