import gymnasium as gym
from src.agent import DQNAgent
import torch
import numpy as np

env = gym.make("Acrobot-v1")

state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n


Agent = DQNAgent(state_shape, num_actions, use_per=False)
scores = []

episode = 1000 
batch_size = 64
update_freq = 1000
frame_counter = 1


def get_epsilon(current_frame):
    epsilon_start = 1.0
    epsilon_end = 0.05
    decay_frames = 20000

    epsilon = max(
        epsilon_end,
        epsilon_start - (current_frame / decay_frames) * (epsilon_start - epsilon_end),
    )
    return epsilon


for i in range(episode):
    state, info = env.reset()
    episode_reward = 0
    while True:
        frame_counter += 1
        action = Agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        Agent.buffer.add_priorities(state, action, reward, next_state, done)

        Agent.learn()

        Agent.epsilon = get_epsilon(frame_counter)
        if frame_counter % update_freq == 0:
            Agent.sync_target()

        state = next_state
        episode_reward += reward
        if done:
            break
    scores.append(episode_reward)
    if i % 10 == 0:
        print(f"Episode: {i}, Score: {episode_reward:.1f}, Eps: {Agent.epsilon:.2f}")

    avg_score = np.mean(scores[-100:])
    if avg_score > -100 and i > 50:
        # torch.save(Agent.policy_net.state_dict(), "models/GymAgent.pth") - cartpole
        #torch.save(Agent.policy_net.state_dict(), "models/Acrobot_PER.pth")
        np.save("results/acrobot_uniform_scores.npy", np.array(scores))
        break
