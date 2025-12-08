import gymnasium as gym
import torch
from src.agent import DQNAgent

env = gym.make("CartPole-v1", render_mode="human")

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQNAgent(input_dim, output_dim)

try:
    agent.policy_net.load_state_dict(torch.load("models/GymAgent.pth"))
    agent.policy_net.eval() 
    print("Model loaded successfully!")
except FileNotFoundError:
    print("No model found! Run training first.")
    exit()

state, info = env.reset()
done = False
total_reward = 0

while not done:
    agent.epsilon = 0.0
    action = agent.act(state)

    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Game Over! Score: {total_reward}")
env.close()
