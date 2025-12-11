import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, ArrayConversion
from gymnasium import Wrapper
import numpy as np
import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
import random


class DQNNetExample(nn.Module):
    """
        Example of a neural network class used for instantiating the policy
        wrapper.

        Replace with your architecture of choice for a given problem.

        The interface of the model is entirely up to you, the only requirement
        is that agent/agent.py is able to load and evaluate your trained agent.
    """
    def __init__(self, input_size, output_size) -> None:
        super(DQNNetExample, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        """
            This logic reflects a DQN-style policy,
        """
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=-1)
        return actions

    def save(self, path : str) -> None:
        torch.save(self.state_dict(), path)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)



def train_model(env, model, optimizer, gamma=0.99, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, batch_size=64, target_update_frequency=5, learning_rate=0.0005):
    """
    Train the model using DQN algorithm
    """
    # Set max episode steps to 1000
    env = TimeLimit(env=env, max_episode_steps=1000)
    # Convert environment to produce torch tensors
    env = ArrayConversion(env=env, env_xp=np, target_xp=torch)
    # Wrap environment to record episode stats for last 10 episodes
    env = RecordEpisodeStatistics(env=env, stats_key='episode', buffer_length=10)

    # Create target network
    target_model = DQNNetExample(env.observation_space.shape[0], env.action_space.n)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)

    epsilon = epsilon_start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_model = target_model.to(device)

    for ep in range(episodes):
        state, _ = env.reset()
        state = state.to(device)
        done = False
        episode_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state.unsqueeze(0))
                    action = torch.argmax(q_values, dim=-1).item()
                    # Print Q-values for all actions (uncomment for debugging)
                    # print(f"Q-values: {q_values[0].cpu().numpy()}, Selected action: {action}")

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.to(device)
            done = terminated or truncated
            episode_reward += reward

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Train on batch from replay buffer
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # Current Q-values
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Target Q-values using target network
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and update model
                loss = F.mse_loss(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if (ep + 1) % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        # Print progress with reward info
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{episodes}, Epsilon: {epsilon:.3f}, Episode Reward: {episode_reward:.2f}")
        
        if (ep + 1) % 30 == 0:
            gui_env = gym.make('LunarLander-v3', render_mode="human")
            gui_env = ArrayConversion(env=gui_env, env_xp=np, target_xp=torch)
            state, _ = gui_env.reset()
            state = state.to(device)
            done = False
            while not done:
                with torch.no_grad():
                    action = model.act(state.unsqueeze(0)).item()
                state, reward, terminated, truncated, info = gui_env.step(action)
                state = state.to(device)
                done = terminated or truncated



if __name__ == "__main__":
    # Initialize LunarLander environment
    env = gym.make('LunarLander-v3')

    # Initialize model and optimizer
    model = DQNNetExample(env.observation_space.shape[0], env.action_space.n)
    # Train the model using DQN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train_model(env, model, optimizer)

    # Save the trained agent for evaluation
    model.save("weights.pth")

    # Visualize the trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    gui_env = gym.make('LunarLander-v3', render_mode="human")
    gui_env = ArrayConversion(env=gui_env, env_xp=np, target_xp=torch)
    state, _ = gui_env.reset()
    state = state.to(device)
    done = False
    while not done:
        with torch.no_grad():
            action = model.act(state.unsqueeze(0)).item()
        state, reward, terminated, truncated, info = gui_env.step(action)
        state = state.to(device)
        done = terminated or truncated
