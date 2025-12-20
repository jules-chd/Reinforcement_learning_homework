# train_ex3_Max4.py

import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from gymnasium.wrappers import TimeLimit


class PreprocessFrame(gym.ObservationWrapper):
    """
    Convert RGB (96,96,3) -> grayscale (84,84) and normalize
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs

class FrameStack(gym.Wrapper):
    """
    Stack last k frames to capture velocity
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(k, 84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames), reward, terminated, truncated, info

class ActionRepeat(gym.Wrapper):
    """
    Repeat the same action for N environment steps
    """
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

class CarRacingDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @torch.no_grad()
    def act(self, x):
        return torch.argmax(self.forward(x), dim=1)

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.cat(states, dim=0),        
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.cat(next_states, dim=0),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

def train(
    env,
    model,
    optimizer,
    episodes=1200,
    gamma=0.99,
    batch_size=128,       
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.998, 
    target_update_steps=500, 
    action_repeat=4 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target_model = CarRacingDQN(env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())

    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start
    global_step = 0

    best_reached = 300

    for ep in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state).unsqueeze(0).to(device)
        done = False
        episode_reward = 0

        while not done:
            global_step += 1

            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.act(state).item()

            # Repeat action for action_repeat frames
            total_reward = 0
            for _ in range(action_repeat):
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if done:
                    break

            episode_reward += total_reward
            next_state_t = torch.tensor(next_state).unsqueeze(0).to(device)
            replay_buffer.push(state.cpu(), action, total_reward, next_state_t.cpu(), done)
            state = next_state_t

            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = states.to(device), actions.to(device), rewards.to(device), next_states.to(device), dones.to(device)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = model(next_states).argmax(1)
                    next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards + gamma * next_q * (1 - dones)

                loss = F.smooth_l1_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            if global_step % target_update_steps == 0:
                target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {ep+1}/{episodes} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f}")

        # Save model if reward exceeds thresholds
        if episode_reward > 300:
            torch.save(model.state_dict(), f"car_racing_reward_{int(episode_reward)}.pth")
        if episode_reward > best_reached:
            torch.save(model.state_dict(), f"car_racing_reward_{int(episode_reward)}.pth")
            best_reached = episode_reward
            print(f"new best reached {best_reached}")

if __name__ == "__main__":
    env = gym.make(
        "CarRacing-v3",
        continuous=False,
        render_mode=None
    )

    env = TimeLimit(env, max_episode_steps=1000)
    env = ActionRepeat(env, repeat=4)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)

    model = CarRacingDQN(env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(env, model, optimizer)