import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

TEAM_NAME = "COCHARD_GIRARDOT"  # Replace with your actual team name

# =========================================================
# Model Architecture (Must match training exactly)
# =========================================================
class CarRacingDQN(nn.Module):
    def __init__(self, num_actions=5):
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

# =========================================================
# Environment Wrapper
# =========================================================
class EnvironmentWrapper(gym.Wrapper):
    """
    Combines ActionRepeat (4), PreprocessFrame (Gray+Resize 84x84), and FrameStack (4).
    """
    def __init__(self, env, k=4, repeat=4):
        super().__init__(env)
        self.k = k
        self.repeat = repeat
        self.frames = deque(maxlen=k)
        
        # Define the observation space to match the output (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(k, 84, 84), dtype=np.float32
        )

    def _process_frame(self, frame):
        # 1. Convert to Grayscale
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize to 84x84 (Matching your training code)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 3. Normalize
        frame = frame.astype(np.float32) / 255.0
        return frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_frame = self._process_frame(obs)
        
        # Clear buffer and stack the initial frame k times
        for _ in range(self.k):
            self.frames.append(processed_frame)
            
        return self._get_stacked_obs(), info

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        last_obs = None

        # --- Action Repeat Logic ---
        # Repeat the action 'self.repeat' times (4 times)
        for _ in range(self.repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            last_obs = obs
            
            if term or trunc:
                done = term or trunc
                break
        
        # Process the last frame observed during the repeat
        if last_obs is not None:
            processed_frame = self._process_frame(last_obs)
            self.frames.append(processed_frame)

        return self._get_stacked_obs(), total_reward, done, truncated, info

    def _get_stacked_obs(self):
        # Stack frames to shape (k, 84, 84) -> (4, 84, 84)
        return np.stack(self.frames, axis=0)

# =========================================================
# Policy Interface
# =========================================================
class Policy:
    def __init__(self):
        self.device = torch.device("cpu")  # Submission usually runs on CPU
        # 5 discrete actions for CarRacing-v3 discrete
        self.model = CarRacingDQN(num_actions=5).to(self.device)

    def act(self, state):
        """
        Input state shape: (4, 84, 84) (from EnvironmentWrapper)
        """
        # Add batch dimension: (1, 4, 84, 84)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.model.act(state_tensor).item()
            
        return action

    def load(self, path):
        try:
            # Map location ensures weights load even if trained on GPU
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded model from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")