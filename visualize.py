import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, ArrayConversion
import numpy as np
import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
import random
import pygame
from gymnasium import Wrapper


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



class RewardDisplayWrapper(Wrapper):
    """Wrapper to display rewards and Q-values on pygame window"""
    def __init__(self, env, model, device):
        super().__init__(env)
        self.model = model
        self.device = device
        self.current_reward = 0
        self.cumulative_reward = 0
        self.current_q_values = None
        self.last_state = None
        self.action_names = ["Engine off", "Main engine", "Rotate left", "Rotate right"]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_reward = reward
        self.cumulative_reward += reward
        
        # Calculate Q-values for display
        with torch.no_grad():
            state_tensor = obs.unsqueeze(0).to(self.device) if isinstance(obs, torch.Tensor) else torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            self.current_q_values = self.model(state_tensor)[0].cpu().numpy()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Call original render
        self.env.render()
        
        # Get the pygame display and draw text
        try:
            pygame_display = pygame.display.get_surface()
            if pygame_display is None:
                return
            
            font = pygame.font.Font(None, 24)
            
            # Prepare text to display
            text_lines = [
                f"Current Reward: {self.current_reward:.4f}",
                f"Total Reward: {self.cumulative_reward:.4f}",
                "Q-values by action:"
            ]
            
            if self.current_q_values is not None:
                for i, (q_val, action_name) in enumerate(zip(self.current_q_values, self.action_names)):
                    text_lines.append(f"  {action_name}: {q_val:.4f}")
            
            # Render text on display
            y_offset = 10
            for line in text_lines:
                text_surface = font.render(line, True, (255, 255, 255))
                # Create a semi-transparent background
                text_rect = text_surface.get_rect()
                text_rect.topleft = (10, y_offset)
                
                # Draw background rectangle
                bg_rect = pygame.Rect(5, y_offset - 2, text_rect.width + 10, text_rect.height + 4)
                pygame.draw.rect(pygame_display, (0, 0, 0), bg_rect)
                
                # Draw text
                pygame_display.blit(text_surface, (10, y_offset))
                y_offset += 30
            
            pygame.display.flip()
        except:
            pass



if __name__ == "__main__":
    # Initialize LunarLander environment
    env = gym.make('LunarLander-v3')
    
    # Create model and load weights
    model = DQNNetExample(env.observation_space.shape[0], env.action_space.n)
    model_path = "weights.pth"
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found")
        exit(1)

    # Visualize the trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    gui_env = gym.make('LunarLander-v3', render_mode="human")
    gui_env = ArrayConversion(env=gui_env, env_xp=np, target_xp=torch)
    gui_env = RewardDisplayWrapper(gui_env, model, device)

    
    state, _ = gui_env.reset()
    state = state.to(device)
    done = False

    while not done:
        with torch.no_grad():
            action = model.act(state.unsqueeze(0)).item()

        state, reward, terminated, truncated, info = gui_env.step(action)
        state = state.to(device)

        gui_env.render()
        
        done = terminated or truncated
    
    print(f"\n=== Episode finished ===")
    print(f"Total episode reward: {gui_env.cumulative_reward:.2f}")
    gui_env.close()