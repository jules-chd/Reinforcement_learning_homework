import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, ArrayConversion
import numpy as np
import torch as torch
import torch.nn as nn
from torch.nn import functional as F



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
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layer(x)

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



def train_model(env, model, optimizer, gamma=0.99, episodes=100):

        """
            Train the model to predict the value of the uniform random policy

            You can utilize predefined env wrappers from gymnasium:
        """
        # Set max episode steps to 1000
        env = TimeLimit(env=env, max_episode_steps=1000)
        # Convert environment to produce torch tensors
        env = ArrayConversion(env=env, env_xp=np, target_xp=torch)
        # Wrap environment to record episode stats for last 10 episodes
        env = RecordEpisodeStatistics(env=env, stats_key='episode', buffer_length=10)

        state, _ = env.reset()
        done = False

        for ep in range(episodes):
            while not done:
                action = env.action_space.sample() # Sample random action
                q = model(state)[action] # Get Q-value for the action
                next_state, reward, terminated, truncated, _ = env.step(action) # Perform step
                done = terminated or truncated # Check if the episode ended or was truncated due to time limit

                # Compute loss and update model
                next_action = env.action_space.sample() # Next action according to the policy
                next_q = model(next_state)[next_action] # Next Q-value
                with torch.no_grad(): # We use semi-gradient descent, so we don't backpropagate through the target
                    target = reward + gamma * next_q * (1 - done) # Temporal difference target
                loss = F.mse_loss(q, target) # Mean squared error loss between current Q estimate and target

                optimizer.zero_grad() # Set gradients buffers to zero
                loss.backward() # Backpropagate the loss -> compute gradients
                optimizer.step() # Update model parameters based on the gradients

                state = next_state



if __name__ == "__main__":
    # Initialize LunarLander environment
    env = gym.make('LunarLander-v3')

    # Initialize model and optimizer
    model = DQNNetExample(env.observation_space.shape[0], env.action_space.n)
    # Train the model to predict the value of the uniform random policy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(env, model, optimizer)

    # Save the trained agent for evaluation
    # model.save("weights.pth")

    # Visualize the random policy
    gui_env = gym.make('LunarLander-v3', render_mode="human")
    state, _ = gui_env.reset()
    done = False
    while not done:
        action = gui_env.action_space.sample()
        state, reward, terminated, truncated, info = gui_env.step(action)
        done = terminated or truncated
