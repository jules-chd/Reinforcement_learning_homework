import gymnasium as gym
from gymnasium.wrappers import ArrayConversion, RecordEpisodeStatistics 
import numpy as np
from time import sleep
import torch
import torch.nn as nn
from typing import Tuple

"""
    Please make sure this script does not import anything outside of
    - the standard library,
    - PyTorch, and
    - the packages listed in requirements.txt.
"""


# Specify your team name in a global variable like so:
TEAM_NAME = "COCHARD_GIRARDOT"



class DQNNetExample(nn.Module):
    """
        Same model as in train_ex1.py
    """
    def __init__(self, input_size, output_size) -> None:
        super(DQNNetExample, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        """
            DQN-style policy
        """
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=-1)
        return actions

    def save(self, path : str) -> None:
        torch.save(self.state_dict(), path)


class Policy:
    """
        Basic policy wrapper that enables loading and evaluating a pre-trained
        neural network policy.

        Modify the logic in the load and act functions to reflect your
        specific model architecture and other needs.

        Your agent will be evaluated by:
            a) Loading the approapriate model weights by calling the method load("agent/weights.pth")
            b) Interacting with the agent via act(observation) 
    """

    def __init__(self):
        self.model = None

    def load(self, model_path : str) -> None:
        """
            Construct and load pre-trained model from specified path
        """
        lunar_action_count = 4
        lunar_obs_dim = 8
        self.model = DQNNetExample(input_size=lunar_obs_dim,
                                   output_size=lunar_action_count)
        self.model.load_state_dict(torch.load(model_path))

        # Set mode to eval (no gradient accumulation, etc.)
        self.model.eval()


    def act(self, observation : torch.Tensor) -> int:
        """
            Get action for a single observation from policy. 
        """
        if self.model is None:
            raise ValueError("Model was not loaded successfuly. Call load(path) before act().")

        # Get action from model
        action = self.model.act(observation)

        # Convert the single action to an integer for gym
        action = action.item()
        return action


class EnvironmentWrapper:
    """
        Basic environment wrapper that enables us to evaluate with the same
        environment transformations as you did during training.

        If you've used any wrappers that transform observations, actions,
        rewards and you want them to be applied during evaluation, you need to
        supply them here. See the example in the constructor.

        Your agent will interact with the evaluation environment via:
            a) Constructing the EnvWrapper with the env instance
            b) Applying wrappers you used in __init__
            c) Interacting with the env via reset() and step()

        You need to implement steps b) and c) as needed.

    """

    def __init__(self, env : gym.Env) -> None:
        """
            env - instance of a gym environment that will be wrapped
            

            Since we applied the ArrayConversion wrapper during training,
            we also need to apply it here to take effect during evaluation.
        """
        self.wrapped_env = ArrayConversion(env=env, env_xp=np, target_xp=torch)

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        return self.wrapped_env.reset(*args, **kwargs)

    def step(self, action : int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # You can also adjust the interaction here if needed.
        return self.wrapped_env.step(action)
