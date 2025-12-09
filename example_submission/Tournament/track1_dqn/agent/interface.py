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
TEAM_NAME = "Your Team Name Here"



class NetExample(nn.Module):
    """
        Same model as in train_example.py
    """
    def __init__(self, input_size, output_size) -> None:
        super(NetExample, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        """
            Example of Q net
        """
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=-1)

        """
            Example of PG policy net
        """
        logits = self.forward(x)
        distributions = torch.distributions.Categorical(logits=logits)

        # Sample from the distribution
        actions = distributions.sample()

        # You can also get logprobs for sampled actions like so:
        log_probs = distributions.log_prob(actions)

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
        self.model = NetExample(input_size=lunar_obs_dim,
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
