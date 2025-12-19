import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

TEAM_NAME = "COCHARD_GIRARDOT"

# Neural Network Architecture
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Policy
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)       )
        
        # Value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value

class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)

class Policy:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = ActorCritic().to(self.device)

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(state)
            action = torch.argmax(logits).item()
        return action

    def load(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model from {path}: {e}")