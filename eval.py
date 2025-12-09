import gymnasium as gym
import importlib
import argparse
import numpy as np

"""
Evaluate a policy for a given environment.
"""

env_names = ["LunarLander-v3", "CarRacing-v3"]

class eval_wrapper(gym.Wrapper):
    """Gym style wrapper for evaluation."""
    def __init__(self, env):
        super().__init__(env)
        self.ep_rews = []
        self.ep_rew = 0

    def reset(self, **kwargs):
        self.ep_rews.append(self.ep_rew)
        self.ep_rew = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.ep_rew += reward
        return obs, reward, done, truncated, info

    def render(self):
        return self.env.render()

def run(track, load_path, n_eps):
    env = gym.make(env_names[track // 2], continuous=False)

    module_name = load_path.replace("/", ".")
    module_name += f".interface"
    interface = importlib.import_module(module_name)

    policy = interface.Policy()
    load_path = load_path + f"/weights.pth"
    policy.load(load_path)

    env.reset(seed=42)
    eval_env = eval_wrapper(env)
    env = interface.EnvironmentWrapper(eval_env)

    # run evaluation loop
    for ep in range(n_eps):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = policy.act(obs)
            obs, reward, done, truncated, info = env.step(action)

    env.reset()

    return np.mean(eval_env.ep_rews), np.median(eval_env.ep_rews), np.std(eval_env.ep_rews), np.max(eval_env.ep_rews), np.min(eval_env.ep_rews),  interface.TEAM_NAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, required=True, help="Track id: DQN=0, Policy gradients=1, Cars=2, Cars+Pretraining=3")
    parser.add_argument('--load_path', type=str, required=True, help="Path to the submission folder")
    parser.add_argument('--eps', type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument('--results_path', type=str, default="results.csv", help="Path to the CSV file to store results")
    args = parser.parse_args()
    mean, median, std, max, min, team_name = run(args.track, args.load_path, args.eps)

    with open(args.results_path, "a") as f:
        f.write(f"{team_name},{args.track},{mean},{median},{std},{max},{min}\n")



