import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from temp.ex2.interface import ActorCritic

# parameters
ENV_NAME = "LunarLander-v3"
LEARNING_RATE = 3e-4 
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_TIMESTEPS = 600000
UPDATE_TIMESTEPS = 2048
K_EPOCHS = 10 
BATCH_SIZE = 64

def compute_gae(rewards, values, masks, next_value, gamma, lam):
    returns = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
        next_value = values[step]
    return returns

def train():
    env = gym.make(ENV_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = ActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    global_step = 0
    state, _ = env.reset(seed=42)
    
    ep_reward = 0
    ep_rewards_history = deque(maxlen=20)
    best_mean_reward = -float('inf')
    highest_ep_score = -100 # just to save model that beats this score   
    num_updates = TOTAL_TIMESTEPS // UPDATE_TIMESTEPS

    for update in range(1, num_updates + 1):
        # Learning Rate Decay
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        # Data Collection
        states_list, actions_list, log_probs_list = [], [], []
        rewards_list, dones_list, values_list = [], [], []
        
        for _ in range(UPDATE_TIMESTEPS):
            global_step += 1
            
            state_tensor = torch.FloatTensor(state).to(device)
            states_list.append(state_tensor)
            
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(state_tensor)
                
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            ep_reward += reward
            
            values_list.append(value.item())
            log_probs_list.append(log_prob)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(0.0 if done else 1.0)
            
            state = next_state
            
            if done:
                ep_rewards_history.append(ep_reward)
                
                # Save the best models so far (if better than highest_ep_score)
                if ep_reward > highest_ep_score:
                    highest_ep_score = ep_reward
                    clean_score = int(ep_reward)
                    
                    print(f"Best score : {clean_score}. Saving weights_{clean_score}.pth")
                    torch.save(agent.state_dict(), f"weights_{clean_score}.pth")

                ep_reward = 0
                state, _ = env.reset()

        if len(ep_rewards_history) > 0:
            avg_rew = np.mean(ep_rewards_history)
            if update % 5 == 0:
                print(f"Step: {global_step} | Avg (last 20): {avg_rew:.2f} | LR: {lrnow:.2e}")
            
            if avg_rew > best_mean_reward:
                best_mean_reward = avg_rew
                print(f"--> New Best Avg: {best_mean_reward:.2f}. Saving weights.pth")
                torch.save(agent.state_dict(), "weights.pth")

        # GAE and advantage
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state).to(device)
            _, next_value = agent(next_state_tensor)
            next_value = next_value.item()
            
        returns = compute_gae(rewards_list, values_list, dones_list, next_value, GAMMA, GAE_LAMBDA)
        
        b_returns = torch.tensor(returns).float().to(device)
        b_values = torch.tensor(values_list).float().to(device)
        b_states = torch.stack(states_list)
        b_actions = torch.stack(actions_list)
        b_log_probs = torch.stack(log_probs_list)
        b_advantages = b_returns - b_values
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO Update
        b_inds = np.arange(UPDATE_TIMESTEPS)
        for epoch in range(K_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, UPDATE_TIMESTEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]
                
                mb_states = b_states[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_old_log_probs = b_log_probs[mb_inds]
                
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(mb_states, mb_actions)
                new_values = new_values.squeeze(1)

                logratio = new_log_probs - mb_old_log_probs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                
                loss = pg_loss + VALUE_LOSS_COEF * v_loss - ENTROPY_COEF * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

    print("Training finished.")
    env.close()

if __name__ == "__main__":
    train()