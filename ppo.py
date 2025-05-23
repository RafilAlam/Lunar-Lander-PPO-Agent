import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

def init_weights(m):
    """Initialize weights for layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, dones):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(dones)

    def clear_memory(self):
        self.states.clear()
        self.probs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.vals.clear()

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dim=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dims),
            nn.ReLU(),
        )
        self.mean = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Parameter(T.zeros(n_actions))  # Learnable std dev

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(init_weights)

    def forward(self, state):
        features = self.actor(state)
        mean = self.mean(features)
        std = T.exp(self.log_std)  # Convert log std to std
        return mean, std

    def sample_normal(self, state):
        mean, std = self.forward(state)
        std = T.clamp(std, min=1e-6)  # Clamp std to prevent numerical issues
        dist = Normal(mean, std)
        
        action = dist.sample()
        action = 2 * T.tanh(action)  # Ensure action is within [-2, 2]

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # Sum over action dimensions
        return action, log_prob, dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(init_weights)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10,
                 entropy_coef=0.01):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.running_mean = 0.0
        self.running_var = 1.0
        self.epsilon = 1e-8

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def normalize_reward(self, reward):
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        self.running_mean = 0.99 * self.running_mean + 0.01 * reward
        self.running_var = 0.99 * self.running_var + 0.01 * (reward - self.running_mean) ** 2
        normalized_reward = (reward - self.running_mean) / (T.sqrt(self.running_var) + self.epsilon)
        assert not T.isnan(normalized_reward).any() and not T.isinf(normalized_reward).any(), "Invalid reward normalization!"
        return normalized_reward

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        with T.no_grad():
            action, log_prob, val = self.actor.sample_normal(state)
            value = self.critic(state)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            last_adv = 0

            for t in reversed(range(len(reward_arr) - 1)):
                delta = reward_arr[t] + self.gamma * vals_arr[t + 1] * (1 - int(dones_arr[t])) - vals_arr[t]
                advantage[t] = last_adv = delta + self.gamma * self.gae_lambda * last_adv

            advantage = T.tensor(advantage).to(self.actor.device)
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)

                mean, std = self.actor.forward(states)
                std = T.clamp(std, min=1e-4)
                dist = Normal(mean, std)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions).sum(1, keepdim=True)
                prob_ratio = T.exp(new_probs - old_probs)

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]

                entropy_loss = dist.entropy().mean()
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() - self.entropy_coef * entropy_loss

                returns = advantage[batch] + critic_value.detach()
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimiser.zero_grad()
                self.critic.optimiser.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor.optimiser.step()
                self.critic.optimiser.step()

        self.memory.clear_memory()
