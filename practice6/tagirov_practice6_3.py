import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, batch_size=128,
                 epsilon=0.4, epoch_n=30, pi_lr=1e-4, v_lr=5e-2):

        super().__init__()

        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, action_dim))

        self.v_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 1))

        self.gamma = gamma
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.batch_size = batch_size
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

    def get_action(self, state):
        logits = self.pi_model(torch.FloatTensor(state))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.numpy()

    def fit(self, states, next_states, actions, rewards, dones):

        states, next_states, actions, rewards, dones = map(np.array, [states, next_states, actions, rewards, dones])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, next_states, actions, rewards = map(torch.FloatTensor, [states, next_states, actions, rewards])

        logits = self.pi_model(states)
        dist = Categorical(logits=logits)
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(rewards.shape[0])
            for i in range(0, rewards.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_states = states[b_idxs]
                b_next_states = next_states[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_rewards + self.gamma * self.v_model(b_next_states) - self.v_model(b_states)

                b_logits = self.pi_model(b_states)
                b_dist = Categorical(logits=b_logits)
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon, 1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = 3

    agent = PPO(state_dim, action_dim)

    episode_n = 25
    trajectory_n = 50

    total_rewards = []
    ppo_v1_totals_rewards = []

    for attempt in range(3):
        for episode in range(episode_n):

            states, next_states, actions, rewards, dones = [], [], [], [], []

            for _ in range(trajectory_n):
                total_reward = 0

                state = env.reset()
                for t in range(200):
                    states.append(state)

                    action = agent.get_action(state)
                    actions.append(action)

                    next_state, reward, done, _ = env.step(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                    dones.append(done)

                    state = next_state

                    total_reward += reward

                total_rewards.append(total_reward)

            agent.fit(states, next_states, actions, rewards, dones)

    plt.plot(total_rewards)
    plt.title('Total Rewards')
    plt.grid()
    plt.show()
