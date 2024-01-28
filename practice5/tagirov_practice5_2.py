import gym
import wandb
import torch
import random
import numpy as np
import torch.nn as nn

from tagirov_practice5_1 import DQN, Qfunction


class DQNHardTarget(DQN):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01,
                 epsilon_min=0.01, target_upd_freq=100):
        super().__init__(state_dim, action_dim, gamma, lr, batch_size, epsilon_decrease, epsilon_min)
        self.target_q_function = Qfunction(self.state_dim, self.action_dim)
        self.target_q_function.load_state_dict(self.q_function.state_dict())
        self.target_upd_freq = target_upd_freq

    def fit(self, state, action, reward, done, next_state, step):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.target_q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease

            if (step + 1) % self.target_upd_freq:
                self.update_target_q_function()

    def update_target_q_function(self):
        self.target_q_function.load_state_dict(self.q_function.state_dict())


class DQNSoftTarget(DQN):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01,
                 epsilon_min=0.01, tau=0.01):
        super().__init__(state_dim, action_dim, gamma, lr, batch_size, epsilon_decrease, epsilon_min)
        self.target_q_function = Qfunction(self.state_dim, self.action_dim)
        self.target_q_function.load_state_dict(self.q_function.state_dict())
        self.tau = tau

    def fit(self, state, action, reward, done, next_state, step):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.target_q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.update_target_q_function()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease

    def update_target_q_function(self):
        for param, target_param in zip(self.q_function.parameters(), self.target_q_function.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class DoubleDQN(DQNSoftTarget):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=3, epsilon_decrease=0.01,
                 epsilon_min=0.01, tau=0.01):
        super().__init__(state_dim, action_dim, gamma, lr, batch_size, epsilon_decrease, epsilon_min, tau)

    def fit(self, state, action, reward, done, next_state, step):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            target_q_argmax_actions = torch.argmax(self.target_q_function(next_states), dim=1)
            targets = rewards + self.gamma * (1 - dones) \
                      * self.q_function(next_states)[torch.arange(self.batch_size), target_q_argmax_actions]

            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease

            self.update_target_q_function()


def run(env, agent, episode_n, trajectories_n):
    np.random.seed(42)
    torch.manual_seed(42)

    for episode in range(episode_n):
        total_reward = 0

        state = env.reset()
        for t in range(trajectories_n):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            agent.fit(state, action, reward, done, next_state, t)

            state = next_state

            if done:
                break

        wandb.log({'reward': total_reward})
        print(f'episode: {episode}, total_reward: {total_reward}')


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    episode_n = 100
    trajectories_n = 500

    params = {
        'models': ['DQNHardTarget', 'DQNSoftTarget', 'DoubleDQN'],
        'lr': [0.001],
        'tau': [0.01, 0.05, 0.1],
        'target_upd_freq': [50, 100, 200]
    }
    wandb.login(key='8224f98472fd102b8a3c3806a80663939495aade')

    for model_name in params['models']:
        for lr in params['lr']:
            for target_upd_freq, tau in zip(params['target_upd_freq'], params['tau']):
                config = {
                    'trajectories': trajectories_n,
                    'learning_rate': lr,
                    'batch_size': 64,
                    'target_update_frequency': target_upd_freq,
                    'tau': tau
                }

                if model_name == 'DQNHardTarget':
                    agent = locals()[model_name](state_dim, action_dim, lr=lr, target_upd_freq=target_upd_freq)
                    name = f'model_{model_name}_lr={lr}_target_upd_freq={target_upd_freq}'
                    del config['tau']
                else:
                    agent = locals()[model_name](state_dim, action_dim, lr=lr, tau=tau)
                    name = f'model_{model_name}_lr={lr}_tau={tau}'
                    del config['target_update_frequency']

                wandb.init(project=f"DQN vs other", config=config, name=name)

                run(env, agent, episode_n, trajectories_n)
                wandb.finish()
