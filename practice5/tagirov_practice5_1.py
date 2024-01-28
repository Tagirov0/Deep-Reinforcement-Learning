import gym
import wandb
import torch
import random
import numpy as np
import torch.nn as nn


class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states):
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions


class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrease


def run(env, episode_n, trajectories_n, lr, batch_size, epsilon_decrease):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    np.random.seed(42)
    torch.manual_seed(42)
    agent = DQN(state_dim, action_dim, lr=lr, batch_size=batch_size, epsilon_decrease=epsilon_decrease)

    for episode in range(episode_n):
        total_reward = 0

        state = env.reset()
        for t in range(trajectories_n):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            agent.fit(state, action, reward, done, next_state)

            state = next_state

            if done:
                break

        wandb.log({'reward': total_reward})
        print(f'episode: {episode}, total_reward: {total_reward}')


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')

    params = {
        'episode_n': [100, 250, 500],
        'trajectories_n': [300, 500, 1000],
        'lr': [0.001, 0.005, 0.01, 0.05],
        'batch_size': [32, 64, 128],
        'epsilon_decrease': [0.005, 0.01]
    }
    wandb.login(key='8224f98472fd102b8a3c3806a80663939495aade')

    for episode_n in params['episode_n']:
        for trajectory_n in params['trajectories_n']:
            for lr in params['lr']:
                for batch_size in params['batch_size']:
                    for epsilon_decrease in params['epsilon_decrease']:
                        wandb.init(
                            project=f"DQN",
                            config={
                                'episode_n': episode_n,
                                'trajectories': trajectory_n,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'epsilon_decrease': epsilon_decrease
                            },
                            name=f'episode_n={episode_n}_trajectories={trajectory_n}_lr={lr}_batch_size={batch_size}'
                        )

                        # print(f'episode_n={episode_n}_trajectories={trajectory_n}_lr={lr}_batch_size={batch_size}_{epsilon_decrease}')
                        run(env, episode_n, trajectory_n, lr, batch_size, epsilon_decrease)
                        wandb.finish()
