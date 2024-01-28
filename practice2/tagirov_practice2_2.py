import gym
import wandb
import torch
import numpy as np
from torch import nn


class CEM(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.tanh = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        noise = torch.FloatTensor(state.shape[0]).uniform_(-0.01, 0.01)
        logits = self.forward(state) + noise
        return self.tanh(logits).data.numpy()

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])

        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.FloatTensor(np.array(elite_actions))

        pred_actions = self.forward(elite_states)
        loss = self.loss(pred_actions, elite_actions)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def get_trajectory(env, agent, max_len=500, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state = env.reset()

    for i in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        if visualize:
            env.render()

        if done:
            break

    return trajectory


def train(env, agent, iteration_n, trajectory_n, q_param):
    for iteration in range(iteration_n):
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))
        wandb.log({'reward': np.mean(total_rewards)})

        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)


if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')

    np.random.seed(42)
    torch.manual_seed(42)
    agent = CEM(state_dim=2)
    iteration_n = 30
    trajectory_n = 50
    q_param = 0.6

    wandb.login(key='8224f98472fd102b8a3c3806a80663939495aade')
    wandb.init(
        project=f"MountainCarContinuous",
        config={
            'q_param': q_param,
            'trajectories': trajectory_n,
            'learning_rate': 0.05
        },
        name=f'q={q_param}_trajectories={trajectory_n}_lr={0.05}'
    )
    train(env, agent, iteration_n, trajectory_n, q_param)
    trajectory = get_trajectory(env, agent, max_len=999, visualize=True)

