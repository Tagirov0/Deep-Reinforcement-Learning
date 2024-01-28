import gym
import wandb
import numpy as np


class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        proba = self.model[state] / np.sum(self.model[state])
        action = np.random.choice(np.arange(self.action_n), p=proba)
        return int(action)

    def fit(self, elite_trajectories, smoothing_type=None, lambda_=0):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                if smoothing_type == 'laplace':
                    new_model[state] = new_model[state] + lambda_ / (np.sum(new_model[state]) + lambda_ * self.action_n)
                else:
                    new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        if smoothing_type == 'police':
            new_model = lambda_ * new_model + (1 - lambda_) * self.model

        self.model = new_model


def get_trajectory(env, agent, max_len=1000, visualize=False):
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


def train(smoothing_type, env, state_n, action_n, iterations, params):
    for q_param in params['q_params']:
        for trajectory_n in params['trajectories_n']:
            for lambda_ in params['lambdas']:
                wandb.init(
                    project=f"RL_taxi_{smoothing_type}_smoothing",
                    config={
                        'q_param': q_param,
                        'trajectories': trajectory_n,
                        'lambda': lambda_
                    },
                    name=f'q={q_param}_trajectories={trajectory_n}_lambda={lambda_}'
                )
                agent = CrossEntropyAgent(state_n, action_n)

                for iteration in range(iterations):
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

                    agent.fit(elite_trajectories, smoothing_type, lambda_)

                wandb.finish()


if __name__ == "__main__":
    state_n = 500
    action_n = 6
    env = gym.make('Taxi-v3')

    iterations = 50
    params = {
        'lambdas': [0.3, 0.5, 0.7, 0.9][::-1],
        'q_params': [0.6, 0.7, 0.8],
        'trajectories_n': [1000, 5000]
    }

    wandb.login(key='8224f98472fd102b8a3c3806a80663939495aade')
    train('laplace', env, state_n, action_n, iterations, params)
    # train('police', env, state_n, action_n, iterations, params)
