import time
import wandb
import numpy as np
from Frozen_Lake import FrozenLakeEnv

env = FrozenLakeEnv()


def get_q_values(v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * env.get_reward(state, action, next_state)
                q_values[state][action] += gamma * env.get_transition_prob(state, action, next_state) * v_values[next_state]
    return q_values


def init_v_values():
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values


def value_iteration_step(v_values, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = init_v_values()
    for state in env.get_all_states():
        new_v_values[state] += max(list(q_values[state].values()), default=0)
    return new_v_values


def value_iteration(gamma, value_iter_n):
    v_values = init_v_values()
    for _ in range(value_iter_n):
        v_values = value_iteration_step(v_values, gamma)
    q_values = get_q_values(v_values, gamma)

    return q_values


def get_policy(q_values):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        argmax_action = None
        max_q_value = float('-inf')
        for action in env.get_possible_actions(state):
            policy[state][action] = 0
            if q_values[state][action] > max_q_value:
                argmax_action = action
                max_q_value = q_values[state][action]
        policy[state][argmax_action] = 1
    return policy


wandb.login()

iters = [30, 50, 100, 300]
gammas = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

for iter_n in iters:
    for gamma in gammas:
        wandb.init(
            project="Value Iteration",
            config={
                'iter_n': iter_n,
                'gamma': gamma
            },
            name=f'iter_n={iter_n}-gamma={gamma}'
        )

        q_values = value_iteration(gamma, iter_n)
        policy = get_policy(q_values)

        total_rewards = []
        for _ in range(1000):
            total_reward = 0
            state = env.reset()
            for _ in range(1000):
                action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
                state, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    break

            total_rewards.append(total_reward)

        wandb.log({'reward': np.mean(total_rewards)})
        wandb.finish()
