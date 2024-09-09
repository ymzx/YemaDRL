import numpy as np
import os


class Qlearning(object):
    def __init__(self, args):
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.action_dim = args.action_dim
        self.Q = np.zeros((args.state_dim, args.action_dim))

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.Q[state].argmax()
        else:
            action = np.random.randint(self.action_dim)
        return action

    def learn(self, observation=None, action=None, reward=None, next_observation=None, done=None):
        '''Update Q table'''
        q_value = self.Q[observation, action]
        target_q_value = reward + (1 - done) * self.gamma * self.Q[next_observation].max()
        self.Q[observation, action] += self.lr * (target_q_value - q_value)

    def save(self, save_dir, env_name, number):
        '''save Q table'''
        save_q_table_path = os.path.join(save_dir, "{}_Q_table_{}.npy".format(env_name, number))
        np.save(save_q_table_path, self.Q)
        print(save_q_table_path + ' Q table saved.')

    def load(self, save_dir, env_name, number):
        '''load Q table'''
        save_q_table_path = os.path.join(save_dir, "{}_Q_table_{}.npy".format(env_name, number))
        self.Q = np.load(save_q_table_path)
        print(save_q_table_path + ' Q table loaded.')

    def predict(self, state):
        action = self.Q[state].argmax()
        return action

    def evaluate_policy(self, env, turns):
        total_rewards = 0
        for j in range(turns):
            state, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.predict(state=state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                episode_reward += reward
                done = (terminated or truncated)
            total_rewards += episode_reward
        return total_rewards / turns

