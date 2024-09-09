import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.actions = np.zeros((args.buffer_size, args.action_dim))
        self.log_probs = np.zeros((args.buffer_size, args.action_dim))
        self.observations = np.zeros((args.buffer_size, args.state_dim))
        self.rewards = np.zeros((args.buffer_size, 1))
        self.next_observations = np.zeros((args.buffer_size, args.state_dim))
        self.dead_or_win = np.zeros((args.buffer_size, 1))
        self.done = np.zeros((args.buffer_size, 1))
        self.count = 0

    def store(self, observation, action, log_prob, reward, next_observation, dead_or_win, done):
        self.observations[self.count] = observation
        self.actions[self.count] = action
        self.log_probs[self.count] = log_prob
        self.rewards[self.count] = reward
        self.next_observations[self.count] = next_observation
        self.dead_or_win[self.count] = dead_or_win
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        observations = torch.tensor(self.observations, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float)
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        next_observations = torch.tensor(self.next_observations, dtype=torch.float)
        dead_or_win = torch.tensor(self.dead_or_win, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        return observations, actions, log_probs, rewards, next_observations, dead_or_win, done
