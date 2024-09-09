import numpy as np
import torch
from collections import deque


class ReplayBuffer(object):
    def __init__(self, args):
        self.mini_batch_size = args.mini_batch_size
        self.buffer_size = args.buffer_size
        self.actions = deque(maxlen=self.buffer_size)
        self.observations = deque(maxlen=self.buffer_size)
        self.rewards = deque(maxlen=self.buffer_size)
        self.next_observations = deque(maxlen=self.buffer_size)
        self.dead_or_win = deque(maxlen=self.buffer_size)
        self.done = deque(maxlen=self.buffer_size)
        self.count = 0

    def store(self, observation, action, reward, next_observation, dead_or_win, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dead_or_win.append(dead_or_win)
        self.done.append(done)
        self.count = len(self.observations)

    def sample(self):
        index = np.random.choice(self.count, size=self.mini_batch_size)
        observations = torch.tensor(np.array(self.observations)[index], dtype=torch.float)
        actions = torch.tensor(np.array(self.actions)[index], dtype=torch.float)
        rewards = torch.tensor(np.array(self.rewards)[index], dtype=torch.float).view(self.mini_batch_size, -1)
        next_observations = torch.tensor(np.array(self.next_observations)[index], dtype=torch.float)
        dead_or_win = torch.tensor(np.array(self.dead_or_win)[index], dtype=torch.float).view(self.mini_batch_size, -1)
        done = torch.tensor(np.array(self.done)[index], dtype=torch.float).view(self.mini_batch_size, -1)
        return observations, actions, rewards, next_observations, dead_or_win, done

