import collections
import random
import numpy as np
import torch


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer = collections.deque(maxlen=self.buffer_size)

    def store(self, observation, action, reward, next_observation, done):
        self.buffer.append((observation, action, reward, next_observation, done))

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        observations, actions, rewards, next_observations, dones = zip(*transitions)
        observations = torch.tensor(np.array(observations), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).view(-1, 1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.float).view(-1, 1)
        return observations, actions, rewards, next_observations, dones

    def size(self):
        return len(self.buffer)


