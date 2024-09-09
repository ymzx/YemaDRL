import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.n = args.n
        self.observation_dim = args.observation_dim
        self.state_dim = args.state_dim
        self.max_episode_steps = args.max_episode_steps
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'observations': np.empty([self.batch_size, self.max_episode_steps, self.n, self.observation_dim]),
                       'state': np.empty([self.batch_size, self.max_episode_steps, self.state_dim]),
                       'values': np.empty([self.batch_size, self.max_episode_steps + 1, self.n]),
                       'actions': np.empty([self.batch_size, self.max_episode_steps, self.n]),
                       'actions_logprob': np.empty([self.batch_size, self.max_episode_steps, self.n]),
                       'rewards': np.empty([self.batch_size, self.max_episode_steps, self.n])}
        self.episode_num = 0

    def store(self, episode_step=None, observations=None, actions=None, actions_logprob=None, rewards=None, dones=None,
              values=None, agent_names=None):
        observations = np.array([observations[agent_name] for agent_name in agent_names])
        # In MPE, global state is the concatenation of all agents' local obs.
        state = observations.flatten()
        rewards = np.array([rewards[agent_name] for agent_name in agent_names])
        actions = np.array([actions[agent_name] for agent_name in agent_names])
        values = np.array([values[agent_name] for agent_name in agent_names])
        actions_logprob = np.array([actions_logprob[agent_name] for agent_name in agent_names])
        self.buffer['observations'][self.episode_num][episode_step] = observations
        self.buffer['actions'][self.episode_num][episode_step] = actions
        self.buffer['actions_logprob'][self.episode_num][episode_step] = actions_logprob
        self.buffer['rewards'][self.episode_num][episode_step] = rewards
        self.buffer['values'][self.episode_num][episode_step] = values
        self.buffer['state'][self.episode_num][episode_step] = state

    def store_last_value(self, episode_step, last_value):
        last_value = np.array([value for agent_name, value in last_value.items()])
        self.buffer['values'][self.episode_num][episode_step] = last_value
        self.episode_num += 1

    def sample(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'actions':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
