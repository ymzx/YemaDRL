import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import numpy as np
import os


# Trick: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        prob = torch.softmax(self.fc3(s), dim=1)
        return prob

    def get_distribution(self, s):
        prob = self.forward(s)
        prob = Categorical(probs=prob)
        return prob


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO(object):
    def __init__(self, args):
        self.policy_distribution = args.policy_distribution
        self.max_action = args.max_action
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.gae_lambda = args.gae_lambda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.set_adam_eps = args.set_adam_eps
        self.mini_batch_size = args.mini_batch_size
        self.buffer_size = args.buffer_size
        self.entropy_coef = args.entropy_coef
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.max_train_steps = args.max_train_steps
        self.actor = Actor(args)
        self.critic = Critic(args)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(state))
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        action = action.numpy()[0]
        action_logprob = action_logprob.numpy()[0]
        return action, action_logprob

    def learn(self, replay_buffer, total_steps):
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        # get train data
        states, actions, action_logprobs, rewards, next_states, dead_or_win, done = replay_buffer.numpy_to_tensor()
        # calculate TD_target and GAE
        advantage, gae = [], 0
        with torch.no_grad():
            states_value = self.critic(states)
            nex_states_value = self.critic(next_states)
            deltas = rewards + self.gamma * nex_states_value * (1.0 - dead_or_win) - states_value
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - d)
                advantage.insert(0, gae)
            advantage = torch.tensor(advantage, dtype=torch.float).view(-1, 1)
            target_value = advantage + states_value
            advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition
            for idx_list in BatchSampler(sampler=SubsetRandomSampler(range(self.buffer_size)), batch_size=self.mini_batch_size, drop_last=False):
                distribution_now = self.actor.get_distribution(states[idx_list])
                distribution_entropy = distribution_now.entropy().view(-1, 1)
                action_logprobs_now = distribution_now.log_prob(actions[idx_list].squeeze()).view(-1, 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(action_logprobs_now - action_logprobs[idx_list])
                surrogate1 = ratios * advantage[idx_list]
                surrogate2 = torch.clamp(input=ratios, min=1-self.epsilon, max=1+self.epsilon) * advantage[idx_list]
                actor_loss = - torch.min(surrogate1, surrogate2) - self.entropy_coef * distribution_entropy
                # update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                # update critic
                critic_loss = F.mse_loss(target_value[idx_list], self.critic(states[idx_list]))
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def predict(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        prob = self.actor(state).detach().numpy().flatten()
        action = np.argmax(prob)
        return action

    def save(self, save_dir, env_name, number):
        save_actor_path = os.path.join(save_dir, "{}_actor_{}.pth".format(env_name, number))
        save_critic_path = os.path.join(save_dir, "{}_critic_{}.pth".format(env_name, number))
        torch.save(self.actor.state_dict(), save_actor_path)
        torch.save(self.critic.state_dict(), save_critic_path)

    def load(self, save_dir, env_name, number):
        save_actor_path = os.path.join(save_dir, "{}_actor_{}.pth".format(env_name, number))
        save_critic_path = os.path.join(save_dir, "{}_critic_{}.pth".format(env_name, number))
        self.actor.load_state_dict(torch.load(save_actor_path))
        self.critic.load_state_dict(torch.load(save_critic_path))

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




