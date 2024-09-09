import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        action = self.max_action * torch.tanh(self.fc3(s))  # [-max,max]
        return action


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class DDPG(object):
    '''
    Deep Deterministic Policy Gradient (DDPG).

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    '''
    def __init__(self, args):
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Softly update the target network
        self.lr_a = args.lr_a  # learning rate of actor
        self.lr_c = args.lr_c  # learning rate of critic
        self.use_lr_decay = args.use_lr_decay
        self.max_train_steps = args.max_train_steps

        self.actor = Actor(args)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(args)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.mse_loss = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        with torch.no_grad():
            action = self.actor(state).data.numpy().flatten()
        return action

    def learn(self, replay_buffer, total_steps):

        def soft_update(net_target, net):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        states, actions, rewards, next_states, dead_or_win, done = replay_buffer.sample()
        # compute the target Q
        with torch.no_grad():
            q_ = self.critic_target(next_states, self.actor_target(next_states))
            target_q = rewards + self.gamma * (1 - dead_or_win) * q_

        # Compute the current q and the critic loss
        current_q = self.critic(states, actions)
        critic_loss = self.mse_loss(target_q, current_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = - self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        soft_update(net_target=self.critic_target, net=self.critic)
        soft_update(net_target=self.actor_target, net=self.actor)
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

    def predict(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        with torch.no_grad():
            action = self.actor(state).detach().numpy().flatten()
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





















