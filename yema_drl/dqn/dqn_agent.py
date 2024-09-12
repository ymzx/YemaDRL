import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F


# Trick: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Qnet(nn.Module):
    def __init__(self, args):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, state):
        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)


class DQN(object):
    def __init__(self, args):
        self.tau = args.tau
        self.buffer_size = args.buffer_size
        self.use_grad_clip = args.use_grad_clip
        self.target_update_freq = args.target_update_freq
        self.set_adam_eps = args.set_adam_eps
        self.use_soft_update = args.use_soft_update
        self.use_double = args.use_double
        self.max_train_steps = args.max_train_steps
        self.use_lr_decay = args.use_lr_decay
        self.lr = args.lr
        self.action_dim = args.action_dim
        self.gamma = args.gamma  # 折扣因子
        self.epsilon_max = args.epsilon_max
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.q_net = Qnet(args)
        self.target_q_net = Qnet(args)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.update_count = 0
        self.epsilon = self.epsilon_max

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float)
            action = self.q_net(state).argmax().item()
        return action

    def predict(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float)
        action = self.q_net(state).argmax().item()
        return action

    def learn(self, replay_buffer, total_steps):
        # get train data
        observations, actions, rewards, next_observations, dones = replay_buffer.sample()
        q_values = self.q_net(observations).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.use_double:
            max_action = self.q_net(next_observations).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_observations).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_observations).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        if self.use_grad_clip: torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        if self.use_soft_update:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            # hard update of the target network's weights
            # θ′ ← θ
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.epsilon_decay()

    def save(self, save_dir, env_name, number):
        save_qnet_path = os.path.join(save_dir, "{}_qnet_{}.pth".format(env_name, number))
        torch.save(self.q_net.state_dict(), save_qnet_path)

    def load(self, save_dir, env_name, number):
        save_qnet_path = os.path.join(save_dir, "{}_qnet_{}.pth".format(env_name, number))
        self.q_net.load_state_dict(torch.load(save_qnet_path))

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps) + 0.01 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def epsilon_decay(self):
        epsilon_delta = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_steps
        self.epsilon -= epsilon_delta
        self.epsilon = max(self.epsilon_min, self.epsilon)

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





