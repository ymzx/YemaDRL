import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import os


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class ActorMLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(ActorMLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class CriticMLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(CriticMLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        x1 = self.activate_func(self.fc1(critic_input))
        x2 = self.activate_func(self.fc2(x1))
        value = self.fc3(x2)
        return value


class MAPPO(object):
    def __init__(self, args):
        self.n = args.n
        self.action_dim = args.action_dim
        self.observation_dim = args.observation_dim
        self.state_dim = args.state_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.max_episode_steps = args.max_episode_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.actor_input_dim = args.observation_dim
        self.critic_input_dim = args.state_dim

        self.actor = ActorMLP(args, self.actor_input_dim)
        self.critic = CriticMLP(args, self.critic_input_dim)
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, observations):
        agent_names = [agent_name for agent_name, observation in observations.items()]
        observations = np.array([observation for agent_name, observation in observations.items()])  # (n, observation_dim)
        with torch.no_grad():
            observations = torch.tensor(observations, dtype=torch.float32)
            prob = self.actor(observations)
            dist = Categorical(probs=prob)
            actions = dist.sample()
            actions_logprob = dist.log_prob(actions)
        actions_dict = {agent_name: int(actions[i].numpy()) for i, agent_name in enumerate(agent_names)}
        actions_logprob_dict = {agent_name: actions_logprob[i].numpy() for i, agent_name in enumerate(agent_names)}
        return actions_dict, actions_logprob_dict

    def get_critic_value(self, observations):
        agent_names = [agent_name for agent_name, observation in observations.items()]
        observations = np.array([observation for agent_name, observation in observations.items()])  # (n, observation_dim)
        state = observations.flatten()
        with torch.no_grad():
            # Because each agent has the same global state, we need to repeat the global state 'n' times.
            state = torch.tensor(state, dtype=torch.float32).repeat(self.n, 1)  # (n, state_dim)
            values = self.critic(state)  # values.shape(n,1)
            values = values.numpy().flatten()
        values_dict = {agent_name: values[i] for i, agent_name in enumerate(agent_names)}
        return values_dict

    def learn(self, replay_buffer, total_steps):
        batch = replay_buffer.sample()
        # Calculate the advantage using GAE
        advantage, gae = [], 0
        with torch.no_grad():
            deltas = batch['rewards'] + self.gamma * batch['values'][:, 1:] - batch['values'][:, :-1]
            for t in reversed(range(self.max_episode_steps)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                advantage.insert(0, gae)
            advantage = torch.stack(advantage, dim=1)  # adv.shape(batch_size, max_episode_steps, n)
            value_target = advantage + batch['values'][:, :-1]  # v_target.shape(batch_size,max_episode_steps,n)
            advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))
        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = batch['observations'], batch['state'].unsqueeze(2).repeat(1, 1, self.n, 1)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                probs_now = self.actor(actor_inputs[index])
                values_now = self.critic(critic_inputs[index]).squeeze(-1)
                distribution_now = Categorical(probs_now)
                distribution_entropy = distribution_now.entropy()
                actions_logprob_now = distribution_now.log_prob(batch['actions'][index])
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(actions_logprob_now - batch['actions_logprob'][index].detach())
                surr1 = ratios * advantage[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantage[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * distribution_entropy
                critic_loss = (values_now - value_target[index]) ** 2
                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def predict(self, observations):
        agent_names = [agent_name for agent_name, observation in observations.items()]
        observations = np.array([observation for agent_name, observation in observations.items()])  # (n, observation_dim)
        with torch.no_grad():
            observations = torch.tensor(observations, dtype=torch.float32)
            prob = self.actor(observations)
            actions = prob.argmax(dim=-1)
        actions_dict = {agent_name: int(actions[i].numpy()) for i, agent_name in enumerate(agent_names)}
        return actions_dict

    def save(self, save_dir, env_name, number):
        save_actor_path = os.path.join(save_dir, "{}_actor_{}.pth".format(env_name, number))
        save_critic_path = os.path.join(save_dir, "{}_critic_{}.pth".format(env_name, number))
        torch.save(self.actor.state_dict(), save_actor_path)
        torch.save(self.critic.state_dict(), save_critic_path)

    def evaluate_policy(self, env, turns):
        total_rewards = 0
        for j in range(turns):
            observations, infos = env.reset()
            done = False
            episode_reward = 0
            while not done:
                actions = self.predict(observations=observations)
                next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
                observations = next_observations
                dones = np.array([truncateds[agent] for agent in truncateds.keys()])
                done = all(dones)
                if done: break
                episode_reward += np.mean(np.array(list(rewards.values())))
            total_rewards += episode_reward
        return total_rewards / turns














