import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from ippo_agent import IPPO
from pettingzoo.mpe import simple_spread_v4
import os
from yema_drl.common.normalization import Normalization

current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "model")
os.makedirs(save_dir, exist_ok=True)


def make_env(max_cycles, render_mode="None"):
    '''
    当 local_ratio=0 时，代理只根据 全局奖励 进行训练，即整个团队的表现决定了所有代理的奖励。
    当 local_ratio=1 时，代理只根据 局部奖励 进行训练，即每个代理的表现只影响自己，完全忽略全局奖励。
    当 local_ratio=0.5 时，代理的奖励是 全局奖励 和 局部奖励 的混合，每种奖励各占 50%。
    '''
    env = simple_spread_v4.parallel_env(N=3, max_cycles=max_cycles, local_ratio=1.0,
                                        render_mode=render_mode, continuous_actions=False)
    env.reset(seed=42)
    return env


class RunnerIPPO(object):
    def __init__(self, args, env_name, seed):
        self.args = args
        self.env_name = env_name
        self.seed = seed

        # set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create env
        self.env = make_env(max_cycles=args.max_episode_steps)
        self.env_evaluate = make_env(max_cycles=args.max_episode_steps)

        # The number of agents
        self.args.n = self.env.max_num_agents

        # 获取N个智能体的观测空间和动作空间
        self.args.observations_dim = [self.env.observation_space(agent).shape[0] for agent in self.env.agents]
        self.args.actions_dim = [self.env.action_space(agent).n for agent in self.env.agents]
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        # The dimensions of an agent's observation space
        self.args.observation_dim = self.args.observations_dim[0]
        # The dimensions of an agent's action space
        self.args.action_dim = self.args.actions_dim[0]

        print("observation_space=", self.env.observation_space(self.env.agents[0]))
        print("observations_dim={}".format(self.args.observations_dim))
        print("action_space=", self.env.action_space(self.env.agents[0]))
        print("actions_dim={}".format(self.args.actions_dim))

        # Create N agents
        self.agents = IPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Build a tensorboard
        log_dir = 'logs/env_{}_seed_{}'.format(env_name, seed)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_reward_norm:
            self.reward_norm = Normalization(shape=self.args.n)

    def _reward_norm(self, rewards):
        agent_names = [agent_name for agent_name, reward in rewards.items()]
        rewards = np.array([rewards[agent_name] for agent_name in agent_names])
        rewards = self.reward_norm(rewards)
        rewards_dict = {agent_name: rewards[i] for i, agent_name in enumerate(agent_names)}
        return rewards_dict

    def run(self):
        evaluate_count = 0  # Record the number of evaluations
        agent_names = [agent_name for agent_name in self.env.agents]
        while self.total_steps < self.args.max_train_steps:
            observations, infos = self.env.reset()
            for episode_step in range(self.args.max_episode_steps):
                actions, actions_logprob = self.agents.choose_action(observations=observations)
                values = self.agents.get_critic_value(observations=observations)
                next_observations, rewards, terminateds, truncateds, infos = self.env.step(actions)
                if self.args.use_reward_norm: rewards = self._reward_norm(rewards)
                # store the current transition
                self.replay_buffer.store(episode_step=episode_step, observations=observations, actions=actions,
                                         actions_logprob=actions_logprob, rewards=rewards, values=values,
                                         dones=terminateds, agent_names=agent_names)
                observations = next_observations
                self.total_steps += 1

            # an episode is over, store value in the last step
            last_value = self.agents.get_critic_value(observations=observations)
            self.replay_buffer.store_last_value(self.args.max_episode_steps, last_value)

            # buffer reaches buffer_size,then update
            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agents.learn(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

            # evaluate the policy every evaluate_steps
            if self.total_steps % args.evaluate_steps == 0:
                evaluate_count += 1
                evaluate_reward = self.agents.evaluate_policy(env=self.env_evaluate, turns=3)
                self.evaluate_rewards.append(evaluate_reward)
                print(f'Env:{self.env_name}, Steps: {int(self.total_steps / 1000)}k, Evaluate Count: {evaluate_count}, '
                      f'Episode Reward:{evaluate_reward}')
                self.writer.add_scalar('step_rewards_{}'.format(self.env_name), self.evaluate_rewards[-1],
                                       global_step=self.total_steps)

            # save model
            if self.total_steps % self.args.save_steps == 0:
                self.agents.save(save_dir=save_dir, env_name=self.env_name, number=int(self.total_steps / args.save_steps))

        self.env.close()
        self.env_evaluate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for IPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_steps", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--max_episode_steps", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--save_steps", type=int, default=3e4, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=True, help="Whether to use tanh, if False, we will use relu")

    args = parser.parse_args()
    env_names = ['simple_spread_v4']
    runner = RunnerIPPO(args, env_name=env_names[0], seed=0)
    runner.run()

