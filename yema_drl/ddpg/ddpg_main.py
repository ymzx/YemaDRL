import argparse
import gymnasium as gym
import torch
import numpy as np
from ddpg_agent import DDPG
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "model")
os.makedirs(save_dir, exist_ok=True)


def reward_adapter(r, env_name):
    '''
    调整reward，利于收敛
    Args:
        r: 奖励值
        env_name: 环境名

    Returns: 调整后的奖励值

    '''
    if env_name == 'Pendulum-v1':  # Pendulum-v1
        r = (r + 8) / 8
    elif env_name == 'BipedalWalker-v3':  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r


def main(args, env_name, seed):
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment

    # Set random seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.spec.max_episode_steps  # Maximum number of steps per episode
    args.noise_std = args.noise_std * args.max_action

    evaluate_count = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = DDPG(args)

    # Build a tensorboard
    log_dir = 'logs/env_{}_seed_{}'.format(env_name, seed)
    writer = SummaryWriter(log_dir=log_dir)

    while total_steps < args.max_train_steps:
        state, info = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            # Take the random actions in the beginning for the better exploration
            if total_steps < args.random_steps:
                action = env.action_space.sample()
            else:
                # Add Gaussian noise to actions for exploration
                action = agent.choose_action(state)
                noise = np.random.normal(0, args.noise_std, size=args.action_dim)
                action = (action + noise).clip(-args.max_action, args.max_action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            reward = reward_adapter(reward, env_name)
            if done and episode_steps != args.max_episode_steps:
                dead_or_win = True
            else:
                dead_or_win = False
            # store the current transition
            replay_buffer.store(observation=state, action=action, reward=reward, next_observation=next_state,
                                dead_or_win=dead_or_win, done=done)
            state = next_state
            total_steps += 1

            # Take 50 steps,then update the networks 50 times
            if total_steps >= args.random_steps and total_steps % args.learn_steps == 0:
                for _ in range(args.learn_steps):
                    agent.learn(replay_buffer, total_steps)

            # evaluate the policy every evaluate_steps
            if total_steps >= args.random_steps and total_steps % args.evaluate_steps == 0:
                evaluate_count += 1
                evaluate_reward = agent.evaluate_policy(env=env_evaluate, turns=3)
                evaluate_rewards.append(evaluate_reward)
                print(f'Env:{env_name}, Steps: {int(total_steps / 1000)}k, Evaluate Count: {evaluate_count}, '
                      f'Episode Reward:{evaluate_reward}')
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)

            # save model
            if total_steps % args.save_steps == 0:
                agent.save(save_dir=save_dir, env_name=env_name, number=int(total_steps / args.save_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--random_steps", type=int, default=int(5e4), help=" Maximum number of training steps")
    parser.add_argument("--learn_steps", type=int, default=1, help="Learn the policy every learn_steps")
    parser.add_argument("--evaluate_steps", type=int, default=int(1e3), help="Evaluate the policy every evaluate_steps")
    parser.add_argument("--save_steps", type=int, default=int(1e4), help="Save frequency")
    parser.add_argument("--noise_std", type=float, default=0.1, help=" The std of Gaussian noise for exploration")
    parser.add_argument("--buffer_size", type=int, default=int(5e5), help="Buffer size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    args = parser.parse_args()
    env_name = ['BipedalWalker-v3', 'Pendulum-v1']
    main(args, env_name=env_name[1], seed=200)
