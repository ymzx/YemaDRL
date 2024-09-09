from torch.utils.tensorboard import SummaryWriter
import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch
import argparse
from qlearning_agent import Qlearning

current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "model")
os.makedirs(save_dir, exist_ok=True)


def main(args, env_name, seed):
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=1000)
    env_evaluate = gym.make(env_name)
    env_evaluate = TimeLimit(env_evaluate, max_episode_steps=1000)

    # Set random seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.n
    args.action_dim = env.action_space.n
    args.max_action = None
    args.max_episode_steps = env.spec.max_episode_steps  # Maximum number of steps per episode

    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_count = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    agent = Qlearning(args)

    # Build a tensorboard
    log_dir = 'logs/env_{}_seed_{}'.format(env_name, seed)
    writer = SummaryWriter(log_dir=log_dir)

    while total_steps < args.max_train_steps:
        state, info = env.reset()
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            agent.learn(observation=state, action=action, reward=reward, next_observation=next_state, done=done)
            state = next_state
            total_steps += 1

            if total_steps % args.evaluate_steps == 0:
                evaluate_count += 1
                evaluate_reward = agent.evaluate_policy(env=env_evaluate, turns=3)
                evaluate_rewards.append(evaluate_reward)
                print(f'Env:{env_name}, Steps: {int(total_steps / 1000)}k, Evaluate Count: {evaluate_count}, '
                      f'Episode Reward:{evaluate_reward}')
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)

            if total_steps % args.save_steps == 0:
                agent.save(save_dir=save_dir, env_name=env_name, number=int(total_steps / args.save_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for Q-learning")
    parser.add_argument("--max_train_steps", type=int, default=int(1e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_steps", type=int, default=int(1e2), help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_steps", type=int, default=int(1e3), help="Save frequency")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.3, help="e-greedy")
    args = parser.parse_args()
    env_name = ['CliffWalking-v0', 'FrozenLake-v1', 'Taxi-v3']
    main(args, env_name=env_name[0], seed=1664)
