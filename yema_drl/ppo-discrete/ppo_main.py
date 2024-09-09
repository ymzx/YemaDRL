import argparse
import gymnasium as gym
import torch
import numpy as np
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from ppo_agent import PPO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "model")
os.makedirs(save_dir, exist_ok=True)


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

    replay_buffer = ReplayBuffer(args)
    agent = PPO(args)

    # Build a tensorboard
    log_dir = 'logs/env_{}_seed_{}'.format(env_name, seed)
    writer = SummaryWriter(log_dir=log_dir)

    while total_steps < args.max_train_steps:
        state, info = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            action, action_logprob = agent.choose_action(state=state)
            if args.policy_distribution == 'Beta':
                action = 2 * (action - 0.5) * args.max_action  # [0,1]->[-max,max]
            next_state, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            if done and episode_steps != args.max_episode_steps:
                dead_or_win = True
            else:
                dead_or_win = False
            # store the current transition
            replay_buffer.store(observation=state, action=action, log_prob=action_logprob, reward=reward,
                                next_observation=next_state, dead_or_win=dead_or_win, done=done)
            state = next_state
            total_steps += 1

            # buffer reaches buffer_size,then update
            if replay_buffer.count == args.buffer_size:
                agent.learn(replay_buffer, total_steps)
                replay_buffer.count = 0

            # evaluate the policy every evaluate_steps
            if total_steps % args.evaluate_steps == 0:
                evaluate_count += 1
                evaluate_reward = agent.evaluate_policy(env=env_evaluate, turns=3)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_count:{} \t evaluate_reward:{} \t".format(evaluate_count, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)

            # save model
            if total_steps % args.save_steps == 0:
                agent.save(save_dir=save_dir, env_name=env_name, number=int(total_steps / args.save_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_steps", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_steps", type=int, default=3e4, help="Save frequency")
    parser.add_argument("--policy_distribution", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--buffer_size", type=int, default=2048, help="Buffer size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    env_name = ['CartPole-v1']  # 离散
    main(args, env_name=env_name[0], seed=10)



