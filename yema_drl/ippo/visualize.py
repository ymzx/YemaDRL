import gymnasium as gym
from ippo_agent import ActorMLP
from types import SimpleNamespace
import torch
import os
from pettingzoo.mpe import simple_spread_v3
import numpy as np
from torch.distributions import Categorical
from PIL import Image


def choose_action(observations,actor):
    agent_names = [agent_name for agent_name, observation in observations.items()]
    observations = np.array([observation for agent_name, observation in observations.items()])  # (n, observation_dim)
    with torch.no_grad():
        actor_inputs = []
        observations = torch.tensor(observations, dtype=torch.float32)
        actor_inputs.append(observations)
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        prob = actor(actor_inputs)
        dist = Categorical(probs=prob)
        actions = dist.sample()
        actions_logprob = dist.log_prob(actions)
    actions_dict = {agent_name: actions[i].numpy() for i, agent_name in enumerate(agent_names)}
    actions_logprob_dict = {agent_name: actions_logprob[i].numpy() for i, agent_name in enumerate(agent_names)}
    return actions_dict, actions_logprob_dict

# 创建保存视频的目录
video_folder = r'videos'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

args = {'mlp_hidden_dim': 64, 'max_episode_steps': 25, 'use_tanh': True, 'action_dim': 5, 'use_orthogonal_init': None, 'max_action': None}
args = SimpleNamespace(**args)
save_actor_path = r'model\simple_spread_v3_actor_100.pth'

render_mode = ['rgb_array', "human"]
env = simple_spread_v3.parallel_env(N=3, max_cycles=args.max_episode_steps, local_ratio=0.5, render_mode=render_mode[0], continuous_actions=False)
env.reset()
env_name = save_actor_path.split('\\')[-1].split('_')[0]
args.observation_dim = [env.observation_space(agent).shape[0] for agent in env.agents][0]
args.action_dim = [env.action_space(agent).n for agent in env.agents][0]

# 使用 RecordVideo 包装器保存动画
# env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=env_name)
actor = ActorMLP(args=args)

actor.load_state_dict(torch.load(save_actor_path))

observations, infos = env.reset()
total_reward = 0
frame_list = []
for step in range(args.max_episode_steps):
    env.render()
    actions, actions_logprob = choose_action(observations=observations, actor=actor)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    frame_list.append(Image.fromarray(env.render()))
    observations = observations
    total_reward += sum(list(rewards.values()))
    # 如果终止或截断，退出循环，避免重复录制
    dones = all(list(terminations.values())) or all(list(truncations.values()))
    if dones:
        break
env.close()
print('总得分:', total_reward)
# save gif
gif_name = save_actor_path.split('\\')[-1]
frame_list[0].save(os.path.join(video_folder, f'{gif_name}.gif'),
                   save_all=True, append_images=frame_list[1:], duration=1, loop=0)
