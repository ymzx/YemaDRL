import gymnasium as gym
from dqn_agent import Qnet
from types import SimpleNamespace
import torch
import os
import numpy as np
import imageio

# 初始化
video_folder = r'videos'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)
args = {'hidden_dim': 64, 'use_tanh': True, 'state_dim': None, 'action_dim': None, 'use_orthogonal_init': None, 'max_action': None}
args = SimpleNamespace(**args)
save_qnet_path = r'model\CartPole-v1_qnet_97.pth'
env_name = save_qnet_path.split('\\')[-1].split('_')[0]
gif_path = os.path.join(video_folder, env_name + ".gif")
render_mode = "rgb_array"

# 使用 RecordVideo 包装器保存动画
env = gym.make(env_name, render_mode=render_mode)
args.max_episode_steps = env.spec.max_episode_steps

args.state_dim = env.observation_space.shape[0]
if isinstance(env.action_space, gym.spaces.Discrete):
    args.action_dim = env.action_space.n
else:
    args.action_dim = env.action_space.shape[0]
qnet = Qnet(args=args)
qnet.load_state_dict(torch.load(save_qnet_path))

state, info = env.reset()
frames = []  # 用于保存渲染的帧图像
for step in range(args.max_episode_steps):
    # 渲染环境
    if render_mode == "rgb_array":
        frame = env.render()  # 获取渲染的帧
        frames.append(frame)  # 保存帧用于GIF
    else:
        env.render()  # 在 human 模式下直接渲染
    state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
    prob = qnet(state).detach().numpy().flatten()
    action = np.argmax(prob)
    obs, reward, terminated, truncated, info = env.step(action)
    state = obs
    # 如果终止或截断，退出循环，避免重复录制
    if terminated or truncated:
        break
env.close()

# 如果是 rgb_array 模式，保存为GIF
if render_mode == "rgb_array" and frames:
    imageio.mimsave(gif_path, frames, fps=30)  # 生成GIF，30帧每秒
    print(f"GIF saved at {gif_path}")
