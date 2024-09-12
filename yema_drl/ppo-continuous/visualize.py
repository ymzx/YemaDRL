import gymnasium as gym
from ppo_agent import ActorGaussian
from types import SimpleNamespace
import torch
import os

# 创建保存视频的目录
video_folder = r'videos'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

args = {'hidden_width': 64, 'use_tanh': True, 'state_dim': None, 'action_dim': None, 'use_orthogonal_init': None, 'max_action': None}
args = SimpleNamespace(**args)

save_actor_path = r'model\BipedalWalker-v3_actor_100.pth'
env_name = save_actor_path.split('\\')[-1].split('_')[0]

# 使用 RecordVideo 包装器保存动画
env = gym.make(env_name, render_mode="rgb_array")
args.max_episode_steps = env.spec.max_episode_steps
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=env_name)
args.state_dim = env.observation_space.shape[0]
args.action_dim = env.action_space.shape[0]
args.max_action = float(env.action_space.high[0])

actor = ActorGaussian(args=args)
actor.load_state_dict(torch.load(save_actor_path))

state, info = env.reset()
total_reward = 0
for step in range(args.max_episode_steps):
    env.render()
    state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
    action = actor(state).detach().numpy().flatten()
    obs, reward, terminated, truncated, info = env.step(action)
    state = obs
    total_reward += reward
    # 如果终止或截断，退出循环，避免重复录制
    if terminated or truncated:
        break
        # state, info = env.reset()
env.close()
print('总得分:', total_reward)
