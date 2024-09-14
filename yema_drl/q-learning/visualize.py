import time
import numpy as np
import gymnasium as gym
import imageio
import os

# 创建保存视频的目录
video_folder = r'videos'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

gif_name = 'CliffWalking-v0'
gif_path = os.path.join(video_folder, f'{gif_name}.gif')


def render_policy(env, max_steps=100):
    state, info = env.reset()
    done = False
    step_count = 0
    frames = []  # List to store frames

    # Capture initial frame
    frames.append(env.render())

    while not done and step_count < max_steps:
        action = Q[state].argmax()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = (terminated or truncated)
        state = next_state

        frame = env.render()  # Capture frame at each step
        frames.append(frame)  # Add frame to the list

        step_count += 1

    # Save frames as GIF
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"GIF saved at {gif_path}")


if __name__ == '__main__':
    # Load the saved Q-table
    Q = np.load('model/CliffWalking-v0_Q_table_100.npy')
    # Create environment
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")  # Set render_mode to 'rgb_array'
    # Render policy and generate GIF
    render_policy(env, max_steps=100)
    # Close the environment
    env.close()
