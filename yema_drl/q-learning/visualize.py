import time
import numpy as np
import gymnasium as gym


def render_policy(env, max_steps=100):
    state, info = env.reset()
    done = False
    step_count = 0
    env.render()  # 初始化渲染
    time.sleep(1)  # 短暂停留以便于观察

    while not done and step_count < max_steps:
        action = Q[state].argmax()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = (terminated or truncated)
        state = next_state
        env.render()  # 渲染当前环境状态
        # time.sleep(0.5)  # 暂停以观察动作效果
        step_count += 1


if __name__ == '__main__':
    # 加载保存的 Q 表
    Q = np.load('model/CliffWalking-v0_Q_table_100.npy')
    # 创建环境
    env = gym.make('CliffWalking-v0', render_mode="human")
    # 渲染策略
    render_policy(env)
    # 结束环境渲染
    env.close()

