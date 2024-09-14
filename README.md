<p align="center">
  <img src="https://github.com/ymzx/YemaDRL/raw/master/assets/DRL-LOGO.png" alt="DRL Logo" width="200"/>
</p>

<h3 align="center" style="margin-top: -10px;"><strong>YemaDRL: A Bridge to Deep Reinforcement Learning for Beginners</strong></h3>

---

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> Why was YemaDRL created?

<p align="justify">
  <strong>YemaDRL</strong> was developed to address the steep learning curve often faced by beginners in deep reinforcement learning. Many existing frameworks have complex codebases, unfriendly high-level APIs or inconsistent coding style, making them difficult for newcomers to grasp. YemaDRL simplifies this by offering a clean, modular design that facilitates easy experimentation and rapid prototyping. It serves as a foundation for understanding key RL concepts, allowing users to transition smoothly to more advanced frameworks, whether for single-agent learning frameworks like SpinningUp, Tianshou, Stable-Baselines3, or RLlib, or for multi-agent frameworks such as PettingZoo, PyMARL, MAlib, and MARLlib, making those tools more intuitive and effective for further exploration.
</p>


## 👫 Who is YemaDRL for? 
YemaDRL is designed for beginners in reinforcement learning, researchers, and developers who are interested in exploring or developing deep reinforcement learning (DRL) solutions. It serves as a bridge for those transitioning from basic machine learning concepts to more advanced RL algorithms.
- Learning Stage: For those starting with DRL concepts, offering pre-built algorithms and environments.
- Prototyping Stage: When researchers and developers need a flexible platform to quickly test new ideas or modify existing algorithms.
- Research & Deployment Stage: For researchers applying reinforcement learning to real-world problems such as autonomous systems, robotics, or gaming.


## 🚀 Update Log







## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> Supported Algorithms

| **Category**           | **Algorithm**                            | **Description**                                                      |
|------------------------|------------------------------------------|----------------------------------------------------------------------|
| **Single-Agent**        | [DQN (Deep Q-Network)](https://github.com/ymzx/YemaDRL/tree/master/yema_drl/dqn)                     |  [paper](https://arxiv.org/pdf/1710.02298.pdf)    |
|                        | [PPO (Proximal Policy Optimization)](https://github.com/ymzx/YemaDRL/tree/master/yema_drl/ppo-continuous)        |  [paper](https://arxiv.org/pdf/1707.06347.pdf)  |
|                        | [DDPG (Deep Deterministic Policy Gradient)](https://github.com/ymzx/YemaDRL/tree/master/yema_drl/ddpg) |  [paper](https://arxiv.org/pdf/1509.02971.pdf)                   |
|                        | [Q-Learning](https://github.com/ymzx/YemaDRL/tree/master/yema_drl/q-learning)                               |   [paper](https://link.springer.com/article/10.1007/BF00992698)     |
| **Multi-Agent**         | [MAPPO (Multi-Agent PPO)](https://github.com/ymzx/YemaDRL/tree/master/yema_drl/mappo)                  |   [paper](https://arxiv.org/pdf/1509.02971.pdf)      |
|                        | [IPPO (Independent PPO)](https://github.com/ymzx/YemaDRL/tree/master/yema_drl/ippo)                   |  [paper](https://arxiv.org/pdf/2103.01955v1) |


## 💡 Showcase

[CartPole-v1_dqn](https://github.com/ymzx/YemaDRL/blob/master/assets/gif/CartPole-v1_dqn.gif) | [Simple_spread_v3](https://github.com/ymzx/YemaDRL/blob/master/assets/gif/simple_spread_v3_mappo.gif) 


## <img src="https://user-images.githubusercontent.com/48054808/157799599-e6a66855-bac6-4e75-b9c0-96e13cb9612f.png" width="20"/> Roadmap and Future Features

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> Technical Discussion


## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/> Getting Started
## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> Release Notes

- _2024.09.14_  **Release v0.1**

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="20"> License

This project is licensed under the [Apache 2.0 license](LICENSE).

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Citation

If you use this project in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{YemaDRL,
  author = {ymzx},
  title = {YemaDRL: A Bridge to Deep Reinforcement Learning for Beginners},
  year = {2024},
  url = {https://github.com/ymzx/YemaDRL},
  note = {GitHub repository}
}