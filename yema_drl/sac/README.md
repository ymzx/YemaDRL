
#### 简介
Stochastic policy随机策略就是在每一个state上都能输出每一种action的概率，比如有3个action都是最优的，概率一样都最大，那么我们就可以从这些action中随机选择一个做出action输出。最大熵maximum entropy的核心思想就是不遗落到任意一个有用的action，有用的trajectory。对比DDPG的deterministic policy的做法，看到一个好的就捡起来，差一点的就不要了，而最大熵是都要捡起来，都要考虑。

#### paper
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905)
- [Reinforcement Learning with Deep Energy-Based Policies (Soft Q-Learning)](https://arxiv.org/pdf/1702.08165)

