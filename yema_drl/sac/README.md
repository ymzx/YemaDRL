
#### 简介
Soft Actor-Critic (SAC)是面向Maximum Entropy Reinforcement learning 开发的一种off policy算法，和DDPG相比，Soft Actor-Critic使用的是随机策略stochastic policy，Stochastic policy随机策略就是在每一个state上都能输出每一种action的概率，比如有3个action都是最优的，概率一样都最大，那么我们就可以从这些action中随机选择一个做出action输出。最大熵maximum entropy的核心思想就是不遗落到任意一个有用的action，有用的trajectory。对比DDPG的deterministic policy的做法，看到一个好的就捡起来，差一点的就不要了，而最大熵是都要捡起来，都要考虑。

#### paper
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905)
- [Reinforcement Learning with Deep Energy-Based Policies (Soft Q-Learning)](https://arxiv.org/pdf/1702.08165)

#### 基于最大熵的RL算法优势
以前用deterministic policy的算法，我们找到了一条最优路径，学习过程也就结束了。现在，我们还要求熵最大，就意味着神经网络需要去explore探索所有可能的最优路径.

- 学到policy可以作为更复杂具体任务的初始化。因为通过最大熵，policy不仅仅学到一种解决任务的方法，而是所有all。因此这样的policy就更有利于去学习新的任务。比如我们一开始是学走，然后之后要学朝某一个特定方向走。
- 更强的exploration能力，这是显而易见的，能够更容易的在多模态reward （multimodal reward）下找到更好的模式。比如既要求机器人走的好，又要求机器人节约能源
- 更robust鲁棒，更强的generalization。因为要从不同的方式来探索各种最优的可能性，也因此面对干扰的时候能够更容易做出调整。（干扰会是神经网络学习过程中看到的一种state，既然已经探索到了，学到了就可以更好的做出反应，继续获取高reward）


#### 参考
- https://zhuanlan.zhihu.com/p/85003758

