# Reinforcement Learning Algorithms

Collection of general RL algorithms as a training exercise. Work in progress.<br>
Reinforcement learning is a type of machine learning that is concerned with teaching agents how to make decisions in an environment. The agent learns to achieve a goal in an uncertain, potentially complex environment. <br>

Do not run any of these on your CPU.

###  Deep Deterministic Policy Gradient (DDPG)
https://arxiv.org/abs/1509.02971

DDPG is an off-policy, model-free reinforcement learning algorithm that combines deterministic policy gradient methods with experience replays and actor-critic networks. 
It is used for continuous action spaces.

Used in this project to solve [Lunar Lander problem from Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (similar to the deprecated OpenAI Gym) which is considered solved at +200 score. This model was able to achieve a 100 episode running average of +200 at 891 episodes.

Requires Python3, PyTorch, NumPy, Matplotlib and Gymnasium. Ensure compatibility between versions.

### Soft Actor-Critic (SAC)
https://arxiv.org/abs/1801.01290
>TODO

### Proximal Policy Optimization (PPO)
https://arxiv.org/abs/1707.06347
>TODO

### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
https://arxiv.org/abs/1706.02275
>TODO

### QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
https://arxiv.org/abs/1803.11485
>TODO