# General AI

Collection of general ML algorithms as a training exercise. Work in progress.

###  Deep Deterministic Policy Gradient (DDPG)
https://arxiv.org/abs/1509.02971

DDPG is an off-policy, model-free reinforcement learning algorithm that combines deterministic policy gradient methods with experience replays and actor-critic networks. 
It is used for continuous action spaces.

Used in this project to solve Lunar Lander problem from Gymnasium (similar to the deprecated OpenAI Gym) which is considered solved at +200 score.
https://gymnasium.farama.org/environments/box2d/lunar_lander/

Requires Python3, PyTorch, NumPy, Matplotlib and Gymnasium

### Soft Actor-Critic (SAC)
https://arxiv.org/abs/1801.01290