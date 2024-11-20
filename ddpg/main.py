from ddpg import Agent
import gymnasium as gym
import numpy as np
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=env.observation_space.shape, tau=0.001, env = env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])
    np.random.seed(0)

    score_history = []
    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))
        if i % 25 == 0:
            agent.save_models()

    filename = 'LunarLander-alpha000025-beta00025-400-300.png'
    plot_learning_curve(score_history, filename, window=100)