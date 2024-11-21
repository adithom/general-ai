from ddpg import Agent
import gymnasium as gym
import numpy as np
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=env.observation_space.shape, tau=0.001, env = env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])
    np.random.seed(0)

    score_history = []
    #render_env = False

    for i in range(1000):
        obs, info = env.reset()
        done = False
        score = 0
        while not done:
            #if render_env:
                #env.render()
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        running_avg = np.mean(score_history[-100:])
        print('Episode ', i, 'score %.2f' % score, '100 game running average %.2f' % running_avg)
        #if running_avg > 200 and not render_env:
        #    print(f"Environment solved in {i} episodes! Rendering enabled.")
        #    render_env = True
        if i % 25 == 0:
            agent.save_models()

    filename = 'LunarLander-alpha000025-beta00025-400-300.png'
    plot_learning_curve(score_history, filename, window=100)