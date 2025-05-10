import gymnasium as gym
import numpy as np
from ppo import Agent
from utils import plot_learning_curve
import torch as T
import itertools

if __name__ == '__main__':
    mode = input('Train? y/n: ').lower()

    env = gym.make('LunarLander-v3', continuous=True, render_mode='human' if mode == 'n' else None)
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size,
                  n_epochs=n_epochs, alpha=alpha,
                  input_dims=env.observation_space.shape)

    n_games = itertools.count()
    figure_file = 'plots/LunarLander-v3.png'

    best_score = -np.inf  # Set to a large negative number initially
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    if mode == 'y':  # Training mode
        agent.load_models()
        for i in n_games:
            observation, _ = env.reset()
            done = False
            score = 0

            while not done:
                with T.no_grad():
                    action, prob, val = agent.choose_action(observation)

                observation_, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                score += reward
                n_steps += 1

                agent.remember(observation, action, prob, val, reward, done)

                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1

                observation = observation_

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print(f'Episode {i}, Score: {score:.1f}, Avg Score: {avg_score:.1f}, '
                  f'Time Steps: {n_steps}, Learning Steps: {learn_iters}')
            
            x = [i + 1 for i in range(len(score_history))]
            plot_learning_curve(x, score_history, figure_file)
    else:  # Evaluation mode
        agent.load_models()
        for i in n_games:
            observation, _ = env.reset()
            done = False
            score = 0

            while not done:
                with T.no_grad():
                    action, _, _ = agent.choose_action(observation)
                
                observation_, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                score += reward
                observation = observation_

            print(f'Episode {i}, Score: {score:.1f}')
