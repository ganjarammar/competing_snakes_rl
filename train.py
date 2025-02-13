import pygame
import numpy as np
from snake_env import SnakeGameEnv
from model import Agent
import matplotlib.pyplot as plt

def plot(scores1, scores2, mean_scores1, mean_scores2):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Define colors
    SNAKE1_COLOR = 'blue'
    SNAKE1_MEAN_COLOR = 'lightblue'
    SNAKE2_COLOR = 'darkred'
    SNAKE2_MEAN_COLOR = 'lightcoral'
    
    # Plot with consistent colors
    plt.plot(scores1, color=SNAKE1_COLOR, label='Snake 1')
    plt.plot(scores2, color=SNAKE2_COLOR, label='Snake 2')
    plt.plot(mean_scores1, color=SNAKE1_MEAN_COLOR, label='Snake 1 Mean')
    plt.plot(mean_scores2, color=SNAKE2_MEAN_COLOR, label='Snake 2 Mean')
    
    plt.ylim(ymin=0)
    plt.legend()
    if len(scores1) > 0:
        plt.text(len(scores1)-1, scores1[-1], str(scores1[-1]))
    if len(scores2) > 0:
        plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    plt.draw()
    plt.pause(0.1)

def train():
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 5))
    
    plot_scores1 = []
    plot_scores2 = []
    plot_mean_scores1 = []
    plot_mean_scores2 = []
    total_score1 = 0
    total_score2 = 0
    record = 0
    
    # Initialize environment and agents
    env = SnakeGameEnv()
    agent1 = Agent()
    agent2 = Agent()
    
    try:
        game_number = 0  # Initialize game number
        while True:
            game_number += 1  # Increment game number each loop
            
            # Get old states
            state1, state2 = env.reset()
            
            done = False
            while not done:
                # Get moves from both agents
                action1 = agent1.get_action(state1)
                action2 = agent2.get_action(state2)
                
                # Perform moves and get new states
                (new_state1, new_state2), reward1, reward2, done = env.step(action1, action2)
                
                # Train both agents
                agent1.train_step([state1], [action1], [reward1], [new_state1], [done])
                agent2.train_step([state2], [action2], [reward2], [new_state2], [done])
                
                state1 = new_state1
                state2 = new_state2
                
                # Render game with current game number
                env.render(game_number)
                
                if done:
                    break
            
            # Train long memory
            agent1.train_long_memory()
            agent2.train_long_memory()
            
            agent1.n_games += 1
            agent2.n_games += 1
            
            # Update scores and plotting
            plot_scores1.append(env.score1)
            plot_scores2.append(env.score2)
            total_score1 += env.score1
            total_score2 += env.score2
            mean_score1 = total_score1 / agent1.n_games
            mean_score2 = total_score2 / agent2.n_games
            plot_mean_scores1.append(mean_score1)
            plot_mean_scores2.append(mean_score2)
            
            if env.score1 > record or env.score2 > record:
                record = max(env.score1, env.score2)
            
            print('Game', agent1.n_games, 'Score1:', env.score1, 'Score2:', env.score2, 'Record:', record)
            
            # Plot progress
            plot(plot_scores1, plot_scores2, plot_mean_scores1, plot_mean_scores2)
            
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    finally:
        env.close()
        plt.ioff()  # Turn off interactive mode
        plt.show()

if __name__ == '__main__':
    train()
