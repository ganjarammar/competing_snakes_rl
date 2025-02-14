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

def select_curriculum():
    print("\nPilih metode Curriculum Learning:")
    print("1. Dynamic Food (Jumlah makanan berkurang seiring peningkatan)")
    print("2. Dynamic Arena (Arena membesar seiring peningkatan)")
    print("3. Combined (Kombinasi food dan arena)")
    
    while True:
        try:
            choice = int(input("Masukkan pilihan (1-3): "))
            if choice in [1, 2, 3]:
                if choice == 1:
                    return 'food'
                elif choice == 2:
                    return 'arena'
                else:
                    return 'combined'
            else:
                print("Pilihan tidak valid. Silakan pilih 1-3.")
        except ValueError:
            print("Input tidak valid. Masukkan angka 1-3.")

def train():
    curriculum_type = select_curriculum()
    
    plt.ion()
    plt.figure(figsize=(10, 5))
    
    plot_scores1 = []
    plot_scores2 = []
    plot_mean_scores1 = []
    plot_mean_scores2 = []
    total_score1 = 0
    total_score2 = 0
    record = 0
    
    # Initialize environment and agents with selected curriculum
    env = SnakeGameEnv(curriculum_type=curriculum_type)
    agent1 = Agent()
    agent2 = Agent()
    
    try:
        game_count = 0  # Start from 0 for terminal/plot tracking
        while True:
            # Get old states
            state_old1 = agent1.get_state(env, 1)
            state_old2 = agent2.get_state(env, 2)
            
            # Get moves
            final_move1 = agent1.get_action(state_old1)
            final_move2 = agent2.get_action(state_old2)
            
            # Perform moves and get new states
            done, score1, score2, reward1, reward2 = env.play_step(final_move1, final_move2)
            state_new1 = agent1.get_state(env, 1)
            state_new2 = agent2.get_state(env, 2)
            
            # Train short memory
            agent1.train_short_memory(state_old1, final_move1, reward1, state_new1, done)
            agent2.train_short_memory(state_old2, final_move2, reward2, state_new2, done)
            
            # Remember
            agent1.remember(state_old1, final_move1, reward1, state_new1, done)
            agent2.remember(state_old2, final_move2, reward2, state_new2, done)
            
            if done:
                # Train long memory (replay)
                env.reset()  # This will update current_game to n_games
                game_count += 1
                env.n_games = game_count  # Update game count in environment after game is done
                agent1.n_games = game_count  # Ensure agent1 also tracks the same game count
                agent2.n_games = game_count  # Ensure agent2 also tracks the same game count
                agent1.train_long_memory()
                agent2.train_long_memory()
                
                # Update scores
                plot_scores1.append(score1)
                plot_scores2.append(score2)
                total_score1 += score1
                total_score2 += score2
                mean_score1 = total_score1 / agent1.n_games
                mean_score2 = total_score2 / agent2.n_games
                plot_mean_scores1.append(mean_score1)
                plot_mean_scores2.append(mean_score2)
                
                # Update curriculum based on average performance
                if env.update_curriculum((mean_score1 + mean_score2) / 2):
                    print(f"\nLevel up! Current level: {env.current_level}")
                    if curriculum_type == 'food':
                        print(f"Food count reduced to: {env.food_count}")
                    elif curriculum_type == 'arena':
                        print(f"Arena size increased to: {env.width}x{env.height}")
                    else:
                        print(f"Food count: {env.food_count}, Arena size: {env.width}x{env.height}")
                
                # Plot
                plot(plot_scores1, plot_scores2, plot_mean_scores1, plot_mean_scores2)
                
                print(f'Game {game_count}, Score1: {score1}, Mean Score1: {mean_score1:.2f}')
                print(f'Score2: {score2}, Mean Score2: {mean_score2:.2f}')
                
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        pygame.quit()

if __name__ == '__main__':
    train()
