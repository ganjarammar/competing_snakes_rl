# Competing Snakes Reinforcement Learning

This project implements a competitive version of the classic Snake game where two AI-controlled snakes learn to compete against each other using Deep Q-Learning (DQN). The snakes must learn to navigate the environment, collect food, and avoid collisions while competing with each other.

## How Reinforcement Learning Works in This Game

### 1. State Space (What the Snakes Can "See")
Each snake receives 11 input values representing its current state:
- Danger straight ahead (1/0)
- Danger to the right (1/0)
- Danger to the left (1/0)
- Current direction (4 values: up/down/left/right)
- Food location relative to head (4 values: food_left/food_right/food_up/food_down)

### 2. Action Space (What the Snakes Can Do)
Each snake can perform three actions:
- [1,0,0]: Continue straight
- [0,1,0]: Turn right
- [0,0,1]: Turn left

### 3. Reward System
The snakes receive rewards based on their actions:
- +10 points for eating food
- -10 points for colliding with walls or other snake
- 0 points for surviving each move

### 4. Learning Process
- Both snakes use Deep Q-Networks (DQN) to learn optimal strategies
- Experience Replay: Stores past experiences in memory for batch learning
- Epsilon-greedy exploration: Balances random exploration vs. exploitation
- The neural network learns to predict the best action based on the current state

## Project Structure

- `snake_env.py`: Game environment with Pygame
- `model.py`: DQN implementation and Agent class
- `train.py`: Training loop and visualization
- `environment.yml`: Conda environment configuration

## How to Run the Project

1. **Create the Conda Environment**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Environment**
   ```bash
   conda activate snake_rl
   ```

3. **Set OpenMP Environment Variable**
   ```bash
   set KMP_DUPLICATE_LIB_OK=TRUE
   ```

4. **Run the Training**
   ```bash
   python train.py
   ```

## What You'll See When Running

1. **Game Window**
   - Blue snake: First AI agent
   - Green snake: Second AI agent
   - Red squares: Food
   - Score display in top-left corner

2. **Training Plot**
   - Real-time score plotting for both snakes
   - Moving averages to show learning progress
   - Updated after each game

## How the Snakes Learn

1. **Initial Phase**
   - Snakes make mostly random moves (high exploration)
   - Learn basic survival skills (avoiding walls)

2. **Middle Phase**
   - Start developing food-seeking behavior
   - Learn to avoid basic collisions

3. **Advanced Phase**
   - Develop sophisticated strategies
   - Learn to compete for food
   - Improve survival techniques

## Controls

- Close the game window or press Ctrl+C in the terminal to stop training
- The training automatically saves the best scores
- Each training session builds upon previous learning

## Notes

- The snakes start with more random actions and gradually become more strategic
- Training can take several hours to see significant improvements
- The matplotlib window shows learning progress in real-time
- CPU-only implementation, optimized for systems without dedicated GPUs
