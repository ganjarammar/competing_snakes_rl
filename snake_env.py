import pygame
import numpy as np
from enum import Enum
import random
import os

# Center the pygame window
os.environ['SDL_VIDEO_CENTERED'] = '1'

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameEnv:
    def __init__(self, width=640, height=480, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Competing Snakes RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()

    def reset(self):
        # Initialize snakes positions (one on left side, one on right side)
        self.snake1 = [(self.width//4//self.grid_size * self.grid_size, 
                       self.height//2//self.grid_size * self.grid_size)]
        self.snake2 = [(3*self.width//4//self.grid_size * self.grid_size, 
                       self.height//2//self.grid_size * self.grid_size)]
        
        self.snake1_direction = Direction.RIGHT
        self.snake2_direction = Direction.LEFT
        
        self.food = self._place_food()
        self.score1 = 0
        self.score2 = 0
        self.frame_iteration = 0
        
        # Process any pending events
        pygame.event.pump()
        
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width-self.grid_size)//self.grid_size) * self.grid_size
            y = random.randint(0, (self.height-self.grid_size)//self.grid_size) * self.grid_size
            food_pos = (x, y)
            if food_pos not in self.snake1 and food_pos not in self.snake2:
                return food_pos

    def _get_state(self):
        # Create state representations for both snakes
        state1 = self._get_snake_state(self.snake1, self.snake1_direction, self.snake2)
        state2 = self._get_snake_state(self.snake2, self.snake2_direction, self.snake1)
        return state1, state2

    def _get_snake_state(self, snake, direction, other_snake):
        head = snake[0]
        
        # Danger straight, right, left
        point_l = point_r = point_u = point_d = head
        if direction == Direction.RIGHT:
            point_r = (head[0] + self.grid_size, head[1])
            point_u = (head[0], head[1] - self.grid_size)
            point_d = (head[0], head[1] + self.grid_size)
        elif direction == Direction.LEFT:
            point_l = (head[0] - self.grid_size, head[1])
            point_u = (head[0], head[1] - self.grid_size)
            point_d = (head[0], head[1] + self.grid_size)
        elif direction == Direction.UP:
            point_u = (head[0], head[1] - self.grid_size)
            point_l = (head[0] - self.grid_size, head[1])
            point_r = (head[0] + self.grid_size, head[1])
        elif direction == Direction.DOWN:
            point_d = (head[0], head[1] + self.grid_size)
            point_l = (head[0] - self.grid_size, head[1])
            point_r = (head[0] + self.grid_size, head[1])

        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        food = self.food
        
        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            food[0] < head[0],  # food left
            food[0] > head[0],  # food right
            food[1] < head[1],  # food up
            food[1] > head[1]   # food down
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, pt):
        # Check if point collides with walls or snakes
        if (pt[0] >= self.width or pt[0] < 0 or 
            pt[1] >= self.height or pt[1] < 0):
            return True
        if pt in self.snake1[1:] or pt in self.snake2:
            return True
        return False

    def step(self, action1, action2):
        self.frame_iteration += 1
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, 0, 0, True
        
        # Process actions for both snakes
        reward1, done1 = self._move_snake(1, action1)
        reward2, done2 = self._move_snake(2, action2)
        
        done = done1 or done2
        if done:
            if done1 and done2:
                reward1 = reward2 = -10
            elif done1:
                reward1, reward2 = -10, 10
            else:
                reward1, reward2 = 10, -10
        
        return self._get_state(), reward1, reward2, done

    def _move_snake(self, snake_num, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        snake = self.snake1 if snake_num == 1 else self.snake2
        direction = self.snake1_direction if snake_num == 1 else self.snake2_direction
        
        idx = clock_wise.index(direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn
            
        if snake_num == 1:
            self.snake1_direction = new_dir
        else:
            self.snake2_direction = new_dir
            
        x = snake[0][0]
        y = snake[0][1]
        if direction == Direction.RIGHT:
            x += self.grid_size
        elif direction == Direction.LEFT:
            x -= self.grid_size
        elif direction == Direction.DOWN:
            y += self.grid_size
        elif direction == Direction.UP:
            y -= self.grid_size
            
        new_head = (x, y)
        
        # Check if snake died
        if self._is_collision(new_head):
            return -10, True
            
        snake.insert(0, new_head)
        
        # Check if snake ate food
        reward = 0
        if new_head == self.food:
            reward = 10
            if snake_num == 1:
                self.score1 += 1
            else:
                self.score2 += 1
            self.food = self._place_food()
        else:
            snake.pop()
            
        if snake_num == 1:
            self.snake1 = snake
        else:
            self.snake2 = snake
            
        return reward, False

    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), 
                        pygame.Rect(self.food[0], self.food[1], 
                                  self.grid_size-2, self.grid_size-2))
        
        # Draw snake1 (blue)
        for pt in self.snake1:
            pygame.draw.rect(self.screen, (0, 0, 255), 
                           pygame.Rect(pt[0], pt[1], 
                                     self.grid_size-2, self.grid_size-2))
            
        # Draw snake2 (green)
        for pt in self.snake2:
            pygame.draw.rect(self.screen, (0, 255, 0), 
                           pygame.Rect(pt[0], pt[1], 
                                     self.grid_size-2, self.grid_size-2))
        
        # Draw scores
        score_text = f'Blue: {self.score1}  Green: {self.score2}'
        text_surface = self.font.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(10)  # Control game speed

    def close(self):
        pygame.quit()
