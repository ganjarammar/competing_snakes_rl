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
    def __init__(self, width=640, height=520, grid_size=20):
        self.width = width
        self.height = height - 40  # Adjust height to fit arena below header
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
        # Snake 1 - starts with head and one body segment, moving right
        head1_x = self.width//4//self.grid_size * self.grid_size
        head1_y = self.height//2//self.grid_size * self.grid_size
        self.snake1 = [
            (head1_x, head1_y),                    # Head
            (head1_x - self.grid_size, head1_y)    # Body segment (one unit behind head)
        ]
        self.snake1_direction = Direction.RIGHT

        # Snake 2 - starts with head and one body segment, moving left
        head2_x = 3*self.width//4//self.grid_size * self.grid_size
        head2_y = self.height//2//self.grid_size * self.grid_size
        self.snake2 = [
            (head2_x, head2_y),                    # Head
            (head2_x + self.grid_size, head2_y)    # Body segment (one unit behind head)
        ]
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
            
        # Update head position based on direction
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
            
        # Check if snake died
        new_pos = (x, y)
        if self._is_collision(new_pos):
            return -10, True

        # Store the last position before moving (for growing)
        last_pos = snake[-1]

        # Move body segments
        for i in range(len(snake)-1, 0, -1):
            snake[i] = snake[i-1]
        
        # Update head position
        snake[0] = new_pos
            
        # Check if snake ate food
        reward = 0
        if new_pos == self.food:
            reward = 10
            if snake_num == 1:
                self.score1 += 1
                # Add new segment at the end using the last position
                self.snake1.append(last_pos)
            else:
                self.score2 += 1
                # Add new segment at the end using the last position
                self.snake2.append(last_pos)
            self.food = self._place_food()
            
        if snake_num == 1:
            self.snake1 = snake
        else:
            self.snake2 = snake
            
        return reward, False

    def render(self, game_number):
        # Fill the screen with a white background for the header
        self.screen.fill((255, 255, 255), (0, 0, self.width, 40))
        
        # Draw scores with matching colors
        text_surface1 = self.font.render('Snake 1: ' + str(self.score1), True, (0, 0, 255))  # Blue
        text_surface2 = self.font.render('Snake 2: ' + str(self.score2), True, (255, 0, 0))  # Red
        game_number_surface = self.font.render(f'Game Number: {game_number:6}', True, (0, 0, 0))  # Black
        self.screen.blit(text_surface1, (10, 5))
        self.screen.blit(text_surface2, (150, 5))
        self.screen.blit(game_number_surface, (self.width - 250, 5))
        
        # Fill the rest of the screen with black for the arena
        self.screen.fill((0, 0, 0), (0, 40, self.width, self.height))
        
        # Draw food (green)
        pygame.draw.rect(self.screen, (0, 255, 0), 
                         pygame.Rect(self.food[0], self.food[1] + 40, 
                                     self.grid_size-2, self.grid_size-2))
        
        # Draw snake1 (blue for body, light blue for head)
        for i, pt in enumerate(self.snake1):
            color = (100, 100, 255) if i == 0 else (0, 0, 255)  # Light blue head, blue body
            if i == 0:  # Head
                pygame.draw.rect(self.screen, color, 
                               pygame.Rect(pt[0], pt[1] + 40, self.grid_size-2, self.grid_size-2))
                # Add eyes to make head more distinctive
                eye_size = 4
                if self.snake1_direction in [Direction.RIGHT, Direction.LEFT]:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + self.grid_size//2, pt[1] + 40 + self.grid_size//3), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + self.grid_size//2, pt[1] + 40 + 2*self.grid_size//3), eye_size)
                else:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + self.grid_size//3, pt[1] + 40 + self.grid_size//2), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + 2*self.grid_size//3, pt[1] + 40 + self.grid_size//2), eye_size)
            else:  # Body
                pygame.draw.rect(self.screen, color, 
                               pygame.Rect(pt[0], pt[1] + 40, self.grid_size-2, self.grid_size-2))
        
        # Draw snake2 (red for body, pink for head)
        for i, pt in enumerate(self.snake2):
            color = (255, 100, 100) if i == 0 else (255, 0, 0)  # Pink head, red body
            if i == 0:  # Head
                pygame.draw.rect(self.screen, color, 
                               pygame.Rect(pt[0], pt[1] + 40, self.grid_size-2, self.grid_size-2))
                # Add eyes to make head more distinctive
                eye_size = 4
                if self.snake2_direction in [Direction.RIGHT, Direction.LEFT]:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + self.grid_size//2, pt[1] + 40 + self.grid_size//3), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + self.grid_size//2, pt[1] + 40 + 2*self.grid_size//3), eye_size)
                else:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + self.grid_size//3, pt[1] + 40 + self.grid_size//2), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (pt[0] + 2*self.grid_size//3, pt[1] + 40 + self.grid_size//2), eye_size)
            else:  # Body
                pygame.draw.rect(self.screen, color, 
                               pygame.Rect(pt[0], pt[1] + 40, self.grid_size-2, self.grid_size-2))
        
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()
