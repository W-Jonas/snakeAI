import pygame
import random
from stable_baselines3 import PPO
from gym.spaces import Discrete, Box, Tuple, Dict
import numpy as np
from gym import Env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Initialize Pygame
pygame.init()

# Set up the game window
width, height = 640, 480
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Define colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Define the snake class
class Snake:
    def __init__(self):
        self.size = 1
        self.positions = [(random.randint(0, 63) * 10, random.randint(0, 47) * 10)]
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
    
    def move(self):
        x, y = self.positions[0]
        if self.direction == "UP":
            y -= 10
        elif self.direction == "DOWN":
            y += 10
        elif self.direction == "LEFT":
            x -= 10
        elif self.direction == "RIGHT":
            x += 10
        self.positions.insert(0, (x, y))
        if len(self.positions) > self.size:
            self.positions.pop()
    
    def change_direction(self, direction):
        if direction == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif direction == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif direction == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif direction == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"
        elif direction == "NOOP":
            pass
    
    def draw(self, window):
        for position in self.positions:
            pygame.draw.rect(window, GREEN, (position[0], position[1], 10, 10))

# Define the food class
class Food:
    def __init__(self):
        self.position = None
    
    def spawn(self, snake_positions):
        valid_positions = []
        for x in range(0, width, 10):
            for y in range(0, height, 10):
                if (x, y) not in snake_positions:
                    valid_positions.append((x, y))
        self.position = random.choice(valid_positions)
    
    def draw(self, window):
        pygame.draw.rect(window, RED, (self.position[0], self.position[1], 10, 10))

# Define the SnakeGame environment
class SnakeGameEnv(Env):
    def __init__(self):
        self.action_space = Discrete(5)
        self.max_body_parts_positions = int(width * height / (10 * 10) * 2)  # Each body part has 2 positions (x and y)
        self.observation_space = Box(low=0, high=1, shape=(10 + self.max_body_parts_positions,), dtype=np.float32)
        self.snake = Snake()
        self.food = Food()
        self.food.spawn(self.snake.positions)
        self.done = False
        self.reward = 0
        self.snake.change_direction(random.choice(["UP", "DOWN", "LEFT", "RIGHT"]))
        self.reset()
    
    def reset(self):
        #self.render()
        self.snake = Snake()
        self.food = Food()
        self.food.spawn(self.snake.positions)
        self.done = False
        self.reward = 0
        self.snake.change_direction(random.choice(["UP", "DOWN", "LEFT", "RIGHT"]))
        return self._get_observation()
    
    def step(self, action):
        if action == 0:
            self.snake.change_direction("UP")
        elif action == 1:
            self.snake.change_direction("DOWN")
        elif action == 2:
            self.snake.change_direction("LEFT")
        elif action == 3:
            self.snake.change_direction("RIGHT")
        elif action == 4:
            self.snake.change_direction("NOOP")

        reward = 0
        done = False
        # Check the previous move
        previous_move = self.snake.direction

        # Update the snake's direction
        self.snake.move()

        # Check the current move
        current_move = self.snake.direction

        # Add punishment for erratic movement
        if previous_move != current_move:
            reward -= 1

        # Check the distance to the food
        distance_to_food = np.sqrt((self.snake.positions[0][0] - self.food.position[0]) ** 2 + (self.snake.positions[0][1] - self.food.position[1]) ** 2)
        if distance_to_food != 0:
            reward += (1 / distance_to_food) * 10

        # Check if the snake has collided with itself
        if self.snake.positions[0] in self.snake.positions[1:]:
            reward -= 10
            done = True
        
        # Check if the snake is out of bounds
        if self.snake.positions[0][0] < 0 or self.snake.positions[0][0] > width - 10:
            reward -= 10
            done = True
        
        # Check if the snake is out of bounds
        if self.snake.positions[0][1] < 0 or self.snake.positions[0][1] > height - 10:
            reward -= 10
            done = True

        
        # Check if the snake has eaten the food
        if self.snake.positions[0] == self.food.position:
            self.snake.size += 1
            self.food.spawn(self.snake.positions)
            reward += 100

        self.render()
        
        observation = self._get_observation()
        return observation, reward, done, {}
    
    def render(self):
        window = pygame.display.set_mode((width, height))
        window.fill(BLACK)
        self.snake.draw(window)
        self.food.draw(window)
        pygame.display.flip()
        pygame.event.get()
        pygame.display.set_caption("Snake Game | Score: " + str(self.snake.size - 1))
    
    def _get_observation(self):
        snake_x, snake_y = self.snake.positions[0]
        food_x, food_y = self.food.position
        body_parts = self.snake.size
        distance_to_food = np.sqrt((snake_x - food_x) ** 2 + (snake_y - food_y) ** 2)
        
        # Convert direction to one-hot encoding
        direction = np.zeros(4)
        if self.snake.direction == "UP":
            direction[0] = 1
        elif self.snake.direction == "DOWN":
            direction[1] = 1
        elif self.snake.direction == "LEFT":
            direction[2] = 1
        elif self.snake.direction == "RIGHT":
            direction[3] = 1
        
        # Normalize each element according to its own scale
        snake_x /= width
        snake_y /= height
        food_x /= width
        food_y /= height
        body_parts /= (width * height / (10 * 10))  # Assuming each body part occupies 10x10 space
        distance_to_food /= np.sqrt(width ** 2 + height ** 2)  # Maximum possible distance
        
        body_parts_positions = []
        for i in range(self.max_body_parts_positions // 2):  # Divide by 2 here
            if i < len(self.snake.positions):
                position = self.snake.positions[i]
                body_parts_positions.append(position[0] / width)  # Normalize x
                body_parts_positions.append(position[1] / height)  # Normalize y
            else:
                body_parts_positions.extend([0, 0])  # Pad with zeros

        observation = np.array([snake_x, snake_y, food_x, food_y, body_parts, distance_to_food] + list(direction) + body_parts_positions)

        return observation

def make_env():
    def _init():
        return SnakeGameEnv()
    return _init

def train():
    
    # Train the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("snake_model")

def continue_training():

    # Load the trained model with the new environment
    model = PPO.load("snake_model", env)

    # Continue training the PPO agent
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("snake_model")

def play():
    # Load the trained model
    model = PPO.load("snake_model", env)

    # Play the game
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    # Create multiple environments
    num_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    #train()
    while True:
        continue_training()

    #play()
        
# Check the environment
#env = SnakeGameEnv()
#check_env(env, warn=True)
