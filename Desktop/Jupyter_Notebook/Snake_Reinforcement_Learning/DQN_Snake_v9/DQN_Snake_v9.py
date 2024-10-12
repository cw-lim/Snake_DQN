import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import torch.nn.functional as F  # Import functional module
import cv2

# Define constants
GRID_SIZE = 20
PIXEL_SIZE = 30
FOOD_REWARD = 50
MOVE_PENALTY = -0.1
CLOSER_REWARD = 1.2
FURTHER_PENALTY = -1
DEATH_PENALTY = -20
TIME_PENALTY = -0.1
MIN_EPSILON = 0.0
FPS = 60
EXPERIENCE_REPLAY_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.8
MODEL_SAVE_INTERVAL = 5000
LEARNING_RATE = 0.00005
EPISODES = 1000000  # Set the number of episodes for training
STUCK_THRESHOLD = 2500  # Number of steps to consider the snake as stuck
DECAY_RATE = 0.99999  # Epsilon decay rate
STAGNATION_THRESHOLD = 50000  # Number of episodes for stagnation

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeGame:
    def __init__(self, render=True):
        self.grid_size = GRID_SIZE
        self.pixel_size = PIXEL_SIZE
        self.render_enabled = render
        if self.render_enabled:
            self.screen = pygame.display.set_mode((self.grid_size * self.pixel_size, self.grid_size * self.pixel_size))
            pygame.display.set_caption('Improved Snake Game with DQN')
        self.reset()
        self.clock = pygame.time.Clock()  # For controlling the FPS

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Initial direction is moving right
        self.place_food()
        self.score = 0
        self.steps_since_food = 0
        self.previous_distance = self.get_distance(self.snake[0], self.food)
        self.steps_since_last_progress = 0  # Initialize steps since last progress
        return self.get_state()

    def place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if self.food not in self.snake:
                break
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Calculate normalized positions
        head_x_norm = head_x / self.grid_size
        head_y_norm = head_y / self.grid_size
        food_x_norm = food_x / self.grid_size
        food_y_norm = food_y / self.grid_size

        # Direction vectors (normalized)
        direction_x, direction_y = self.direction
        magnitude_direction = np.sqrt(direction_x**2 + direction_y**2)
        if magnitude_direction != 0:
            direction_x /= magnitude_direction
            direction_y /= magnitude_direction

        # Distance to food (normalized)
        distance_to_food = self.get_distance(self.snake[0], self.food) / (self.grid_size * 2)

        # Angle to food (normalized)
        relative_food_x = food_x - head_x
        relative_food_y = food_y - head_y
        dot_product = direction_x * relative_food_x + direction_y * relative_food_y
        magnitude_food = np.sqrt(relative_food_x**2 + relative_food_y**2)
        angle_to_food = np.arccos(np.clip(dot_product / magnitude_food, -1.0, 1.0)) / np.pi if magnitude_food != 0 else 0

        # Normalize distances to the nearest walls
        distance_to_left_wall = head_x / self.grid_size
        distance_to_right_wall = (self.grid_size - 1 - head_x) / self.grid_size
        distance_to_top_wall = head_y / self.grid_size
        distance_to_bottom_wall = (self.grid_size - 1 - head_y) / self.grid_size

        # Distance to the closest body segment
        distances_to_body_segments = [self.get_distance(self.snake[0], segment) for segment in self.snake[1:]]
        distance_to_closest_body_segment = min(distances_to_body_segments) / self.grid_size if distances_to_body_segments else 1

        # Snake length (normalized)
        snake_length = len(self.snake) / (self.grid_size * 2)

        # Danger detection: Straight, Left, Right
        danger_straight = self.is_collision((head_x + self.direction[0], head_y + self.direction[1]))

        left_direction = (-self.direction[1], self.direction[0])  # Left turn
        danger_left = self.is_collision((head_x + left_direction[0], head_y + left_direction[1]))

        right_direction = (self.direction[1], -self.direction[0])  # Right turn
        danger_right = self.is_collision((head_x + right_direction[0], head_y + right_direction[1]))

        # Current direction (encoded as one-hot)
        direction_left = int(self.direction == (-1, 0))
        direction_right = int(self.direction == (1, 0))
        direction_up = int(self.direction == (0, -1))
        direction_down = int(self.direction == (0, 1))

        # Calculate the relative direction of food
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)

        # Final state vector: Combined features
        state = np.array([
            head_x_norm, head_y_norm,                 # Snake head position (normalized)
            food_x_norm, food_y_norm,                 # Food position (normalized)
            direction_x, direction_y,                 # Snake direction
            distance_to_food,                         # Distance to food
            angle_to_food,                            # Angle to food
            distance_to_left_wall, distance_to_right_wall, distance_to_top_wall, distance_to_bottom_wall,  # Wall distances (normalized)
            distance_to_closest_body_segment,         # Closest body segment
            snake_length,                             # Snake length (normalized)
            danger_straight, danger_left, danger_right,  # Dangers
            direction_left, direction_right, direction_up, direction_down,  # Current direction (one-hot encoded)
            food_left, food_right, food_up, food_down  # Food relative direction
        ], dtype=np.float32)

        return state

    def step(self, action):
        self.change_direction(action)
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
    
        reward = MOVE_PENALTY
        done = False
    
        if self.is_collision(new_head):
            reward = DEATH_PENALTY
            done = True
        else:
            self.snake = [new_head] + self.snake[:-1]
            new_distance = self.get_distance(self.snake[0], self.food)
            self.previous_distance = new_distance
            self.steps_since_food += 1
    
            if new_head == self.food:
                self.snake = [new_head] + self.snake
                self.score += 1  # Update the score
                reward = FOOD_REWARD
                self.place_food()
                self.steps_since_food = 0
                self.previous_distance = self.get_distance(self.snake[0], self.food)
            else:
                old_distance = self.previous_distance
                if new_distance < old_distance:
                    reward += CLOSER_REWARD
                else:
                    reward += FURTHER_PENALTY
    
                reward += - (self.steps_since_food * TIME_PENALTY)
    
                if self.steps_since_last_progress >= STUCK_THRESHOLD:
                    if self.detect_loop() or self.is_in_small_area():
                        done = True
                else:
                    if new_distance != old_distance:
                        self.steps_since_last_progress = 0
                    else:
                        self.steps_since_last_progress += 1
        
        return self.get_state(), reward, done
    
    def detect_loop(self):
        recent_positions = deque(maxlen=20)  # Track last 20 positions
        recent_positions.append(self.snake[0])
        
        # Check if snake is revisiting recent positions too frequently
        revisit_count = sum(1 for pos in self.snake[1:] if pos in recent_positions)
        if revisit_count >= len(self.snake) / 2:  # Arbitrary threshold; adjust as needed
            return True
        
        # Update previous distance
        current_distance = self.get_distance(self.snake[0], self.food)
        if abs(current_distance - self.previous_distance) < 1:
            return True
    
        # Update previous distance for next check
        self.previous_distance = current_distance
        
        return False
    
    def is_in_small_area(self):
        # Check if the snake is confined to a small area
        snake_positions = set(self.snake)
        min_x = min(x for x, y in snake_positions)
        max_x = max(x for x, y in snake_positions)
        min_y = min(y for x, y in snake_positions)
        max_y = max(y for x, y in snake_positions)
        return (max_x - min_x <= 10) and (max_y - min_y <= 10)  # Adjust the area size threshold

    def change_direction(self, action):
        left_turns = {
            (1, 0): (0, 1),   # Moving right, turn left -> down
            (0, 1): (-1, 0),  # Moving down, turn left -> left
            (-1, 0): (0, -1), # Moving left, turn left -> up
            (0, -1): (1, 0)   # Moving up, turn left -> right
        }
        
        right_turns = {
            (1, 0): (0, -1),  # Moving right, turn right -> up
            (0, 1): (1, 0),   # Moving down, turn right -> right
            (-1, 0): (0, 1),  # Moving left, turn right -> down
            (0, -1): (-1, 0)  # Moving up, turn right -> left
        }
        
        if action == 0:  # Left
            self.direction = left_turns[self.direction]
        elif action == 1:  # Right
            self.direction = right_turns[self.direction]
        # No action means move straight (no change in direction)

    def is_collision(self, position):
        x, y = position
        return (x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or
                position in self.snake)
    
    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(segment[0] * self.pixel_size, segment[1] * self.pixel_size, self.pixel_size, self.pixel_size))
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0] * self.pixel_size, self.food[1] * self.pixel_size, self.pixel_size, self.pixel_size))
        pygame.display.flip()
        self.clock.tick(FPS)  # Cap the frame rate

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # Initial number of units
        self.ln1 = nn.LayerNorm(1024)  # Use LayerNorm
        self.dropout1 = nn.Dropout(0.1)  # Dropout rate

        self.fc2 = nn.Linear(1024, 1024)  # Fixed input size to match output of fc1
        self.ln2 = nn.LayerNorm(1024)  # Use LayerNorm
        self.dropout2 = nn.Dropout(0.1)  # Dropout rate

        self.fc3 = nn.Linear(1024, 1024)  # Fixed input size to match output of fc2
        self.ln3 = nn.LayerNorm(1024)  # Use LayerNorm
        self.dropout3 = nn.Dropout(0.1)  # Dropout rate

        self.fc4 = nn.Linear(1024, output_dim)  # Output layer

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.ln3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, input_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.q_network = DQN(input_dim, action_dim).to(device)
        self.target_network = DQN(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        self.replay_buffer = deque(maxlen=EXPERIENCE_REPLAY_SIZE)
        self.update_target_network()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()  # Best action

    def store_experience(self, experience):
        self.replay_buffer.append(experience)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None  # Not enough data to train
    
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
    
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()  # Step the learning rate scheduler
    
        return loss.item()  # Return the loss value

def decay_epsilon(epsilon, episode, decay_rate=DECAY_RATE, min_epsilon=MIN_EPSILON):
    return max(min_epsilon, epsilon * decay_rate)

def train_snake_game(render=False):
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 25
    action_dim = 3
    agent = DQNAgent(input_dim, action_dim, device)
    game = SnakeGame(render=render)
    epsilon = 1.0
    best_score = 0
    best_model_path = "best_snake_dqn_model.pth"
    stagnation_counter = 0

    for episode in range(EPISODES):
        state = game.reset()
        total_reward = 0
        done = False
        stuck_steps = 0

        if episode % 10 == 0:
            while not done and stuck_steps < STUCK_THRESHOLD:
                action = agent.select_action(state, epsilon)
                next_state, reward, done = game.step(action)
                agent.store_experience((state, action, reward, next_state, done))
                loss = agent.train()
                state = next_state
                total_reward += reward

                if render:
                    game.render()

                stuck_steps += 1

            if stuck_steps >= STUCK_THRESHOLD:
                print(f"Episode {episode} skipped due to being stuck.")
                continue

        if game.score > best_score:
            best_score = game.score
            torch.save(agent.q_network.state_dict(), best_model_path)
            print(f"New best score: {best_score}. Model saved to {best_model_path}")
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if episode % MODEL_SAVE_INTERVAL == 0 and episode != 0:
            model_path = f"snake_dqn_model_{episode}.pth"
            torch.save(agent.q_network.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        if episode % 1000 == 0:
            if loss is not None:
                print(f"Episode {episode}, Loss: {loss:.4f}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
            else:
                print(f"Episode {episode}, Loss: Not computed due to insufficient experience")

        if stagnation_counter >= STAGNATION_THRESHOLD:
            print(f"Training has stagnated for {STAGNATION_THRESHOLD} episodes. Learning rate is halved.")
            # LEARNING_RATE = LEARNING_RATE/2.0
            # for param_group in agent.optimizer.param_groups:
            #     param_group['lr'] = LEARNING_RATE
            stagnation_counter = 0

        epsilon = decay_epsilon(epsilon, episode)

        if episode % 10 == 0:
            agent.update_target_network()

    pygame.quit()
    print(f"Training complete. Best score (number of food eaten): {best_score}")

if __name__ == "__main__":
    train_snake_game(render=False)