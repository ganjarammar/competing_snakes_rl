import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# Set device to CPU
device = torch.device('cpu')

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 256, 3)  # 11 input states, 256 hidden size, 3 actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 1000
        self.min_memory_size = 1000  # Minimum memory size before training

    def get_state(self, game, snake_num):
        state = game._get_state_for_snake(snake_num)
        return torch.FloatTensor(state).to(device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)

        # Predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        
        # Q_new = reward + gamma * max(next_predicted Q value)
        if not done:
            next_pred = self.model(next_state)
            target = reward + self.gamma * torch.max(next_pred)
        else:
            target = reward

        # Update Q value for taken action
        pred[torch.argmax(action).item()] = target
        
        # Calculate loss and update weights
        self.optimizer.zero_grad()
        loss = F.mse_loss(pred.unsqueeze(0), target.unsqueeze(0))
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train with a single step of experience"""
        self.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """Train with a batch of experiences from memory"""
        if len(self.memory) < self.min_memory_size:
            return  # Don't train if we don't have enough memories
            
        batch_size = min(self.batch_size, len(self.memory))
        mini_sample = random.sample(self.memory, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in mini_sample:
            # Convert to list if it's a numpy array
            if hasattr(state, 'tolist'):
                state = state.tolist()
            if hasattr(action, 'tolist'):
                action = action.tolist()
            if hasattr(next_state, 'tolist'):
                next_state = next_state.tolist()
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        # Convert directly to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Get all the Q values for current states
        current_q_values = self.model(states)
        
        # Get Q values for next states
        next_q_values = self.model(next_states)
        max_next_q = torch.max(next_q_values, dim=1)[0]
        
        # Calculate target Q values
        target_q_values = current_q_values.clone()
        for idx in range(batch_size):
            action_idx = torch.argmax(actions[idx]).item()
            if dones[idx]:
                target_q_values[idx, action_idx] = rewards[idx]
            else:
                target_q_values[idx, action_idx] = rewards[idx] + self.gamma * max_next_q[idx]
        
        # Update network
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, train=True):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if train and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.FloatTensor(state).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
