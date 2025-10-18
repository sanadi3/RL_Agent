import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
import gymnasium as gym

# define neural network to pass state to.
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),  # input layer -> hidden layer
            nn.ReLU(),
            nn.Linear(64, 64),          # hidden layer -> hidden
            nn.ReLU(),
            nn.Linear(64, action_size)  # output layer -> best Q value per action
        )

    def forward(self, x):
        return self.model(x)    # return predicted Q values

# define agent
class DQNAgent:
        def __init__(self, state_size, action_size, alpha=0.5, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
            self.action_size = action_size
            self.memory= [] # replay buffer
            self.gamma = gamma # discount factor
            self.epsilon = epsilon # exploration rate
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min 

            # init neural network + optimizer + loss function
            self.model = DQN(state_size, action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
            self.loss_fn = nn.MSELoss()
        
        # epsilon - greedy
        def choose_action(self, state, training=True):
            """
            Choose action using epsilon greedy poliscy
            Args:
                state: current state (numpy array or tensor)
                training: if True, use epsilon-greedy; if False, always exploiut
            """
            if training and np.random.rand() < self.epsilon:
                return np.random.randint(self.action_size)
             
            # else exploit
            with torch.no_grad():   # no gradient computation needed
                
                state = torch.FloatTensor(state).unsqueeze(0) # add domensions
                q_values = self.model(state)
                return torch.argmax(q_values).item()   # return index of best action
            
        # store memory
        def remember(self, transition):
            self.memory.append(transition)

            if len(self.memory) > 10000:
                self.memory.pop(0)
        
        # sample a batch and train network
        def replay(self, batch_size=64):
            if len(self.memory) < batch_size:
                return

            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors (stack numpy arrays first for efficiency)
            states = torch.FloatTensor(np.array(states))
            next_states = torch.FloatTensor(np.array(next_states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            dones = torch.FloatTensor(np.array(dones))

            # compute Q values
            # gather method picks the Q value for each action
            q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # compute best target Q value. bellman equation:
            # target = reward + gamma * argmax(Q(next_state)) if not done
            next_q_values = self.model(next_states).max(1)[0].detach()
            targets = rewards + self.gamma*next_q_values * (1-dones)

            # compute loss between current and target Q-values
            loss = self.loss_fn(q_values, targets)

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #reduce epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
