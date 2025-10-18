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
            nn.Linear(state_size, 128),  # input layer -> hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),          # hidden layer -> hidden
            nn.ReLU(),
            nn.Linear(128, 64),           # FIX: forgot 2nd hidden layer
            nn.ReLU(),
            nn.Linear(64, action_size)    # output layer -> best Q value per action
        )

    def forward(self, x):
        return self.model(x)    # return predicted Q values

# define agent
class DQNAgent:
        def __init__(self, state_size, action_size, alpha=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
            self.action_size = action_size
            self.memory= [] # replay buffer
            self.gamma = gamma # discount factor
            self.epsilon = epsilon # exploration rate
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min 
            self.target_update_frequency = 10  # Update target network every 10 episodes

            # init neural network + optimizer + loss function
            self.model = DQN(state_size, action_size)
            self.target_model = DQN(state_size, action_size)  # FIX: new target network to calculate TD
            self.target_model.load_state_dict(self.model.state_dict())  # copy weights from neural network
            # adam function adjusts learning rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=alpha) 
            # measure loss
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
                
                state = torch.FloatTensor(state).unsqueeze(0) # add batch dimension, expected in NN

                q_values = self.model(state)    # return Q value
                return torch.argmax(q_values).item()   # return index of best action. .item() converts tensor to int
            
        # store memory
        def remember(self, transition):
            self.memory.append(transition)

            if len(self.memory) > 10000: # FIX: extra pre-training experiences
                self.memory.pop(0)
        
        # sample a batch and train network through forward and back propagation 
        def replay(self, batch_size=64):
            """
            Train the NN on random batch of experiences

            1. Sample random from buffer
            2. Compute Q prediction from main network
            3. Compute Q target from target network
            4. Compute loss as MSE
            5. Backpropagate to main network

            Args:
                batch_size: number of experiences to sample and train on
            """
            if len(self.memory) < batch_size:
                return

            batch = random.sample(self.memory, batch_size)

            # batch is list of tuples: [(s,a,r,s',d), (s,a,r,s',d), ...]
            # zip(*batch) transposes to: [(s,s,...), (a,a,...), (r,r,...), ...]
            states, actions, rewards, next_states, dones = zip(*batch)

           
            # FIX: Stack numpy arrays first for efficiency then convert to tensors
            # Shape: [batch_size, state_size]
            states = torch.FloatTensor(np.array(states))
            next_states = torch.FloatTensor(np.array(next_states))

            # Actions are integers (indices), use LongTensor
            # Shape: [batch_size]
            actions = torch.LongTensor(np.array(actions))

            # Rewards and dones are floats
            # Shape: [batch_size]
            rewards = torch.FloatTensor(np.array(rewards))
            dones = torch.FloatTensor(np.array(dones))  # 1.0 if done, 0.0 otherwise

            # ====================================================================
            # STEP 1: COMPUTE Q-PREDICTIONS (what we currently predict)
            # ====================================================================
            # Forward pass through main network
            # Returns Q-values for ALL actions: [batch_size, action_size]
            all_q_values = self.model(states)
            
            # Extract Q-values for actions that were actually taken
            # gather(1, actions.unsqueeze(1)) selects one Q-value per row
            # Result shape: [batch_size, 1] -> squeeze to [batch_size]
            q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # ====================================================================
            # STEP 2: COMPUTE Q-TARGETS (what we want to predict)
            # TD target: r + γ * max Q(s', a') if not done, else just r
            # ====================================================================
            with torch.no_grad():   # for target NN specifically
                # forward pass
                next_q_values_all = self.target_model(next_states)
                
                # Take maximum Q-value for each next state
                # .max(1) returns (values, indices), we only need values
                # Shape: [batch_size]
                next_q_values = next_q_values_all.max(1)[0]

                # Bellman equation: Q(s,a) = r + γ * max Q(s',a')
                # If episode done, future value is 0: multiply by (1 - done)
                # (1 - dones) zeros out next_q_values for terminal states
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            # ====================================================================
            # STEP 3: COMPUTE LOSS
            # MSE = mean((prediction - target)²)
            # ====================================================================
            loss = self.loss_fn(q_values, targets)

            # ====================================================================
            # STEP 4: BACKPROPAGATION (update network weights)
            # ====================================================================
            # Zero out gradients from previous step
            # PyTorch accumulates gradients by default, so we must clear them
            self.optimizer.zero_grad()

            # Compute gradients: ∂Loss/∂(each weight) via chain rule
            # PyTorch automatically builds computation graph and applies backprop
            loss.backward()

            # Update weights: w_new = w_old - learning_rate * gradient
            # Adam optimizer handles the actual update with momentum & adaptive rates
            self.optimizer.step()


            # ====================================================================
            # STEP 5: adjust epsilon
            # ====================================================================
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        def update_target_network(self):
            """
            Synchronize target network with main network
            
            Copy all weights from main network to target network.
            This creates a stable target for TD learning.
            Call this every N episodes (e.g., 10).
            """
            # .state_dict() returns OrderedDict of all parameters
            # .load_state_dict() copies parameters to target network
            # This is a deep copy
            self.target_model.load_state_dict(self.model.state_dict())
