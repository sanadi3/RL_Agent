# Reinforcement Learning: Q-Learning vs Deep Q-Learning

A comparative study of tabular Q-Learning and Deep Q-Networks (DQN) on the FrozenLake environment, implemented with PyTorch and visualized using Streamlit.

## 🎯 Project Overview

This project demonstrates the fundamental differences between **tabular Q-Learning** and **Deep Q-Learning (DQN)** by training agents to solve the FrozenLake environment. The interactive Streamlit dashboard allows you to compare both approaches side-by-side with real-time visualization.

## 🧠 Understanding Q-Learning vs Deep Q-Learning

### Tabular Q-Learning
- **What it is**: A model-free reinforcement learning algorithm that learns the value of state-action pairs in a lookup table
- **How it works**: 
  - Maintains a Q-table: `Q(state, action) = expected future reward`
  - Updates Q-values using the Bellman equation: `Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]`
  - Uses ε-greedy policy for exploration vs exploitation
- **Pros**: Simple, interpretable, guaranteed convergence
- **Cons**: Doesn't scale to large state spaces, requires discrete states (wouldnt work with the CartPole environment)

### Deep Q-Learning (DQN)
- **What it is**: Uses a neural network to approximate Q-values instead of a lookup table
- **How it works**:
  - Neural network takes state as input, outputs Q-values for all actions
  - Uses experience replay to break correlation between consecutive samples
  - Employs target network for stable training
  - Updates network weights using gradient descent
- **Pros**: Scales to continuous/high-dimensional state spaces, can handle complex environments
- **Cons**: More complex, requires tuning, can be unstable

### Key Differences

| Aspect | Tabular Q-Learning | Deep Q-Learning |
|--------|-------------------|-----------------|
| **State Representation** | Discrete states (0-15) | One-hot encoded vectors |
| **Q-Value Storage** | Lookup table (16×4) | Neural network approximation |
| **Memory** | O(states × actions) | O(neural network parameters) |
| **Scalability** | Limited to small state spaces | Scales to large/continuous spaces |
| **Training** | Direct Q-table updates | Gradient descent on neural network |
| **Stability** | Guaranteed convergence | Can be unstable, requires careful tuning |

## 🏗️ Architecture

### Tabular Q-Learning Agent
```python
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.Q = np.zeros((num_states, num_actions))  # Q-table
        # ... other parameters
```

### Deep Q-Network Architecture
```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),    # Input layer
            nn.ReLU(),
            nn.Linear(128, 128),           # Hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64),            # Hidden layer 2
            nn.ReLU(),
            nn.Linear(64, action_size)     # Output layer
        )
```

## 🚀 Getting Started

### Prerequisites
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install gymnasium streamlit matplotlib numpy
```

### Running the Application
```bash
# Start the Streamlit dashboard
streamlit run main.py
```

## 🎮 How to Use

### 1. **Training Parameters**
- **Training Episodes**: Number of episodes to train (100-50,000)
- **Max Steps**: Maximum steps per episode (10-500)
- **Learning Rate (α)**: How fast the agent learns (0.01-1.0)
- **Discount Factor (γ)**: Importance of future rewards (0.1-0.99)
- **Epsilon**: Exploration rate (0.01-1.0)
- **Epsilon Decay**: How quickly exploration decreases (0.9-0.9999)

### 2. **Algorithm Selection**
Choose between:
- **Q-Learning (tabular)**: Traditional lookup table approach
- **Deep Q-Learning (neural network)**: Neural network approximation

### 3. **DQN-Specific Parameters**
- **Replay Batch Size**: Number of experiences sampled for training (16-256)
- **Warmup Period**: Minimum experiences before training starts (0-5,000)

### 4. **Training Process**
1. Click "Train Agent" to start training
2. Watch real-time progress with reward charts
3. Monitor epsilon decay and learning progress
4. Training completes when all episodes are finished

### 5. **Watching the Agent**
1. Click "Watch Trained Agent" after training
2. See the agent navigate the environment in real-time
3. View Q-values and action explanations (for tabular Q-learning)
4. Observe success/failure outcomes

## 🔧 PyTorch Implementation Details

### Neural Network Setup
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the network (4 layers)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.model(x)
```

### Training Loop
```python
# Experience Replay
def replay(self, batch_size=64):
    if len(self.memory) < batch_size:
        return
    
    # Sample random batch
    batch = random.sample(self.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors
    states = torch.FloatTensor(np.array(states))
    next_states = torch.FloatTensor(np.array(next_states))
    actions = torch.LongTensor(np.array(actions))
    rewards = torch.FloatTensor(np.array(rewards))
    dones = torch.FloatTensor(np.array(dones))
    
    # Current Q-values
    q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q-values using target network
    with torch.no_grad():
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
    
    # Compute loss and update
    loss = self.loss_fn(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Key PyTorch Concepts Used
- **Tensors**: Multi-dimensional arrays for neural network computations
- **Autograd**: Automatic differentiation for gradient computation
- **Optimizers**: Adam optimizer for weight updates
- **Loss Functions**: MSE loss for Q-value approximation
- **Target Networks**: Separate network for stable training targets

## 📊 Expected Results

### Tabular Q-Learning
- **Convergence**: Usually converges within 1,000-5,000 episodes
- **Performance**: Can achieve 100% success rate on non-slippery FrozenLake
- **Interpretability**: Q-table shows learned values for each state-action pair

### Deep Q-Learning
- **Convergence**: May take 5,000-20,000 episodes depending on hyperparameters
- **Performance**: Can achieve high success rates but may be less stable
- **Scalability**: Can handle much larger state spaces than tabular methods

## 🐛 Common Issues & Solutions

### DQN Not Learning
- **Problem**: Learning rate too high (try 0.001)
- **Problem**: Not enough exploration (increase epsilon)
- **Problem**: Target network not updating (check update frequency)

### Poor Performance
- **Problem**: Reward shaping too harsh (adjust reward values)
- **Problem**: Network too small (increase hidden layers)
- **Problem**: Not enough training episodes

### Training Instability
- **Solution**: Use target network (implemented)
- **Solution**: Experience replay (implemented)
- **Solution**: Proper hyperparameter tuning

## 📁 Project Structure

```
RL_Agent/
├── main.py              # Streamlit dashboard and main logic
├── agent.py             # Tabular Q-Learning implementation
├── dqn.py               # Deep Q-Network implementation
├── environment.py       # Custom GridWorld environment
├── README.md            # This file
└── venv/                # Virtual environment
```


## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this educational project!

---

