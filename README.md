# Reinforcement Learning: Q-Learning vs Deep Q-Learning

A comparative study of tabular Q-Learning and Deep Q-Networks (DQN) on OpenAI's Gymnasium environments. Neural networks approximated with PyTorch and deployed interactivity on Streamlit.

## Understanding Q-Learning vs Deep Q-Learning

### Tabular Q-Learning
- **What it is**: Q-learning learns an action-value function Q(s,a) that estimates the expected cumulative reward for taking action a in state s. The algorithm maintains a lookup table mapping state-action pairs to Q-values.
Update Rule
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                      a'
Converges to the optimal Q*(s, a) 
Where:
  α (alpha): Learning rate controlling update step size
  
  γ (gamma): Discount factor for future rewards (0 < γ ≤ 1)
    if -> 1:
      value all rewards equally
    if -> 0:
      value immediate rewards more
  r: Immediate, current reward
  s': Next state
  TD error: r + γ·max Q(s',a') - Q(s,a) measures prediction error

Exploration-Exploitation
Uses ε-greedy policy:
  With probability ε: select random action (explore)
  With probability 1-ε: select argmax Q(s,a) (exploit)
  ε decays over time: ε ← max(ε × decay, ε_min) after more about the environment is known
  
Limitations
  Only suitable for discrete, small state spaces
  Memory grows linearly with state-action pairs
  Cannot generalize to unseen states

### Deep Q-Learning (DQN)
- **What it is**: DQN replaces the Q-table with a neural network that approximates Q(s,a) for all actions simultaneously.
- **How it works**:
    Architecture:
    Neural Network: state → [128] → [128] → [64] → [action_size]
    
    Input: State vector [ex: 4 values for CartPole]
    Output: Q-value for each possible action 
    Activation: ReLU between layers  [hidden layers to allow non-linearity]

    Problem: DQN combines function approximation + using estimates to update estimates + off policy learning. This leads to instability. DQN solves this with techniques here:

    1. Experience Replay
    Stores transitions (s, a, r, s', done) in a replay buffer [eg size = 10,000]. During training sample these batches randomly. 

    2. Target Network
      Maintains a separate target network with frozen weights for computing TD targets:
      target = r + γ·max Q_target(s', a')
                       a'
      The moving target problem: If we use the same network for both prediction and target, we're chasing a moving target:
      Loss = (Q_θ(s,a) - [r + γ·max Q_θ(s',a')])²
                                a'
      As θ updates, the target shifts, causing oscillations.

    Solution: Freeze target network weights for N episodes, then sync:
      θ_target ← θ_main  (every 10 episodes)
      This creates a stable target during each training phase, proven to improve convergence.

    3. Loss Function
    Mean Squared Error between predicted and target Q-values:
    Loss = (1/N) Σ [Q(s,a) - (r + γ·max Q_target(s',a'))]²
    Minimizing this loss via gradient descent makes predictions closer to Bellman targets.

    Training Process

    1. Act: Agent selects action using ε-greedy on main network Q(s,·)
    2. Observe: Environment returns (s', r, done)
    3. Store: Add transition to replay buffer
    4. Sample: Draw random batch of 64 transitions from buffer
    5. Compute targets: Use frozen target network for stable TD targets
    6. Update: Backpropagate loss to update main network weights
    7. Sync: Periodically copy θ_main → θ_target
    8. Decay: Reduce ε

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

## PyTorch Implementation Details
  
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
nn.Module: Base class providing parameter tracking and device management.
nn.Sequential: Container that chains layers (output of layer i → input of layer i+1).
nn.Linear(in, out): Applies affine transformation y = xW^T + b

Weights W initialized with Kaiming/Xavier initialization.
Bias b initialized to zeros


Why these dimensions? In 128 → 128 → 64, first layers learn general features, later layers learn task-specific features. Total params ≈ 25K for CartPole.

2. Forward Pass
pythondef forward(self, x):
    return self.model(x)
Defines data flow through network. PyTorch automatically calls this via model(input).
The forward pass computes:
x → W1·x + b1 → ReLU → W2·x + b2 → ReLU → ... → Q-values

3. Tensor Operations
pythonstate = torch.FloatTensor(state)     # numpy array → PyTorch tensor
q_values = model(state)               # Forward pass
action = torch.argmax(q_values).item() # Get index of max value
Tensors are PyTorch's multi-dimensional arrays optimized for
Vectorized operations (SIMD),
GPU memory layout (coalesced access),
Automatic gradient tracking.

.unsqueeze(0) adds a batch dimension: [4] → [1, 4] because networks expect batches.

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
.gather() selects Q-values for actions taken: converts [batch, num_actions] → [batch].
Example: If actions = [1, 0, 2] and Q = [[0.1, 0.3], [0.5, 0.2], [0.4, 0.7]], gather returns [0.3, 0.5, 0.7].
python
# Compute targets using frozen target network
with torch.no_grad():
    next_q_values = self.target_model(next_states).max(1)[0]
    targets = rewards + gamma * next_q_values * (1 - dones)
torch.no_grad(): Disables autograd (no gradient calculation). Why?

Target network is frozen (no backprop needed)

(1 - dones) zeros out future value if episode ended (terminal state has no future).

# Backpropagation
loss = self.loss_fn(q_values, targets)
self.optimizer.zero_grad()  # Clear old gradients (they accumulate by default)
loss.backward()              # Compute ∂Loss/∂θ via chain rule
self.optimizer.step()        # Update θ ← θ - lr·∇Loss

Autograd mechanics: PyTorch builds a dynamic computation graph as operations execute. Each tensor remembers its creation operation. When .backward() is called, it traverses the graph in reverse computing gradients via chain rule.

# Model Synchronization
pythonself.target_model.load_state_dict(self.model.state_dict())
.state_dict() returns an OrderedDict of all parameters: {'model.0.weight': tensor(...), 'model.0.bias': tensor(...), ...}
.load_state_dict() copies these parameters to target network. This is a deep copy—modifying one network doesn't affect the other.

### Key PyTorch Concepts Used
- **Tensors**: Multi-dimensional arrays for neural network computations
- **Autograd**: Automatic differentiation for gradient computation
- **Optimizers**: Adam optimizer for weight updates
- **Loss Functions**: MSE loss for Q-value approximation
- **Target Networks**: Separate network for stable training targets

## Getting Started

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

## Expected Results

### Tabular Q-Learning
- **Convergence**: Usually converges within 1,000-5,000 episodes
- **Performance**: Can achieve 100% success rate on non-slippery FrozenLake
- **Interpretability**: Q-table shows learned values for each state-action pair

### Deep Q-Learning
- **Convergence**: May take 5,000-20,000 episodes
- **Performance**: Can achieve high success rates but may be less stable
- **Scalability**: Can handle much larger state spaces than tabular methods

## Common Issues & Solutions

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
## References:
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://medium.com/intro-to-artificial-intelligence/deep-q-network-dqn-in-pytorch-b769f99c02d2
---
