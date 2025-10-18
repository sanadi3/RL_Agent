import numpy as np
import gymnasium as gym

class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.env = env
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min 

        # FrozenLake has discrete states (0-15) and 4 actions
        num_states = env.observation_space.n  # 16 states (4x4 grid)
        num_actions = env.action_space.n     # 4 actions (left, down, right, up)
        self.Q = np.zeros((num_states, num_actions))

    def get_state_index(self, state):
        # FrozenLake states are already discrete (0-15)
        return int(state)
    
    def choose_action(self, state):
        """epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(4)  # FrozenLake has 4 actions
        else:
            # Best action (exploitation)
            s = self.get_state_index(state)

            # FIX: np.argmax() returns first index with ties
            q_values = self.Q[s]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
        
    def update(self, state, action, reward, next_state):
        """update Q-value for a given(state, action) pair"""

        # tracking
        s = self.get_state_index(state)
        s_next = self.get_state_index(next_state)

        # greedy choice
        q_values = self.Q[s_next]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        best_next_action = np.random.choice(best_actions)

        # compute TD target (estimation of next Q-value)
        # immediate reward + discounted future value of the bext next action
        td_target = reward + self.gamma * self.Q[s_next, best_next_action]

        # compute TD error 
        # TD target - the last estimation of the Q value
        td_error = td_target - self.Q[s, action]

        # new Q value estimation = former + alpha * error
        self.Q[s, action] += self.alpha * td_error
