import numpy as np
import matplotlib.pyplot as plt

plt.ion()

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0,0)
        self.goal = (size - 1, size - 1)    # end of grid
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        # numpy arrays, x represents row, y represents column
        if action == 0: # up
            x = max(x - 1, 0) # to take care of borders
        if action == 1: # down
            x = min(x + 1, self.size - 1)   # to not go past border
        if action == 2: # left
            y = max( y - 1, 0)
        if action == 3: # right
            y = min(y+1, self.size - 1)
        
        # FIX: Update agent position
        self.agent_pos = (x, y)
        
        reward = 1 if self.agent_pos == self.goal else -0.01 # reward of 1 reached goal
        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done
    
    def render(self):
        grid = np.zeros((self.size, self.size)) # create grid
        x, y = self.agent_pos
        gx, gy = self.goal
        grid[gx, gy] = 0.5  # Goal position
        grid[x, y] = 1.0    # Agent position
        plt.clf()   # clear previous frame
        plt.imshow(grid, cmap="coolwarm", origin="upper", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"GridWorld - Agent at ({x},{y}), Goal at ({gx},{gy})")
        plt.xlabel("Y (Column)")
        plt.ylabel("X (Row)")
        plt.draw()
        plt.pause(0.1)


        