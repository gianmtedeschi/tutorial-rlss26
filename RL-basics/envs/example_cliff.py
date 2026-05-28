import numpy as np
from gymnasium import Env

UP    = 0
RIGHT = 1
DOWN  = 2
LEFT  = 3

class CliffWalkingEnv(Env):
    ROWS = 4
    COLS = 12
    CLIFF = {(3, c) for c in range(1, 11)}
    GOAL = (3, 11)

    def reset(self, *, seed=None, options=None):
        valid_states = [
            (r, c)
            for r in range(self.ROWS)
            for c in range(self.COLS)
            if (r, c) not in self.CLIFF and (r, c) != self.GOAL
        ]
        # Sample uniformly the initial state
        state = valid_states[np.random.randint(len(valid_states))]
        self.state = np.array(state, dtype=np.int32)
        # Count the number of interactions
        self.interactions = 0
        return self.state, {}

    def step(self, action: int):
        # If possible perform the action
        if action == UP and self.state[0] > 0:
            self.state = self.state + np.array([-1, 0], dtype=np.int32)
        elif action == RIGHT and self.state[1] < self.COLS - 1:
            self.state = self.state + np.array([0, 1], dtype=np.int32)
        elif action == DOWN and self.state[0] < self.ROWS - 1:
            self.state = self.state + np.array([1, 0], dtype=np.int32)
        elif action == LEFT and self.state[1] > 0:
            self.state = self.state + np.array([0, -1], dtype=np.int32)

        self.interactions += 1

        terminated = True if tuple(self.state) == self.GOAL else False
        truncated = self.interactions >= 50

        reward = -100 if tuple(self.state) in self.CLIFF else -1

        # When the agent falls off the cliff, terminate the episode
        if reward == -100:
           terminated = True

        return self.state, reward, terminated, truncated, {}
