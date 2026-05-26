import numpy as np
from gymnasium import Env

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
        # uniformly sample the initial state
        state = valid_states[np.random.randint(len(valid_states))]
        self.state = np.array(state, dtype=np.int32)
        # count the number of interactions
        self.interactions = 0
        return self.state, {}

    def step(self, action: int):
        # if possible perform the action
        if action == 0 and self.state[0] > 0:
            self.state = self.state + np.array([-1, 0], dtype=np.int32)
        elif action == 1 and self.state[1] < self.COLS - 1:
            self.state = self.state + np.array([0, 1], dtype=np.int32)
        elif action == 2 and self.state[0] < self.ROWS - 1:
            self.state = self.state + np.array([1, 0], dtype=np.int32)
        elif action == 3 and self.state[1] > 0:
            self.state = self.state + np.array([0, -1], dtype=np.int32)

        self.interactions += 1

        terminated = tuple(self.state) == self.GOAL
        truncated = self.interactions >= 500

        reward = -100 if tuple(self.state) in self.CLIFF else -1

        # if fall off the cliff resample the state
        if reward == -100:
            valid_states = [
                (r, c)
                for r in range(self.ROWS)
                for c in range(self.COLS)
                if (r, c) not in self.CLIFF and (r, c) != self.GOAL
            ]
            state = valid_states[np.random.randint(len(valid_states))]
            self.state = np.array(state, dtype=np.int32)

        return self.state, reward, terminated, truncated, {}

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        return np.random.randint(0, self.action_space)