import numpy as np

class MBAgent:
    def __init__(self, alpha=0.1, beta=5.0, epsilon=0.1):
        self.alpha = alpha          # reward learning rate
        self.beta = beta            # softmax inverse temp
        self.epsilon = epsilon

        # transition model: P(next_state | action)
        # initialize uniform = 0.5/0.5
        self.T = np.ones((2, 2)) * 0.5   # actions: A,B; next states: s1,s2

        # reward model for stage-2:
        # R_hat[state][action]
        self.R = np.ones((2, 2)) * 0.5

    def softmax(self, q):
        e = np.exp(self.beta * (q - np.max(q)))
        return e / e.sum()

    def select_action(self, state):
        q = self.Q_values(state)
        probs = self.softmax(q)  # already implemented
        return np.random.choice([0,1], p=probs)


    def Q_values(self, state):
        # stage-1 planning
        if state == 0:
            qA = self.T[0,0] * np.max(self.R[0]) + self.T[0,1] * np.max(self.R[1])
            qB = self.T[1,0] * np.max(self.R[0]) + self.T[1,1] * np.max(self.R[1])
            return np.array([qA, qB])
        
        # stage-2 values
        return self.R[state-1]

    def update_transition(self, action, next_state):
        # convert next_state {1,2} to index {0,1}
        s_idx = next_state - 1
        # one-hot update
        target = np.array([0,0])
        target[s_idx] = 1
        self.T[action] += self.alpha * (target - self.T[action])

    def update_reward(self, state, action, reward):
        s_idx = state - 1
        self.R[s_idx][action] += self.alpha * (reward - self.R[s_idx][action])
    
    def update(self, state, action, reward, next_state, done, info=None):
        """
        Unified update used by the Trainer.

        - If we are at stage 1 (state == 0) and move to stage 2 (1 or 2),
          we update the transition model.
        - If we are at stage 2 (state == 1 or 2) and the episode ends (done=True),
          we update the reward model.
        """
        # Stage 1 -> Stage 2: update transition model
        if state == 0 and next_state in (1, 2):
            self.update_transition(action, next_state)

        # Stage 2 -> terminal with reward: update reward model
        if state in (1, 2) and done:
            # state is 1 or 2 here; action is Left/Right (0/1)
            self.update_reward(state, action, reward)

