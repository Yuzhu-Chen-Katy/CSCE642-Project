import numpy as np

class MBAgent:
    def __init__(self, alpha=0.1, beta=5.0, epsilon=0.1):
        self.alpha = alpha          
        self.beta = beta            
        self.epsilon = epsilon

        self.T = np.ones((2, 2)) * 0.5 
        self.R = np.ones((2, 2)) * 0.5

    def softmax(self, q):
        e = np.exp(self.beta * (q - np.max(q)))
        return e / e.sum()

    def select_action(self, state):
        q = self.Q_values(state)
        probs = self.softmax(q)  
        return np.random.choice([0,1], p=probs)


    def Q_values(self, state):
        if state == 0:
            qA = self.T[0,0] * np.max(self.R[0]) + self.T[0,1] * np.max(self.R[1])
            qB = self.T[1,0] * np.max(self.R[0]) + self.T[1,1] * np.max(self.R[1])
            return np.array([qA, qB])

        return self.R[state-1]

    def update_transition(self, action, next_state):
        s_idx = next_state - 1
        target = np.array([0,0])
        target[s_idx] = 1
        self.T[action] += self.alpha * (target - self.T[action])

    def update_reward(self, state, action, reward):
        s_idx = state - 1
        self.R[s_idx][action] += self.alpha * (reward - self.R[s_idx][action])
    
    def update(self, state, action, reward, next_state, done, info=None):
        if state == 0 and next_state in (1, 2):
            self.update_transition(action, next_state)

        if state in (1, 2) and done:
            self.update_reward(state, action, reward)

