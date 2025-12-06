import numpy as np

class HybridAgent:
    def __init__(self, mf_agent, mb_agent, w=0.5):
        self.mf = mf_agent
        self.mb = mb_agent
        self.w = w

    def select_action(self, state):
        Q_mf = self.mf.Q_values(state)
        Q_mb = self.mb.Q_values(state)
        Q_combo = self.w * Q_mb + (1 - self.w) * Q_mf
        return np.argmax(Q_combo)

    def update(self, state, action, reward, next_state, done):
        self.mf.update(state, action, reward, next_state, done)
        if state == 0:
            self.mb.update_transition(action, next_state)
        if next_state in [1,2] and done:
            self.mb.update_reward(next_state, action, reward)
