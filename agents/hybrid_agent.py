import numpy as np

class HybridAgent:
    def __init__(self, mf_agent, mb_agent, w=0.5, tau=0.5):
        self.mf = mf_agent
        self.mb = mb_agent
        self.w = w
        self.tau = tau

    def softmax(self, q):
        x = q / self.tau
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def select_action(self, state):
        Q_mf = self.mf.Q_values(state)
        Q_mb = self.mb.Q_values(state)
        Q_combo = self.w * Q_mb + (1 - self.w) * Q_mf
        probs = self.softmax(Q_combo)
        return np.random.choice([0, 1], p=probs)

    def update(self, state, action, reward, next_state, done, info=None):
        self.mf.update(state, action, reward, next_state, done)

        self.mb.update(state, action, reward, next_state, done, info)

