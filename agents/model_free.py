from typing import Optional
import numpy as np

class MFQAgent:
    def __init__(
        self,
        n_states: int = 3,
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 1.0,
        eps: float = 0.1,
        seed: Optional[int] = None,
        tau: float = 0.5,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.tau = tau

        if seed is None:
            seed = 19
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state: int) -> int:
        q = self.Q[state]
        x = q / self.tau
        e = np.exp(x - np.max(x))
        probs = e / e.sum()
        return int(self.rng.choice(self.n_actions, p=probs))

    def Q_values(self, state: int):
        return self.Q[state]

    def update(self, s: int, a: int, r: float, s_next: Optional[int], done: bool):
        q_sa = self.Q[s, a]

        if done or s_next is None:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])

        self.Q[s, a] = q_sa + self.alpha * (target - q_sa)

if __name__ == "__main__":
    agent = MFQAgent()
    print("Initial Q-table:\n", agent.Q)
    agent.update(s=0, a=1, r=1.0, s_next=1, done=False)
    print("\nUpdated Q-table after one fake step:\n", agent.Q)
