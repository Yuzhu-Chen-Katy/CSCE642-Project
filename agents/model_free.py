from typing import Optional
import numpy as np


from typing import Optional
import numpy as np


class MFQAgent:
    """
    Simple model-free Q-learning agent for the two-step task.

    States:
        0 : first-stage (A/B)
        1 : second-stage S1 (Left/Right)
        2 : second-stage S2 (Left/Right)

    Actions:
        At any state: 0 or 1 (interpreted by the environment).
    """

    def __init__(
        self,
        n_states: int = 3,
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 1.0,
        eps: float = 0.1,          # kept for compatibility, not used now
        seed: Optional[int] = None,
        tau: float = 0.5,
    ):
        """
        Args:
            n_states: number of discrete states (default = 3)
            n_actions: number of actions per state (default = 2)
            alpha: learning rate
            gamma: discount factor
            eps: (unused) former epsilon-greedy parameter
            seed: random seed for reproducibility
            tau: softmax temperature
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.tau = tau

        if seed is None:
            seed = 19
        self.rng = np.random.default_rng(seed)

        # Q-table: shape [n_states, n_actions]
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state: int) -> int:
        """Softmax policy over Q-values."""
        q = self.Q[state]
        x = q / self.tau
        e = np.exp(x - np.max(x))
        probs = e / e.sum()
        return int(self.rng.choice(self.n_actions, p=probs))

    def Q_values(self, state: int):
        """Return the Q-values for a given state."""
        return self.Q[state]

    def update(self, s: int, a: int, r: float, s_next: Optional[int], done: bool):
        """Standard one-step Q-learning update."""
        q_sa = self.Q[s, a]

        if done or s_next is None:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])

        self.Q[s, a] = q_sa + self.alpha * (target - q_sa)

if __name__ == "__main__":
    # Tiny smoke test that the agent can act and update.
    agent = MFQAgent()
    print("Initial Q-table:\n", agent.Q)

    # Fake experience: s=0, a=1, r=1, s_next=1, done=False
    agent.update(s=0, a=1, r=1.0, s_next=1, done=False)
    print("\nUpdated Q-table after one fake step:\n", agent.Q)
