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
        eps: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_states: number of discrete states (default = 3)
            n_actions: number of actions per state (default = 2)
            alpha: learning rate
            gamma: discount factor
            eps: epsilon for epsilon-greedy exploration
            seed: random seed for reproducibility
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.rng = np.random.default_rng(seed)

        # Q-table: shape [n_states, n_actions]
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: current state (0, 1, or 2)

        Returns:
            action: 0 or 1
        """
        # exploration
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions))

        # exploitation
        q_values = self.Q[state]
        return int(np.argmax(q_values))

    def update(self, s: int, a: int, r: float, s_next: Optional[int], done: bool):
        """
        Standard one-step Q-learning update.

        Args:
            s: current state
            a: action taken at s
            r: reward received
            s_next: next state (None if terminal)
            done: whether episode ended after this transition
        """
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
