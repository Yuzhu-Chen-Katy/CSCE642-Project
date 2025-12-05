from typing import Optional
import numpy as np



class TwoStepEnv:
    """
    Two-step decision task environment (Daw et al. style).

    States:
        0 : first-stage state (choose between A / B)
        1 : second-stage state S1 (choose Left / Right)
        2 : second-stage state S2 (choose Left / Right)

    Actions:
        At state 0: 0 = A, 1 = B
        At states 1, 2: 0 = Left, 1 = Right

    Rewards:
        Only at second stage (states 1 or 2), with drifting reward probabilities
        per (state, action) pair.

    Step returns: next_state, reward, done, info
    """

    def __init__(
        self,
        p_common: float = 0.7,
        drift_std: float = 0.025,
        min_p: float = 0.0,
        max_p: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            p_common: probability of a common transition (e.g., A -> S1, B -> S2)
            drift_std: standard deviation of Gaussian random-walk for reward probs
            min_p, max_p: clipping bounds for reward probabilities
            seed: random seed for reproducibility (optional)
        """
        self.p_common = p_common
        self.drift_std = drift_std
        self.min_p = min_p
        self.max_p = max_p

        self.rng = np.random.default_rng(seed)

        # Reward probabilities for second-stage states:
        # shape = (2 states, 2 actions) -> [ [p(S1,Left), p(S1,Right)],
        #                                    [p(S2,Left), p(S2,Right)] ]
        self.rew_probs = np.full((2, 2), 0.5, dtype=float)

        self.state = 0
        self.last_transition_type = None  # "common" or "rare"

    def reset(self) -> int:
        """Start a new episode at the first-stage state (0)."""
        self.state = 0
        self.last_transition_type = None
        return self.state

    def _drift_rewards(self):
        """Apply a Gaussian random walk to reward probabilities, then clip."""
        noise = self.rng.normal(loc=0.0, scale=self.drift_std, size=self.rew_probs.shape)
        self.rew_probs = np.clip(self.rew_probs + noise, self.min_p, self.max_p)

    def step(self, action: int):
        """
        Take a step in the environment.

        Args:
            action: int in {0,1}, interpreted depending on the current state.

        Returns:
            next_state: int or None (None if terminal)
            reward: float (0 or 1)
            done: bool (True if episode ended)
            info: dict with metadata, including 'transition' and 'stage'
        """
        # -------- Stage 1: choose between A / B --------
        if self.state == 0:
            assert action in (0, 1), "At state 0, action must be 0 (A) or 1 (B)."

            # Sample whether this is a common or rare transition
            is_common = self.rng.random() < self.p_common
            self.last_transition_type = "common" if is_common else "rare"

            # Map action + common/rare to second-stage state
            if action == 0:  # A
                next_state = 1 if is_common else 2
            else:            # B
                next_state = 2 if is_common else 1

            self.state = next_state
            info = {
                "stage": 1,
                "transition": self.last_transition_type,
                "first_action": action,
            }
            return next_state, 0.0, False, info

        # -------- Stage 2: choose Left / Right --------
        elif self.state in (1, 2):
            assert action in (0, 1), "At state 1 or 2, action must be 0 (Left) or 1 (Right)."

            # Index into rew_probs: state_index = self.state - 1 (0 or 1)
            state_idx = self.state - 1
            p_reward = self.rew_probs[state_idx, action]
            reward = 1.0 if self.rng.random() < p_reward else 0.0

            # Drift rewards after each trial
            self._drift_rewards()

            # Episode ends after second-stage choice
            self.state = None
            info = {"stage": 2}
            return None, reward, True, info

        else:
            raise ValueError(f"Invalid state {self.state}. Did you forget to reset()?")


if __name__ == "__main__":
    # Quick manual test
    env = TwoStepEnv(seed=42)
    for ep in range(3):
        s = env.reset()
        print(f"\nEpisode {ep}, start state: {s}")
        # first stage
        a1 = 0
        s2, r, done, info = env.step(a1)
        print(f"  Stage1: a={a1}, next_state={s2}, reward={r}, done={done}, info={info}")
        # second stage
        a2 = 0
        s3, r, done, info = env.step(a2)
        print(f"  Stage2: a={a2}, next_state={s3}, reward={r}, done={done}, info={info}")
