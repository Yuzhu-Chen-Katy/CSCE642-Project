from typing import Optional
import numpy as np



class TwoStepEnv:

    def __init__(
        self,
        p_common: float = 0.7,
        drift_std: float = 0.025,
        min_p: float = 0.0,
        max_p: float = 1.0,
        seed: Optional[int] = None,
    ):
        
        self.p_common = p_common
        self.drift_std = drift_std
        self.min_p = min_p
        self.max_p = max_p

        self.rng = np.random.default_rng(seed)
        self.rew_probs = np.full((2, 2), 0.5, dtype=float)

        self.state = 0
        self.last_transition_type = None  

    def reset(self) -> int:
        self.state = 0
        self.last_transition_type = None
        return self.state

    def _drift_rewards(self):
        noise = self.rng.normal(loc=0.0, scale=self.drift_std, size=self.rew_probs.shape)
        self.rew_probs = np.clip(self.rew_probs + noise, self.min_p, self.max_p)

    def step(self, action: int):
        if self.state == 0:
            assert action in (0, 1), "At state 0, action must be 0 (A) or 1 (B)."

   
            is_common = self.rng.random() < self.p_common
            self.last_transition_type = "common" if is_common else "rare"

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

        elif self.state in (1, 2):
            assert action in (0, 1), "At state 1 or 2, action must be 0 (Left) or 1 (Right)."

            state_idx = self.state - 1
            p_reward = self.rew_probs[state_idx, action]
            reward = 1.0 if self.rng.random() < p_reward else 0.0

            self._drift_rewards()

            self.state = None
            info = {"stage": 2}
            return None, reward, True, info

        else:
            raise ValueError(f"Invalid state {self.state}. Did you forget to reset()?")


if __name__ == "__main__":
    env = TwoStepEnv(seed=42)
    for ep in range(3):
        s = env.reset()
        print(f"\nEpisode {ep}, start state: {s}")
        a1 = 0
        s2, r, done, info = env.step(a1)
        print(f"  Stage1: a={a1}, next_state={s2}, reward={r}, done={done}, info={info}")
        a2 = 0
        s3, r, done, info = env.step(a2)
        print(f"  Stage2: a={a2}, next_state={s3}, reward={r}, done={done}, info={info}")
