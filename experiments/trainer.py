"""
Trainer for the two-step decision task.

"""
from typing import Optional
import numpy as np

from env.two_step_env import TwoStepEnv
from agents.model_free import MFQAgent
from agents.mb_agent import MBAgent
from agents.hybrid_agent import HybridAgent


class Trainer:
    """
    Generic episodic RL trainer.

    Assumes:
    - env.reset() -> initial_state
    - env.step(action) -> (next_state, reward, done, info_dict)
    - agent.select_action(state) -> action (int)
    - agent.update(state, action, reward, next_state, done [, info]) exists

    If your agent.update does NOT take `info`, this trainer will
    automatically fall back to calling it without info.
    """

    def __init__(self, env, agent, n_episodes=1000):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def run(self, verbose=False):
        """
        Run training for n_episodes.

        Returns:
            rewards: np.ndarray of shape (n_episodes,)
                     total reward per episode.
        """
        rewards = []

        for ep in range(self.n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # 1) choose action from current state
                action = self.agent.select_action(state)

                # 2) step environment
                next_state, reward, done, info = self.env.step(action)

                # 3) update agent
                #    (support both update(..., info) and update(...) signatures)
                try:
                    self.agent.update(state, action, reward, next_state, done, info)
                except TypeError:
                    # if the agent's update doesn't take `info`
                    self.agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

            rewards.append(total_reward)

            if verbose and ((ep + 1) % 100 == 0 or ep == 0):
                print(f"Episode {ep+1}/{self.n_episodes}, total reward = {total_reward:.3f}")

        return np.array(rewards)


def make_agent(agent_type: str, **agent_kwargs):
    """
    Factory to construct MF / MB / Hybrid agents from a string.

    agent_type:
        "mf"      -> MFQAgent
        "mb"      -> MBAgent
        "hybrid"  -> HybridAgent (wraps MFQAgent + MBAgent)

    agent_kwargs are passed through to the underlying constructors.
    """
    agent_type = agent_type.lower()

    if agent_type == "mf":
        return MFQAgent(**agent_kwargs)

    if agent_type == "mb":
        return MBAgent(**agent_kwargs)

    if agent_type == "hybrid":
        # for hybrid, we build internal MF and MB agents
        # you can customize their hyperparameters here if needed
        mf_kwargs = agent_kwargs.get("mf_kwargs", {})
        mb_kwargs = agent_kwargs.get("mb_kwargs", {})
        w = agent_kwargs.get("w", 0.5)

        mf_agent = MFQAgent(**mf_kwargs)
        mb_agent = MBAgent(**mb_kwargs)
        return HybridAgent(mf_agent, mb_agent, w=w)

    raise ValueError(f"Unknown agent_type: {agent_type}. Use 'mf', 'mb', or 'hybrid'.")


def run_training(
    agent_type: str = "mf",
    n_episodes: int = 1000,
    env_kwargs: dict | None = None,
    agent_kwargs: dict | None = None,
    verbose: bool = True,
):
    """
    Convenience function to:
        1) build env
        2) build agent
        3) train agent
        4) return per-episode rewards

    env_kwargs: dict passed to TwoStepEnv(...)
    agent_kwargs: dict passed to make_agent(...)

    Example:
        rewards = run_training(
            agent_type="mf",
            n_episodes=5000,
            env_kwargs={"drift_sigma": 0.02},
            agent_kwargs={"alpha": 0.1, "epsilon": 0.1}
        )
    """
    if env_kwargs is None:
        env_kwargs = {}
    if agent_kwargs is None:
        agent_kwargs = {}

    # 1) build environment
    env = TwoStepEnv(**env_kwargs)

    # 2) build agent
    agent = make_agent(agent_type, **agent_kwargs)

    # 3) train
    trainer = Trainer(env, agent, n_episodes=n_episodes)
    rewards = trainer.run(verbose=verbose)

    return rewards

