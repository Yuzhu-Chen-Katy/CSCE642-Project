from typing import Optional
import numpy as np

from env.two_step_env import TwoStepEnv
from agents.model_free import MFQAgent
from agents.mb_agent import MBAgent
from agents.hybrid_agent import HybridAgent


class Trainer:

    def __init__(self, env, agent, n_episodes=1000):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def run(self, verbose=False, log_behavior=False):

        rewards = []
        episode_log = []

        for ep in range(self.n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            
            first_stage_action = None
            transition_type = None

            while not done:
                
                action = self.agent.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                if state == 0 and first_stage_action is None:
                    first_stage_action = action
                    transition_type = info.get(
                        "transition_type", 
                        getattr(self.env, "last_transition_type", None),
                    )

                try:
                    self.agent.update(state, action, reward, next_state, done, info)
                except TypeError:
                    self.agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

            rewards.append(total_reward)

            if log_behavior:
                episode_log.append(
                    {
                        "episode": ep,
                        "first_stage_action": int(first_stage_action)
                        if first_stage_action is not None
                        else None,
                        "transition_type": transition_type,
                        "reward": float(total_reward),
                    }
                )

            if verbose and ((ep + 1) % 100 == 0 or ep == 0):
                print(f"Episode {ep+1}/{self.n_episodes}, total reward = {total_reward:.3f}")

        if log_behavior:
            return np.array(rewards), episode_log
        else:
            return np.array(rewards)


def make_agent(agent_type: str, **agent_kwargs):
    agent_type = agent_type.lower()

    if agent_type == "mf":
        return MFQAgent(**agent_kwargs)

    if agent_type == "mb":
        return MBAgent(**agent_kwargs)

    if agent_type == "hybrid":

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
    log_behavior: bool = False,
):
    if env_kwargs is None:
        env_kwargs = {}
    if agent_kwargs is None:
        agent_kwargs = {}

    env = TwoStepEnv(**env_kwargs)

    agent = make_agent(agent_type, **agent_kwargs)

    trainer = Trainer(env, agent, n_episodes=n_episodes)
    results = trainer.run(verbose=verbose, log_behavior=log_behavior)
    return results

