from env.two_step_env import TwoStepEnv
from agents.model_free import MFQAgent

def run_one_episode(env, agent):
    s1 = env.reset()
    # stage 1
    a1 = agent.select_action(s1)
    s2, r1, done, info1 = env.step(a1)
    assert not done

    # stage 2
    a2 = agent.select_action(s2)
    s3, r2, done, info2 = env.step(a2)
    assert done

    # MF updates (backwards):
    # update second-stage step first
    agent.update(s=s2, a=a2, r=r2, s_next=None, done=True)
    # then first-stage step, with "reward" = 0 but bootstrapping from state 2
    agent.update(s=s1, a=a1, r=0.0, s_next=s2, done=False)

    return r2, info1

if __name__ == "__main__":
    env = TwoStepEnv(seed=0)
    agent = MFQAgent()

    for ep in range(5):
        r, info = run_one_episode(env, agent)
        print(f"Episode {ep}: reward={r}, transition={info['transition']}")
    print("\nQ-table after 5 episodes:\n", agent.Q)
