from experiments.trainer import run_training

rewards_mf, mf_log = run_training(
    agent_type="mf",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},
    agent_kwargs={"alpha": 0.1, "eps": 0.1},
    verbose=True,
    log_behavior=True,
)

print("MF reward mean:", rewards_mf.mean())
print("First few log entries:", mf_log[:5])


rewards_mb = run_training(
    agent_type="mb",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},
    agent_kwargs={"alpha": 0.1},
)
#print("MB average reward:", rewards_mb.mean())

rewards_hybrid = run_training(
    agent_type="hybrid",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},
    agent_kwargs={"w": 0.5},
)
#print("Hybrid average reward:", rewards_hybrid.mean())
