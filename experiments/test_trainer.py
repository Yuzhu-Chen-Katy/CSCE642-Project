from experiments.trainer import run_training

rewards_mf = run_training(
    agent_type="mf",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},  # adjust key if needed
    agent_kwargs={"alpha": 0.1, "eps": 0.1},
    verbose=True,
)

print("MF average reward:", rewards_mf.mean())

rewards_mb = run_training(
    agent_type="mb",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},
    agent_kwargs={"alpha": 0.1},
)
print("MB average reward:", rewards_mb.mean())

rewards_hybrid = run_training(
    agent_type="hybrid",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},
    agent_kwargs={"w": 0.5},
)
print("Hybrid average reward:", rewards_hybrid.mean())
