# Two-Step RL Project (CSCE 642)

This repo contains an implementation of the Daw Two-Step Task with three reinforcement learning agents:

- Model-Free (MFQAgent)
- Model-Based (MBAgent)
- Hybrid Agent (combination of MF + MB)

We provide code for training, evaluation, and behavioral analysis (stay/switch) using the stochastic two-step environment.

---

## Getting Started

### Environment Setup

Using Conda (recommended):

```bash
env_name='csce642_rl'
conda create -n "$env_name" python=3.10
conda activate "$env_name"
pip install -r requirements.txt
```
The code and data should be structured as follows: later
```
```
## Training and Evaluation
### 1. Run all agents (MF / MB / Hybrid)
```
python -m experiments.test_trainer
```
This script will:
- Train MF, MB, and Hybrid agents
- Print average reward
-(Optional) log behavior for stay/switch analysis

### 2. Train MF with behavior logging
```bash
from experiments.trainer import run_training

rewards_mf, mf_log = run_training(
    agent_type="mf",
    n_episodes=1000,
    env_kwargs={"drift_std": 0.02},
    agent_kwargs={"alpha": 0.1, "eps": 0.1},
    verbose=True,
    log_behavior=True
)

print(rewards_mf.mean())
print(mf_log[:5])
```
### 3. Changing Reward Drift (Volatility)
The environment supports a drifting reward probability:
```
env = TwoStepEnv(drift_std=0.02)
```bash
Sweep different values:
```
for sigma in [0.01, 0.02, 0.03]:
    rewards = run_training(
        agent_type="mf",
        n_episodes=2000,
        env_kwargs={"drift_std": sigma},
        verbose=False
    )
    print(sigma, rewards.mean())

```
