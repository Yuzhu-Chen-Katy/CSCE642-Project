# Two-Step RL Project (CSCE 642)

This repository contains our implementation of the **Daw Two-Step Task** and three reinforcement learning agents:

- **Model-Free (MFQAgent)** – Q-learning with softmax exploration  
- **Model-Based (MBAgent)** – planning using learned transition and reward models  
- **Hybrid Agent** – weighted combination of MF and MB action values  

The project simulates agent behavior under different reward volatilities and analyzes:
- **Stay probabilities**  
- **Reward × Transition logistic regression**  
- **Learning curves**

All analysis and plots are reproducible with one command.

## Getting Started
The code and data should be structured as follows: 
```
csce642-two-step-rl/
│
├── env/ # Two-step task environment
├── agents/ # MF, MB, Hybrid agent implementations
├── experiments/
│ ├── stimulation.py # Runs all (agent × volatility × seed) simulations
│ └── trainer.py # Training loop used by all agents
├── analysis.py # Stay-prob, regression, learning curve analysis
├── main.py # MASTER script (run simulations + analysis)
│
├── results/ # CSV outputs (automatically generated)
├── figures/ # Plots (automatically generated)
│
├── requirements.txt
└── README.md # This file
```
### Environment Setup

Using Conda (recommended):

```
conda create -n two_step_rl python=3.10
conda activate two_step_rl
pip install -r requirements.txt
```

### Running the Full Pipeline
```bash
python main.py
```
**1. Run all agents across volatility levels**

- Model-Free (mf)

- Model-Based (mb)

- Hybrid (hybrid)

- Volatility levels: [0.015, 0.025, 0.04]

- Seeds: configurable in utils/config.py

**2. Perform behavioral analysis**

analysis.py loads the results and computes:

- Stay probabilities (common vs rare, reward vs no reward)
- Logistic regression predicting stay from
- previous reward, previous transition, and their interaction
- Learning curves (mean reward per episode)

It also saves:
```bash
results/regression_data.csv
results/stay_summary.csv
results/logistic_coefs.csv
results/learning_curves.csv
```
All figures are saved to:
```
figures/
    stayprob_mf.png
    stayprob_mb.png
    stayprob_hybrid.png
    interaction_vs_volatility.png
    learningcurves_mf.png
    learningcurves_mb.png
    learningcurves_hybrid.png
```
**3. hyperparameter tuning**
To reproduce hyperparameter tuning results, run:
```bash
python -m experiments.hparam_search
```
