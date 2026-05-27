# Comparing DPO and PPO-RLHF for Reinforcement Learning from Preferences

**EE-568 Applied Machine Learning — EPFL 2026**
**Authors:** João Pinto, Pedro Pinto, Tim Arni

---

## Goal

This project compares two offline preference-learning algorithms — **PPO-RLHF** and **Direct Preference Optimisation (DPO)** — on three classical Gymnasium control environments. Both methods fine-tune a mid-performing policy using only a dataset of labelled trajectory-pair preferences (no hand-crafted reward function). The key research question is how sample efficiency, stability and final performance scale with the number of preference pairs K ∈ {50, 200, 1000} across environments that differ in action geometry and reward density.

---

## Environments

![The three Gymnasium environments used in this study](assets/environments.png)

**Figure 1. The three Gymnasium environments used in this study.**

**Table 1. Per-environment setup and expert returns.**

| Environment | Action | Reward | Algorithm | Expert return |
|---|---|---|---|---|
| CartPole-v1 | discrete | dense | PPO | 500 |
| Pendulum-v1 | continuous | dense | PPO | −261 |
| MountainCarContinuous-v0 | continuous | sparse | SAC (gSDE) | 94.3 |

---

## Main Results

**Table 2. True environment return (mean ± std, 5 seeds).** Best result in each cell is bolded. On CartPole, PPO-RLHF wins at K=50, DPO wins at K=200, and both tie at K=1000. DPO outperforms PPO-RLHF on Pendulum at every K. SAC-RLHF dominates DPO on MountainCar at all K, marginally exceeding the expert return (94.3) at every dataset size. Notably, both methods surpass the Pendulum expert.

| | | K=50 | K=200 | K=1000 |
|---|---|---|---|---|
| CartPole | PPO-RLHF | **500.0 ± 0.0** | 495.7 ± 8.6 | **500.0 ± 0.0** |
| | DPO | 460.5 ± 79.0 | **500.0 ± 0.0** | **500.0 ± 0.0** |
| Pendulum | PPO-RLHF | −257.8 ± 24.6 | −252.4 ± 31.6 | −200.3 ± 18.4 |
| | DPO | **−181.7 ± 22.7** | **−151.4 ± 21.0** | **−168.0 ± 16.8** |
| MountainCar | SAC-RLHF | **95.0 ± 1.0** | **95.4 ± 0.4** | **94.7 ± 1.4** |
| | DPO | 92.9 ± 0.6 | 92.8 ± 0.3 | 92.9 ± 0.3 |

---

**Table 3. Computational cost** (wall-clock time in seconds and gradient steps, mean ± std over 5 seeds). Best result per row is bolded.

| | K | Phase 1 T(s) | Phase 1 Steps | Phase 2 T(s) | Phase 2 Steps | PPO-RLHF Total T(s) | PPO-RLHF Total Steps | DPO T(s) | DPO Steps |
|---|---|---|---|---|---|---|---|---|---|
| CartPole | 50 | 0.7±0.5 | 250 | 90.8±4.3 | 8128±384 | 91.6±4.3 | 8378±384 | **39.2±11.0** | **3120±2488** |
| | 200 | 1.7±0.1 | 1000 | 98.0±11.3 | 8576±1364 | 99.7±11.4 | 9576±1364 | **62.6±2.8** | **1750±105** |
| | 1000 | 9.0±0.5 | 5000 | 87.3±6.1 | 7936±314 | **96.3±5.9** | 12936±314 | 222.7±6.9 | **2112±0** |
| Pendulum | 50 | 1.2±0.2 | 500 | 151.6±14.9 | 25856±1803 | 152.8±14.9 | 26356±1803 | **31.8±6.4** | **3200±616** |
| | 200 | 3.9±0.7 | 2000 | 177.2±33.7 | 30656±5729 | 181.1±33.4 | 32656±5729 | **76.2±20.9** | **3100±758** |
| | 1000 | 16.3±0.6 | 10000 | 169.2±39.1 | 30016±6897 | **185.5±39.7** | 40016±6897 | 304.6±77.9 | **3744±887** |
| MountainCar | 50 | 2.6±0.2 | 1250 | 220.6±36.8 | 29990±6194 | 223.2±36.7 | 31240±6194 | **105.2±31.0** | **4350±1405** |
| | 200 | 11.8±3.8 | 5000 | 212.4±26.5 | 27194±4403 | 224.2±28.5 | 32194±4403 | **168.3±69.0** | **3700±1377** |
| | 1000 | 46.6±1.3 | 25000 | 227.5±23.4 | 26989±3994 | **274.1±22.9** | 51989±3994 | 587.0±231.3 | **4832±1953** |

---

## Repository Structure

```
RL-project/
├── data_generation/   # Step 0 — train policies and generate preference datasets
├── rlhf/              # Steps 1–4 — reward model training, PPO/SAC-RLHF fine-tuning, evaluation
└── dpo/               # Alternative — DPO training directly from preference pairs
```

Each module is self-contained and communicates with the others only through shared data artifacts in `data_generation/outputs/`.

---

## Getting Started

Run the three stages in order:

**1. Generate preference data**
```bash
cd data_generation
python train_policies.py
python generate_preferences.py
```
→ See [`data_generation/README.md`](data_generation/README.md) for full details.

**2. Run PPO-RLHF**
```bash
cd rlhf
python run_efficiency_experiment.py
```
→ See [`rlhf/README.md`](rlhf/README.md) for full details.

**3. Run DPO**
```bash
cd dpo
# Open and run dpo_analysis.ipynb
```
→ See [`dpo/README.md`](dpo/README.md) for full details.
