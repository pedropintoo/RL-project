# Direct Preference Optimisation (DPO) for RL

This directory contains the pipeline for the DPO part of the project: adapting the language-model alignment technique Direct Preference Optimisation to fine-tune RL policies directly from offline preference data, without training an explicit reward model.

The pipeline evaluates how DPO performance scales with the number of preference pairs ($K \in \{50, 200, 1000\}$) across a discrete environment (`CartPole-v1`), a continuous dense-reward environment (`Pendulum-v1`), and a continuous sparse-reward environment (`MountainCarContinuous-v0`), and compares the results against the PPO-RLHF baseline.

## Repository Structure

```text
dpo/
├── config.py              # Environment-level configuration (env_id, algo, episode lengths, baselines)
├── policy.py              # Policy wrappers: DiscretePolicy, ContinuousPolicy, SB3 adapters (PPO + SAC)
├── utils.py               # Evaluation helpers, trajectory log-prob computation
├── dpo_experiments.py     # Core training loop (train_dpo) and experiment runner
├── dpo_plotting.py        # Scaling plots and training-curve plots
├── dpo_analysis.ipynb     # Main notebook: runs experiments and generates all plots
└── outputs/
    └── dpo_scaling/
        ├── {env}/K{k}/seed{s}/
        │   ├── checkpoints/dpo_best.pt   # Best checkpoint (by environment return)
        │   └── summary.json              # Per-run metrics and training curve
        ├── plots/                        # Final .png scaling and training-curve graphs
        └── dpo_scaling_all_envs_*.json   # Merged results file (one per experiment run)
```

## Methodology

The experiment follows the same 5-seed independent run strategy used in the RLHF baseline.

* **Starting point:** Every run begins from the Mid-Anchor policy ($\pi_\text{mid}$), ensuring a consistent baseline across all K and seed combinations.
* **Reference model:** A frozen copy of $\pi_\text{mid}$ is held fixed throughout training as the DPO reference, loaded directly from the checkpoint file to avoid PyTorch deepcopy issues with SAC's weight-norm layers.
* **Checkpoint selection:** Rather than using training loss (which decreases monotonically toward zero on finite datasets), the policy is evaluated in the true environment every `eval_every` epochs and the checkpoint with the highest mean return is restored at the end of training.
* **Evaluation:** Reported metrics are the mean (and standard deviation) of the best-checkpoint return across 5 independent seeds.

## Configuration

All hyperparameters are controlled via a single `RUN_CONFIG` dict in the notebook. The three-level merge means only deviations from global defaults need to be written:

```
1. _HPARAM_DEFAULTS       ← global fallback
2. RUN_CONFIG[env]["default"]   ← per-environment override
3. RUN_CONFIG[env][K]           ← per-(environment, K) override
```

The final configuration used for the reported results is:

```python
RUN_CONFIG = {
    "CartPole-v1": {
        "default": {
            "lr": 1e-2, "beta": 0.1, "kl_coef": 0.1,
            "n_epochs": 300, "plateau_window": 12, "early_stop": True, "eval_every": 5,
        },
        50:   {"batch_size": 2},   # ~25 gradient steps / epoch
        200:  {"batch_size": 8},   # ~25 gradient steps / epoch
        1000: {"batch_size": 32},  # ~31 gradient steps / epoch
    },
    "Pendulum-v1": {
        "default": {
            "lr": 1e-3, "beta": 0.3, "kl_coef": 0.2,
            "n_epochs": 300, "plateau_window": 2, "early_stop": True, "eval_every": 5,
        },
        50:   {"batch_size": 2},
        200:  {"batch_size": 8},
        1000: {"batch_size": 32},
    },
    "MountainCarContinuous-v0": {
        "algo": "SAC",
        "default": {
            "lr": 1e-3, "beta": 0.4, "kl_coef": 0.4,
            "n_epochs": 300, "plateau_window": 12, "early_stop": True, "eval_every": 5,
        },
        50:   {"batch_size": 2},
        200:  {"batch_size": 8},
        1000: {"batch_size": 32},
    },
}
```

## Reproducing the Experiments

Open and run all cells in `dpo_analysis.ipynb`. The notebook will:

1. Load the pre-generated preference datasets from `data_generation/outputs/preferences/`.
2. Load the expert and mid-anchor policies from `data_generation/outputs/policies/`.
3. For each environment and each $(K, \text{seed})$ pair: fine-tune the mid policy with DPO and evaluate it.
4. Save per-run summaries, best checkpoints, and aggregated results.
5. Generate scaling plots and training-curve plots into `outputs/dpo_scaling/plots/`.

The notebook also requires that the RLHF baseline results are available (for the comparison plot), following the same 5-seed convention from the `rlhf/` pipeline.

---

## Hyperparameter Sensitivity & Ablation Study

DPO is significantly more sensitive to hyperparameters than PPO-RLHF. The three critical axes are the **KL penalty coefficient** (`kl_coef`), the **preference temperature** (`beta`), and the **learning rate** (`lr`). The optimal values differ substantially across environments depending on two structural properties: (1) discrete vs. continuous action space, and (2) dense vs. sparse reward signal.

### Why DPO is Sensitive

The DPO loss for a preference pair $(\tau^+, \tau^-)$ is:

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left(\beta \cdot \left[\log\frac{\pi_\theta(\tau^+)}{\pi_\text{ref}(\tau^+)} - \log\frac{\pi_\theta(\tau^-)}{\pi_\text{ref}(\tau^-)}\right]\right) + \lambda_\text{KL} \cdot \mathbb{E}_{s}\!\left[\mathrm{KL}(\pi_\theta(\cdot|s) \,\|\, \pi_\text{ref}(\cdot|s))\right]$$

On a finite offline dataset, the first term decreases monotonically toward zero — the policy will eventually assign probability one to every chosen trajectory and zero to every rejected one, regardless of whether this generalises to unseen states. The KL term is the only force that resists this collapse. Getting the balance wrong in either direction is immediately visible: too low a `kl_coef` causes the policy to memorise the dataset and collapse on unseen states; too high a `kl_coef` prevents the policy from moving away from the reference at all (loss stuck at $\log 2 \approx 0.693$, the value when $\pi_\theta = \pi_\text{ref}$).

### Effect of Action Space (Discrete vs. Continuous)

For **discrete** actions (CartPole), the KL divergence between two Categorical distributions is bounded above by $\log |\mathcal{A}|$. A moderate `kl_coef = 0.1` is enough to anchor the policy while still allowing fast learning; higher learning rates ($10^{-2}$) are stable because the gradient direction is unambiguous.

For **continuous** actions (Pendulum, MountainCar), the KL between two Gaussian policies is unbounded — large changes in mean or variance lead to unbounded penalties. A higher `kl_coef` (0.2–0.4) is needed for the same anchoring effect, and a lower learning rate ($10^{-3}$) avoids overshooting, especially under Adam whose step size is roughly `lr` regardless of gradient magnitude.

### Effect of Reward Density (Dense vs. Sparse)

For **dense** rewards (CartPole, Pendulum), every preference pair carries a meaningful signal: one trajectory is clearly better than the other across the full episode. DPO gradients are informative at every update.

For **sparse** rewards (MountainCar), only a fraction of pairs contain a strong signal (one trajectory reached the goal, the other did not). The remaining pairs differ only by the small action-penalty term ($-0.1 \cdot a^2$ per step), which is nearly uniform. DPO will try to fit those noisy pairs as well, so a stronger KL anchor (`kl_coef = 0.4`) and a higher preference temperature (`beta = 0.4`) are both needed: the KL prevents overfitting to noise, while the higher beta ensures that the informative goal-reaching pairs are exploited more aggressively when they do appear.

### Batch Size: Keeping Updates Per Epoch Constant

The number of gradient steps per epoch equals $K / \text{batch\_size}$. With per-pair SGD (`batch_size = 1`), a run with $K = 1000$ takes 20× more steps per epoch than one with $K = 50$, making the effective learning rate K-dependent. To decouple dataset size from training dynamics, batch sizes are scaled to maintain approximately **25 gradient steps per epoch** across all K:

| K | batch\_size | steps/epoch |
|---|---|---|
| 50 | 2 | 25 |
| 200 | 8 | 25 |
| 1000 | 32 | ≈ 31 |

### Early Stopping and Epoch Budget

With return-based checkpoint selection, the epoch budget (`n_epochs`) is an upper bound rather than an exact training length. Early stopping (`early_stop = True`) halts training when the evaluated return has not exceeded the best seen return for `plateau_window` consecutive evaluations (each evaluation covering `eval_every` epochs).

For CartPole, `plateau_window = 12` gives 60 epochs of patience — enough to avoid premature stopping on a discrete environment where returns can dip temporarily. For Pendulum, `plateau_window = 2` is sufficient because the dense reward signal makes learning steady once it begins; longer patience would waste compute after the policy has converged. For MountainCar, `plateau_window = 12` is needed because the sparse reward makes progress intermittent: the policy can plateau for many evaluations while slowly building the momentum exploitation required to reach the goal.

---

## Results

All numbers below are averaged over 5 independent seeds.

### Scaling Plots

| Environment | Plot |
|---|---|
| CartPole-v1 | `outputs/dpo_scaling/plots/CartPole-v1_scaling_plot.png` |
| Pendulum-v1 | `outputs/dpo_scaling/plots/Pendulum-v1_scaling_plot.png` |
| MountainCarContinuous-v0 | `outputs/dpo_scaling/plots/MountainCarContinuous-v0_scaling_plot.png` |

### Training Curves

| Environment | Plot |
|---|---|
| CartPole-v1 | `outputs/dpo_scaling/plots/CartPole-v1_training_curve.png` |
| Pendulum-v1 | `outputs/dpo_scaling/plots/Pendulum-v1_training_curve.png` |
| MountainCarContinuous-v0 | `outputs/dpo_scaling/plots/MountainCarContinuous-v0_training_curve.png` |

### Summary Table

| Environment | Mid (anchor) | Expert | DPO K=50 | DPO K=200 | DPO K=1000 |
|---|---|---|---|---|---|
| CartPole-v1 | 360.9 | 500.0 | **500.0 ± 0.0** | **500.0 ± 0.0** | **500.0 ± 0.0** |
| Pendulum-v1 | −644.7 | −340.0 | **−179.3 ± 14.8** | **−179.7 ± 10.4** | **−156.2 ± 8.3** |
| MountainCarContinuous-v0 | 61.6 | 94.4 | **92.9 ± 0.6** | **92.8 ± 0.3** | **92.9 ± 0.3** |

### Interpretation

**CartPole-v1 (discrete, dense):** DPO achieves the maximum possible return of 500 across all 5 seeds and all K values, including the smallest dataset of 50 pairs. This demonstrates that for a well-behaved discrete environment, DPO requires very little data to fully recover the expert policy. The PPO-RLHF baseline needed K=200 to reliably reach 500; DPO is more sample-efficient here.

**Pendulum-v1 (continuous, dense):** DPO not only surpasses the mid-anchor policy but also surpasses the expert baseline at every K value. The expert policy ($\pi_1$) achieves −340, while DPO at K=50 already reaches −179 — nearly twice as good. This is a case where DPO's direct fine-tuning signal from preferences is more powerful than the PPO-RLHF pipeline, which must learn an intermediate reward model. Variance is also low (≤ 15 across seeds), indicating stable training. PPO-RLHF reaches −187 only at K=1000.

**MountainCarContinuous-v0 (continuous, sparse, SAC):** Despite the sparse reward and the noisy preference signal, DPO reaches near-expert performance (≈92.9 vs expert 94.4) at all K values, with low variance (std ≤ 0.6). PPO-RLHF on the same environment is highly unstable (std up to 45), often failing to improve over the mid-anchor policy. The strong KL regularisation and higher preference temperature used here were critical: earlier configurations with lower `kl_coef` caused the policy to overfit to the action-penalty noise in uninformative pairs and collapse.
