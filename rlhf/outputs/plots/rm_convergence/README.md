# Reward Model Convergence Analysis

Bradley-Terry loss curves from diagnostic runs used to select the per-environment
hyperparameters (epochs, learning rate) for Phase 1 (reward model training).

Each plot trains reward models for 50 epochs — more than any candidate cutoff — across
5 seeds and all three dataset sizes (K = 50, 200, 1000). The red dashed vertical line
marks the epoch cutoff finally chosen for the full pipeline.

---

## Iteration 0 — uniform settings: lr=3e-4, epochs=10

**CartPole-v1** (`epochs10_lr3e-4`): Clean convergence within 2–3 epochs across all K
and seeds. Loss reaches near-zero well before the cutoff.

**Pendulum-v1** (`epochs10_lr3e-4`): Severe oscillation throughout all 10 epochs,
especially for K=200 and K=1000. The learning rate is too high for the loss landscape.

**MountainCarContinuous-v0** (`epochs10_lr3e-4`): Loss still clearly falling at epoch 10
for all K values — the model is underfitting. More epochs and/or a lower LR are needed.

---

## Iteration 1 — per-environment tuning attempt

- CartPole: epochs=5, lr=3e-4 (reduce to safe minimum)
- Pendulum: epochs=25, lr=1e-4 (lower LR, more epochs)
- MountainCar: epochs=35, lr=3e-4 (more epochs, same LR)

**CartPole-v1** (`epochs5_lr0.0003`): Still converges cleanly by epoch 3. Confirmed that
5 epochs is sufficient.

**Pendulum-v1** (`epochs25_lr0.0001`): Oscillation persists for K=200 at lr=1e-4.
Learning rate is still too high.

**MountainCarContinuous-v0** (`epochs35_lr0.0003`): Strong oscillation for K=50 and
K=200. The LR is the primary problem, not the epoch count.

---

## Iteration 2 — aggressive LR reduction: lr=3e-5 for Pendulum and MountainCar

- Pendulum: epochs=40, lr=3e-5
- MountainCar: epochs=50, lr=3e-5

**Pendulum-v1** (`epochs40_lr3e-05`): All three K values converge cleanly within 5–7
epochs with no oscillation. The loss is flat well before epoch 40.

**MountainCarContinuous-v0** (`epochs50_lr3e-05`): K=50 and K=200 converge within
5 epochs. K=1000 is still gently declining at epoch 20 but reaches a stable plateau
by epoch 25. No oscillation at any K.

---

## Final settings adopted for the full pipeline

| Environment              | epochs | lr   | Rationale                                           |
|--------------------------|--------|------|-----------------------------------------------------|
| CartPole-v1              | 5      | 3e-4 | Converges by epoch 3; 5 gives a safety buffer       |
| Pendulum-v1              | 10     | 3e-5 | Converges by epoch 6–7; 10 gives a safety buffer    |
| MountainCarContinuous-v0 | 25     | 3e-5 | K=1000 stabilises by epoch 20–25; smaller K earlier |

The binding constraint in every case is the largest dataset (K=1000). Smaller K values
converge earlier, but the extra epochs are harmless given the negligible Phase 1 cost.
