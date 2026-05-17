# Per-environment reward model training hyperparameters.
# Epochs and LR were chosen by inspecting loss curves via plot_rm_convergence.py:
#   CartPole    converges in ~3 epochs; 5 is a safe margin.
#   Pendulum    oscillates at lr>=1e-4; lr=3e-5 + 40 epochs stabilises it.
#   MountainCar needs many epochs; lr=3e-5 prevents oscillation.
RM_TRAIN_CONFIG = {
    "CartPole-v1":              {"epochs": 5,  "lr": 3e-4},
    "Pendulum-v1":              {"epochs": 10, "lr": 3e-5},
    "MountainCarContinuous-v0": {"epochs": 25, "lr": 3e-5},
}
