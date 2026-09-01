"""Frozen experimental configuration for TAOT-SRC reproducibility."""
from pathlib import Path

# Features
CELLS = (2, 2)
NBINS = 9
HOG_FLOOR = 1e-6
D = 36

# OT / TAOT
ETA = 0.5
EPSILON = 0.01
LAMBDA = 0.0  # v4 validation-selected real-data value  # default; override after validation if selected_hyperparameters.json differs
ADAM_STEPS = 70
ADAM_LR = 0.35
SINKHORN_OPT_ITERS = 30
SINKHORN_FINAL_ITERS = 50
SINKHORN_PAIR_ITERS = 60

# Baselines
SRC_ALPHA = 1e-4
CRC_RIDGE = 1e-2
SVM_C = 3.0
SVM_GAMMA = "scale"

# Protocol
VALIDATION_SEEDS = tuple(range(0, 5))
HELDOUT_SEEDS = tuple(range(5, 20))
ROTATIONS = (0, 10, 20)
FEWSHOT = (1, 2, 3, 5, 10)
MAX_SHOT = max(FEWSHOT)
DEFAULT_SHOT = 5
DIGITS_NTEST = 10
LFW_NTEST = 30

# Validation grids used historically
LAMBDA_GRID = (0.0, 2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3)
EPSILON_GRID = (0.005, 0.01, 0.02, 0.05)
ETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)

DATASETS = ("Digits", "LFW-Face")

# Synthetic mechanism-test settings (historical/frozen for that experiment)
SYNTH_EPSILON = 0.03
SYNTH_LAMBDA = 2.5e-4
SYNTH_SEEDS = tuple(range(20))
