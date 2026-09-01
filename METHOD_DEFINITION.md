# Exact method definition used by TAOT-SRC v4

## Feature representation

- 2 x 2 spatial HOG grid.
- 9 unsigned orientation bins per cell.
- 36 nonnegative bins total.
- Hard orientation-bin assignment.
- Bin centers used by the ground cost are `(b + 0.5) * pi / 9`.
- A floor of `1e-6` is added before unit-mass normalization.

## Ground cost

Each support bin has a spatial cell and an unsigned orientation in `[0, pi)`. Spatial distance is normalized by the 2 x 2 grid size. Circular angular distance uses the shortest distance modulo pi. Squared spatial and angular distances are combined with weight `eta`, then the cost matrix is normalized by its maximum (with a `1e-12` numerical denominator safeguard).

Linear-TAOT uses the same construction except that angular distance is unwrapped. All real-data OT comparison methods use the same `eta` and `epsilon` selected/frozen for the experiment.

## Sinkhorn quantity

`P_epsilon` is the coupling obtained by entropically regularized Sinkhorn scaling. The experimental OT score is the transport term `<P_epsilon, C>`.

It is not a debiased Sinkhorn divergence and not the complete regularized primal value. A diagnostic full-primal function remains in `code/taot_src.py`, but is not used for classification.

## Class reconstruction

For each candidate class, training histograms are mixed with simplex weights parameterized by softmax. TAOT minimizes

`transport_cost(x, D_c^T w) + lambda * H(w)`

where the code defines exactly

`H(w) = -sum_j w_j log(w_j + 1e-9)`.

The optimizer uses 70 Adam updates at learning rate 0.35, 30 Sinkhorn iterations per coefficient update, and 50 iterations for the final TAOT score.

## Pairwise OT baselines

OT-1NN and post-hoc OT-SRC use 60 Sinkhorn iterations. Post-hoc OT-SRC obtains nonnegative SRC class reconstructions and the Sinkhorn routine normalizes each nonnegative reconstruction to unit mass before transport scoring.

## Other baselines

- canonical signed SRC: Lasso alpha = 1e-4, Euclidean class residual;
- nonnegative SRC: positive Lasso alpha = 1e-4;
- CRC: ridge = 1e-2;
- RBF-SVM: C = 3, gamma = `scale`.

## Validation and held-out protocol

Hyperparameters `lambda`, `epsilon`, and `eta` are selected only on Digits validation seeds 0--4 by `code/run_validation.py`. The resulting JSON is then supplied to the held-out driver. Held-out seeds are 5--19.

## Synthetic mechanism test

The synthetic hairpin experiment keeps epsilon = 0.03 and lambda = 2.5e-4. It is not affected by the real-data split/rotation correction.
