"""Lie-algebra constrained adapter vs unconstrained linear adapter."""

import numpy as np

rng = np.random.default_rng(3)


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# Frozen hidden representations.
train_x = rng.normal(size=(6, 2))
val_x = rng.normal(size=(200, 2))

true_angle = np.deg2rad(32.0)
R_true = rotation(true_angle)
# The tiny adaptation set is noisy, while validation measures retention of the
# clean underlying transformation.
train_y = train_x @ R_true.T + rng.normal(scale=0.12, size=(len(train_x), 2))
val_y = val_x @ R_true.T

# Baseline: unrestricted 2x2 matrix (4 trainable scalars).
W = np.eye(2)
for _ in range(100):
    pred = train_x @ W.T
    grad_W = 2.0 * (pred - train_y).T @ train_x / len(train_x)
    W -= 0.08 * grad_W

# Lie adapter: theta in so(2) (1 trainable scalar). Analytical derivative of R.
theta = 0.0
J = np.array([[0.0, -1.0], [1.0, 0.0]])
for _ in range(100):
    R = rotation(theta)
    pred = train_x @ R.T
    # dR/dtheta = J R = R J in SO(2)
    dR = J @ R
    dpred = train_x @ dR.T
    grad_theta = 2.0 * np.mean(np.sum((pred - train_y) * dpred, axis=1))
    theta -= 0.18 * grad_theta


def mse(pred, target):
    return float(np.mean((pred - target) ** 2))


def norm_distortion(before, after):
    return float(np.mean(np.abs(np.linalg.norm(after, axis=1) - np.linalg.norm(before, axis=1))))


val_unconstrained = val_x @ W.T
R_lie = rotation(theta)
val_lie = val_x @ R_lie.T

print("Unconstrained adapter (4 params)")
print("  train MSE       =", round(mse(train_x @ W.T, train_y), 8))
print("  validation MSE  =", round(mse(val_unconstrained, val_y), 8))
print("  norm distortion =", round(norm_distortion(val_x, val_unconstrained), 8))
print("  W =\n", np.round(W, 4))

print("\nLie adapter SO(2) (1 param)")
print("  learned angle   =", round(np.rad2deg(theta), 4), "deg")
print("  validation MSE  =", round(mse(val_lie, val_y), 8))
print("  norm distortion =", round(norm_distortion(val_x, val_lie), 12))
print("  R =\n", np.round(R_lie, 4))
