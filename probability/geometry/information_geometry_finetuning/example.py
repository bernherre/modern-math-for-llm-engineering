"""Natural-gradient vs Euclidean-gradient tuning on frozen features."""

import numpy as np

rng = np.random.default_rng(7)

# Frozen, strongly anisotropic features: the first coordinate has much larger
# scale than the second. This mimics a poorly conditioned local representation.
n = 300
x1 = rng.normal(size=n)
x2 = 0.35 * x1 + rng.normal(scale=0.25, size=n)
X = np.column_stack([18.0 * x1, x2])
true_theta = np.array([0.12, 3.0])
logits_true = X @ true_theta
labels = (logits_true + rng.normal(scale=0.8, size=n) > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def loss(theta):
    p = sigmoid(X @ theta)
    eps = 1e-12
    return float(-np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))


def accuracy(theta):
    return float(np.mean((sigmoid(X @ theta) >= 0.5) == labels))


def gradient(theta):
    p = sigmoid(X @ theta)
    return X.T @ (p - labels) / n


def fisher(theta, damping=1e-3):
    p = sigmoid(X @ theta)
    weights = p * (1 - p)
    F = X.T @ (X * weights[:, None]) / n
    return F + damping * np.eye(X.shape[1])


steps = 35
theta_gd = np.zeros(2)
theta_ng = np.zeros(2)

# Stable but deliberately conservative Euclidean learning rate because of the
# anisotropic first feature.
for _ in range(steps):
    theta_gd -= 0.0025 * gradient(theta_gd)

# Natural gradient compensates for the local Fisher geometry.
for _ in range(steps):
    g = gradient(theta_ng)
    direction = np.linalg.solve(fisher(theta_ng), g)
    theta_ng -= 0.20 * direction

print("Initial loss:", round(loss(np.zeros(2)), 5))
print("\nEuclidean gradient")
print("  theta    =", np.round(theta_gd, 4))
print("  loss     =", round(loss(theta_gd), 5))
print("  accuracy =", round(accuracy(theta_gd), 4))

print("\nNatural gradient")
print("  theta    =", np.round(theta_ng, 4))
print("  loss     =", round(loss(theta_ng), 5))
print("  accuracy =", round(accuracy(theta_ng), 4))
