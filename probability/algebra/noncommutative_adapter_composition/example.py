"""Order-sensitive composition of low-rank adapter-like operators."""

import numpy as np

rng = np.random.default_rng(11)

D = 4
X = rng.normal(size=(128, D))
I = np.eye(D)

# Rank-1 LoRA-like deltas.
u_a = np.array([0.8, -0.3, 0.2, 0.0])[:, None]
v_a = np.array([0.1, 0.7, -0.4, 0.2])[None, :]
u_b = np.array([-0.2, 0.5, 0.6, -0.1])[:, None]
v_b = np.array([0.6, -0.2, 0.1, 0.5])[None, :]

Delta_A = 0.45 * (u_a @ v_a)
Delta_B = 0.45 * (u_b @ v_b)
A = I + Delta_A
B = I + Delta_B

# Row-vector convention. Applying A then B corresponds to X @ A.T @ B.T.
target = X @ A.T @ B.T
ab = X @ A.T @ B.T
ba = X @ B.T @ A.T
naive_add = X @ (I + Delta_A + Delta_B).T


def mse(a, b):
    return float(np.mean((a - b) ** 2))


commutator = A @ B - B @ A

print("Adapter commutator ||AB - BA||_F =", round(np.linalg.norm(commutator), 6))
print("\nMSE against target behavior (A then B)")
print("  A -> B      :", round(mse(ab, target), 10))
print("  B -> A      :", round(mse(ba, target), 10))
print("  naive A + B :", round(mse(naive_add, target), 10))

if np.linalg.norm(commutator) > 1e-3:
    print("\nDiagnostic: adapters do not approximately commute; order should be evaluated explicitly.")
