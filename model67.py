"""A minimal "Model 67" implemented from scratch.

This module defines a trainable linear regression model with exactly 67
input weights and one bias term. Training uses plain Python gradient
descent (no external ML libraries).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

FEATURE_COUNT = 67


@dataclass
class Model67:
    """Linear regression model with 67 features."""

    learning_rate: float = 0.01
    weights: List[float] = field(default_factory=lambda: [0.0] * FEATURE_COUNT)
    bias: float = 0.0

    def predict_one(self, features: Sequence[float]) -> float:
        if len(features) != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} features, got {len(features)}")
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    def predict(self, xs: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_one(x) for x in xs]

    def mse(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> float:
        preds = self.predict(xs)
        n = len(ys)
        return sum((p - y) ** 2 for p, y in zip(preds, ys)) / n

    def train_epoch(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> float:
        n = len(xs)
        grad_w = [0.0] * FEATURE_COUNT
        grad_b = 0.0

        for x, y in zip(xs, ys):
            pred = self.predict_one(x)
            err = pred - y
            for i in range(FEATURE_COUNT):
                grad_w[i] += (2.0 / n) * err * x[i]
            grad_b += (2.0 / n) * err

        for i in range(FEATURE_COUNT):
            self.weights[i] -= self.learning_rate * grad_w[i]
        self.bias -= self.learning_rate * grad_b

        return self.mse(xs, ys)

    def train(
        self,
        xs: Sequence[Sequence[float]],
        ys: Sequence[float],
        epochs: int = 200,
    ) -> List[float]:
        losses: List[float] = []
        for _ in range(epochs):
            losses.append(self.train_epoch(xs, ys))
        return losses


def make_synthetic_dataset(
    n_samples: int = 600,
    seed: int = 67,
) -> Tuple[List[List[float]], List[float], List[float], float]:
    """Create data generated from a hidden 67-weight linear function."""
    rng = random.Random(seed)
    true_weights = [rng.uniform(-2.0, 2.0) for _ in range(FEATURE_COUNT)]
    true_bias = rng.uniform(-0.5, 0.5)

    xs: List[List[float]] = []
    ys: List[float] = []

    for _ in range(n_samples):
        x = [rng.uniform(-1.0, 1.0) for _ in range(FEATURE_COUNT)]
        noise = rng.uniform(-0.05, 0.05)
        y = sum(w * xi for w, xi in zip(true_weights, x)) + true_bias + noise
        xs.append(x)
        ys.append(y)

    return xs, ys, true_weights, true_bias


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_true) ** 2 for y in y_true)
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot)


def demo() -> None:
    xs, ys, _, _ = make_synthetic_dataset()

    train_x, train_y = xs[:500], ys[:500]
    test_x, test_y = xs[500:], ys[500:]

    model = Model67(learning_rate=0.02)
    start_loss = model.mse(train_x, train_y)
    losses = model.train(train_x, train_y, epochs=250)
    end_loss = losses[-1]

    test_preds = model.predict(test_x)
    test_r2 = r2_score(test_y, test_preds)

    print(f"Start train MSE: {start_loss:.4f}")
    print(f"End train MSE:   {end_loss:.4f}")
    print(f"Test R^2:        {test_r2:.4f}")


if __name__ == "__main__":
    demo()
