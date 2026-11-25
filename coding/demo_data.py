import numpy as np

def make_demo(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.70, 0.25, 0.05])  # 0 request, 1 complaint, 2 urgent

    def model(acc_per_class: list[float]) -> np.ndarray:
        right = rng.random(n) < np.asarray(acc_per_class)[y]
        return np.where(right, y, (y + rng.integers(1, 3, size=n)) % 3)

    return y, model([0.92, 0.80, 0.70]), model([0.90, 0.85, 0.55])
