from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

def _validate_probs(probs, labels=None, n_bins=10):
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probs must be a finite non-negative (N, C) array with N>=1 and C>=2")
    if not np.allclose(p.sum(1), 1.0, atol=1e-6): raise ValueError("probability rows must sum to 1 within 1e-6")
    if not isinstance(n_bins, int) or n_bins < 1: raise ValueError("n_bins must be a positive integer")
    if labels is None: return p, None
    y = np.asarray(labels)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("labels must be integer class indices in [0, C)")
    return p, y

def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[ReliabilityBin]:
    p, y = _validate_probs(probs, labels, n_bins); conf = p.max(1); pred = p.argmax(1); idx = np.minimum(n_bins-1, np.maximum(0, np.ceil(conf*n_bins).astype(int)-1))
    return [ReliabilityBin(b/n_bins, (b+1)/n_bins, int((m:=idx==b).sum()), float(conf[m].mean()) if m.any() else 0.0, float((pred[m]==y[m]).mean()) if m.any() else 0.0) for b in range(n_bins)]

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_bins(probs, labels, n_bins); n = sum(b.count for b in bins)
    return float(sum(b.count * abs(b.mean_confidence-b.accuracy) for b in bins) / n)

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 2 or not np.isfinite(z).all(): raise ValueError("logits must be a finite (N, C) array")
    if not np.isfinite(temperature) or temperature <= 0: raise ValueError("temperature must be finite and positive")
    a = z / temperature; a -= a.max(axis=1, keepdims=True); e = np.exp(a); return e / e.sum(axis=1, keepdims=True)

def fit_temperature(logits: np.ndarray, labels: np.ndarray, t_min: float = 0.05, t_max: float = 20.0) -> float:
    z = np.asarray(logits, dtype=float); y = np.asarray(labels)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 2 or not np.isfinite(z).all(): raise ValueError("logits must be finite with shape (N, C)")
    if y.shape != (len(z),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= z.shape[1]).any(): raise ValueError("invalid labels")
    if not (np.isfinite(t_min) and np.isfinite(t_max) and 0 < t_min <= t_max): raise ValueError("invalid temperature bounds")
    if np.all(np.ptp(z, axis=1) == 0): return 1.0
    def nll(t): return float(-np.log(np.clip(apply_temperature(z, t)[np.arange(len(y)), y], 1e-300, 1.0)).mean())
    # Fine log grid plus golden-section refinement around its best point.
    grid = np.geomspace(t_min, t_max, 401); vals = np.array([nll(t) for t in grid]); i = int(vals.argmin())
    if i == 0: return float(t_min)
    if i == len(grid)-1: return float(t_max)
    lo, hi = grid[i-1], grid[i+1]; phi = (1+5**0.5)/2
    x1, x2 = hi-(hi-lo)/phi, lo+(hi-lo)/phi
    for _ in range(70):
        if nll(x1) < nll(x2): hi, x2 = x2, x1; x1 = hi-(hi-lo)/phi
        else: lo, x1 = x1, x2; x2 = lo+(hi-lo)/phi
    return float((lo+hi)/2)
