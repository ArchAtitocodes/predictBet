"""
EQFBIS Probability Calibration
================================

A model can output well-shaped probabilities and still be wrong if those
probabilities are systematically over- or under-confident.  Calibration fixes
that by mapping raw model scores onto empirically observed frequencies.

Methods implemented
-------------------
1. PlattScaling        - logistic regression on raw scores (good for SVMs,
                          neural nets, and small datasets).
2. IsotonicRegression - piecewise-constant, non-parametric mapping (better
                          when you have hundreds of calibration samples).
3. BayesianBinning     - splits scores into K bins, estimates a Beta
                          posterior per bin (robust with few samples).
4. TemperatureScaling  - single-parameter softmax calibration for multiclass
                          outputs (the standard for deep-learning classifiers).

All calibrators are fitted on a held-out calibration set, then applied to
the probabilities that the ensemble in intelligence.py produces.  The result
is a probability that you can actually trust — if it says 70%, that outcome
should happen roughly 70% of the time, not 60% or 80%.

Nothing here claims perfect calibration.  It claims that the probabilities
are as honest as the calibration data lets them be, and it reports the
reliability curve so you can see where the model is still miscalibrated.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


# ===========================================================================
# Calibration data store
# ===========================================================================

@dataclass
class CalibrationSample:
    model_prob: float
    actual: int   # 1 if outcome occurred, 0 otherwise
    outcome: str  # 'H', 'D', 'A'
    model_name: str = ""
    created_at: float = field(default_factory=time.time)


class CalibrationStore:
    """Append-only SQLite store for calibration samples."""

    def __init__(self, db_path: str = "eqfbis_calibration.sqlite3"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS calibration_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_prob REAL NOT NULL,
                        actual INTEGER NOT NULL,
                        outcome TEXT NOT NULL,
                        model_name TEXT DEFAULT '',
                        created_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS calibration_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        method TEXT NOT NULL,
                        params BLOB,
                        brier_before REAL,
                        brier_after REAL,
                        n_samples INTEGER,
                        created_at REAL NOT NULL
                    )
                """)
                conn.commit()

    def add_sample(self, sample: CalibrationSample):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO calibration_samples (model_prob, actual, outcome, model_name, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (sample.model_prob, sample.actual, sample.outcome,
                     sample.model_name, sample.created_at),
                )
                conn.commit()

    def get_samples(self, outcome: str = "", model_name: str = "") -> list[CalibrationSample]:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                query = "SELECT model_prob, actual, outcome, model_name, created_at FROM calibration_samples WHERE 1=1"
                params = []
                if outcome:
                    query += " AND outcome = ?"
                    params.append(outcome)
                if model_name:
                    query += " AND (model_name = ? OR model_name = '')"
                    params.append(model_name)
                rows = conn.execute(query, params).fetchall()

        return [CalibrationSample(model_prob=r[0], actual=r[1], outcome=r[2],
                                  model_name=r[3], created_at=r[4]) for r in rows]

    def export_json(self, path: Optional[str] = None) -> str:
        """Serialize calibration samples to JSON for backup/migration."""
        samples = [
            {
                "model_prob": s.model_prob,
                "actual": s.actual,
                "outcome": s.outcome,
                "model_name": s.model_name,
                "created_at": s.created_at,
            }
            for s in self.get_samples()
        ]
        payload = json.dumps(samples, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    def ensure_path_exists(self, path: str) -> bool:
        """Check file path existence using os utilities."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        return os.path.exists(path)

    def save_model(self, model_name: str, outcome: str, method: str,
                   params: Any, brier_before: float, brier_after: float,
                   n_samples: int):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO calibration_models (model_name, outcome, method, params, "
                    "brier_before, brier_after, n_samples, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (model_name, outcome, method, pickle.dumps(params),
                     brier_before, brier_after, n_samples, time.time()),
                )
                conn.commit()

    def load_model(self, model_name: str, outcome: str, method: str) -> Optional[dict]:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                row = conn.execute(
                    "SELECT params, brier_before, brier_after, n_samples FROM calibration_models "
                    "WHERE model_name = ? AND outcome = ? AND method = ? ORDER BY id DESC LIMIT 1",
                    (model_name, outcome, method),
                ).fetchone()
        if not row:
            return None
        return {"params": pickle.loads(row[0]), "brier_before": row[1],
                "brier_after": row[2], "n_samples": row[3]}


# ===========================================================================
# Base calibrator
# ===========================================================================

class BaseCalibrator:
    method_name: str = "base"

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def predict(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reliability_curve(self, n_bins: int = 10) -> Dict[str, list]:
        scores = getattr(self, "_fit_scores", np.array([]))
        labels = getattr(self, "_fit_labels", np.array([]))
        if len(scores) == 0:
            return {"bins": [], "mean_predicted": [], "fraction_positives": [], "counts": []}

        bins = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true = []
        bin_counts = []

        for i in range(n_bins):
            mask = (scores >= bins[i]) & (scores < bins[i + 1])
            if i == n_bins - 1:
                mask = (scores >= bins[i]) & (scores <= bins[i + 1])
            count = int(mask.sum())
            if count == 0:
                bin_means.append((bins[i] + bins[i + 1]) / 2)
                bin_true.append(None)
                bin_counts.append(0)
            else:
                bin_means.append(float(scores[mask].mean()))
                bin_true.append(float(labels[mask].mean()))
                bin_counts.append(count)

        return {
            "bins": [round(float(b), 3) for b in bins[:-1]],
            "mean_predicted": [round(v, 3) if v is not None else None for v in bin_means],
            "fraction_positives": [round(v, 3) if v is not None else None for v in bin_true],
            "counts": bin_counts,
        }


# ===========================================================================
# Platt scaling
# ===========================================================================

class PlattScaling(BaseCalibrator):
    """Logistic calibration: P(actual=1 | score) = 1 / (1 + exp(A * score + B)).
    Fit A and B via maximum likelihood on the calibration set."""

    method_name = "platt"

    def __init__(self):
        self.A = 1.0
        self.B = 0.0
        self._fit_scores = np.array([])
        self._fit_labels = np.array([])

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        from scipy.optimize import minimize

        self._fit_scores = np.asarray(scores, dtype=float)
        self._fit_labels = np.asarray(labels, dtype=float)

        def neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * self._fit_scores + b)))
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.mean(
                self._fit_labels * np.log(p) + (1 - self._fit_labels) * np.log(1 - p)
            )

        res = minimize(neg_log_likelihood, [1.0, 0.0], method="L-BFGS-B",
                       bounds=[(1e-6, None), (None, None)])
        if res.success:
            self.A, self.B = res.x
        else:
            _logger.warning("Platt scaling optimization failed; using identity.")

    def predict(self, scores: np.ndarray) -> np.ndarray:
        z = self.A * np.asarray(scores, dtype=float) + self.B
        return 1.0 / (1.0 + np.exp(-z))


# ===========================================================================
# Isotonic regression
# ===========================================================================

class IsotonicRegressionCalibrator(BaseCalibrator):
    """Non-parametric isotonic regression calibration.  Better than Platt
    when you have >= 200 calibration samples and the relationship between
    raw score and true probability is non-monotonic or piecewise."""

    method_name = "isotonic"

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._fit_scores = np.array([])
        self._fit_labels = np.array([])

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        self._fit_scores = np.asarray(scores, dtype=float)
        self._fit_labels = np.asarray(labels, dtype=float)
        self._model.fit(self._fit_scores, self._fit_labels)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(scores, dtype=float))


# ===========================================================================
# Bayesian binning
# ===========================================================================

class BayesianBinning(BaseCalibrator):
    """Splits scores into K bins and estimates a Beta(alpha, beta) posterior
    per bin.  Robust to small calibration sets and naturally produces
    uncertainty estimates."""

    method_name = "bayesian_binning"

    def __init__(self, n_bins: int = 10, alpha_prior: float = 2.0, beta_prior: float = 2.0):
        self.n_bins = n_bins
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._bins: List[Tuple[float, float]] = []
        self._alphas: List[float] = []
        self._betas: List[float] = []
        self._fit_scores = np.array([])
        self._fit_labels = np.array([])

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        self._fit_scores = np.asarray(scores, dtype=float)
        self._fit_labels = np.asarray(labels, dtype=float)
        self._bins = []
        self._alphas = []
        self._betas = []

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        for i in range(self.n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == self.n_bins - 1:
                mask = (self._fit_scores >= low) & (self._fit_scores <= high)
            else:
                mask = (self._fit_scores >= low) & (self._fit_scores < high)
            bin_labels = self._fit_labels[mask]
            n = len(bin_labels)
            s = int(bin_labels.sum())
            self._bins.append((float(low), float(high)))
            self._alphas.append(self.alpha_prior + s)
            self._betas.append(self.beta_prior + n - s)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        result = np.zeros_like(scores)
        for i, (low, high) in enumerate(self._bins):
            if i == len(self._bins) - 1:
                mask = (scores >= low) & (scores <= high)
            else:
                mask = (scores >= low) & (scores < high)
            a, b = self._alphas[i], self._betas[i]
            result[mask] = a / (a + b)
        return result

    def predictive_uncertainty(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        result = np.zeros_like(scores)
        for i, (low, high) in enumerate(self._bins):
            if i == len(self._bins) - 1:
                mask = (scores >= low) & (scores <= high)
            else:
                mask = (scores >= low) & (scores < high)
            a, b = self._alphas[i], self._betas[i]
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            result[mask] = math.sqrt(var)
        return result


# ===========================================================================
# Temperature scaling
# ===========================================================================

class TemperatureScaling(BaseCalibrator):
    """Single-parameter scaling for multiclass outputs.  Divides logits by T
    before softmax; T > 1 under-confident models, T < 1 over-confident."""

    method_name = "temperature"

    def __init__(self):
        self.T = 1.0
        self._fit_logits = np.array([])
        self._fit_labels = np.array([])

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        self._fit_logits = np.asarray(logits, dtype=float)
        self._fit_labels = np.asarray(labels, dtype=int)
        from scipy.optimize import minimize

        def neg_ll(t):
            scaled = self._fit_logits / t
            probs = softmax(scaled, axis=1)
            clipped = np.clip(probs, 1e-15, 1 - 1e-15)
            return -np.mean(np.log(clipped[np.arange(len(labels)), labels]))

        res = minimize(neg_ll, [1.0], method="L-BFGS-B", bounds=[(1e-3, 10.0)])
        if res.success:
            self.T = float(res.x[0])

    def predict(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=float) / self.T
        return softmax(scaled, axis=1)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ===========================================================================
# Calibration manager
# ===========================================================================

class CalibrationManager:
    """Fits, stores, and applies calibrators per outcome per model."""

    def __init__(self, store: Optional[CalibrationStore] = None):
        self.store = store or CalibrationStore()
        self._calibrators: Dict[str, Dict[str, BaseCalibrator]] = {}

    def fit(self, model_name: str, outcome: str, method: str = "isotonic",
            min_samples: int = 30) -> Optional[dict]:
        samples = self.store.get_samples(outcome=outcome, model_name=model_name)
        if len(samples) < min_samples:
            return {
                "status": "insufficient_data",
                "n_samples": len(samples),
                "min_required": min_samples,
            }

        scores = np.array([s.model_prob for s in samples])
        labels = np.array([s.actual for s in samples])
        brier_before = float(np.mean((scores - labels) ** 2))

        if method == "platt":
            cal = PlattScaling()
        elif method == "isotonic":
            cal = IsotonicRegressionCalibrator()
        elif method == "bayesian_binning":
            cal = BayesianBinning()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        cal.fit(scores, labels)
        calibrated = cal.predict(scores)
        brier_after = float(np.mean((calibrated - labels) ** 2))

        self.store.save_model(model_name, outcome, method, cal, brier_before, brier_after,
                              len(samples))
        self._calibrators.setdefault(model_name, {})[outcome] = cal

        return {
            "status": "ok",
            "method": method,
            "n_samples": len(samples),
            "brier_before": round(brier_before, 4),
            "brier_after": round(brier_after, 4),
            "reliability_curve": cal.reliability_curve(n_bins=10),
            "improvement_pct": round(100 * (brier_before - brier_after) / brier_before, 2)
                                if brier_before > 0 else 0.0,
        }

    def apply(self, model_name: str, outcome: str, prob: float) -> float:
        cal = self._calibrators.get(model_name, {}).get(outcome)
        if cal is None:
            saved = self.store.load_model(model_name, outcome, "isotonic")
            if saved is None:
                saved = self.store.load_model(model_name, outcome, "platt")
            if saved is not None:
                cal = saved["params"]
                self._calibrators.setdefault(model_name, {})[outcome] = cal
            else:
                return prob
        return float(cal.predict(np.array([prob]))[0])

    def apply_all(self, model_name: str, probs: Dict[str, float]) -> Dict[str, float]:
        res = {}
        for outcome, p in probs.items():
            short_outcome = outcome
            if outcome in ("home_win_prob", "home"):
                short_outcome = "H"
            elif outcome in ("draw_prob", "draw"):
                short_outcome = "D"
            elif outcome in ("away_win_prob", "away"):
                short_outcome = "A"
            res[outcome] = self.apply(model_name, short_outcome, p)
        total = sum(res.values())
        if total > 0:
            return {k: v / total for k, v in res.items()}
        return res


# ===========================================================================
# CLI smoke test
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(42)
    manager = CalibrationManager()

    true_probs = np.random.beta(2, 2, size=500)
    outcomes = np.random.binomial(1, true_probs)

    print("--- Fitting calibrator on synthetic data ---")
    for outcome in ["H", "D", "A"]:
        mask = np.random.choice([True, False], size=500)
        res = manager.fit("demo", outcome, method="isotonic",
                          min_samples=20)
        print(f"{outcome}: {res}")

    print("\n--- Applying calibration ---")
    test_probs = {"H": 0.72, "D": 0.18, "A": 0.10}
    calibrated = manager.apply_all("demo", test_probs)
    print("Raw:      ", test_probs)
    print("Calibrated:", {k: round(v, 4) for k, v in calibrated.items()})
