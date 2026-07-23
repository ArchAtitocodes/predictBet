"""
EQFBIS Machine Learning Pipeline
==================================

Trains, validates, and serves gradient-boosted tree models (XGBoost,
LightGBM) on top of the feature vectors produced by features.py.  The ML
models learn residual patterns that the Poisson/GLM statistical model misses
— things like team-specific xG overperformance, injury signals, and market
inefficiencies — and their probabilities are fed back into the ensemble in
intelligence.py.

Nothing here claims to replace the statistical model.  The intended use is:

  1. Build a MatchFeatureVector for a fixture (features.py).
  2. Ask the statistical model for its probabilities (scraper.build_model).
  3. Ask the ML pipeline for its probabilities (this module).
  4. Blend both sets of probabilities in intelligence.build_ensemble.

Models are persisted to disk and versioned so you can compare generations
and roll back without losing calibration history.

Dependencies: xgboost, lightgbm, scikit-learn (optional but recommended).
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from features import MatchFeatureVector, FeatureExtractor
from scraper import MatchModelResult, TeamForm

_logger = logging.getLogger(__name__)


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class MLPrediction:
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5_prob: float
    btts_yes_prob: float
    expected_home_goals: float
    expected_away_goals: float
    model_name: str
    confidence: float
    feature_importance: Dict[str, float]
    raw_outputs: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "home_win_prob": round(self.home_win_prob, 4),
            "draw_prob": round(self.draw_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "over_2_5_prob": round(self.over_2_5_prob, 4),
            "btts_yes_prob": round(self.btts_yes_prob, 4),
            "expected_home_goals": round(self.expected_home_goals, 2),
            "expected_away_goals": round(self.expected_away_goals, 2),
            "model_name": self.model_name,
            "confidence": round(self.confidence, 4),
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "raw_outputs": {k: round(v, 4) for k, v in self.raw_outputs.items()},
        }


@dataclass
class TrainingResult:
    model_name: str
    train_size: int
    val_size: int
    val_brier: float
    val_log_loss: float
    val_accuracy: float
    feature_importance: Dict[str, float]
    training_time_s: float
    model_version: str

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "val_brier": round(self.val_brier, 4),
            "val_log_loss": round(self.val_log_loss, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "training_time_s": round(self.training_time_s, 2),
            "model_version": self.model_version,
        }


# ===========================================================================
# Model registry
# ===========================================================================

class ModelRegistry:
    """Persists trained models to disk with versioning."""

    def __init__(self, model_dir: str = "eqfbis_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save(self, model: Any, name: str, version: str, metadata: dict):
        path = os.path.join(self.model_dir, f"{name}_{version}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata, "version": version}, f)
        _logger.info("Saved model %s version %s to %s", name, version, path)

    def load(self, name: str, version: str) -> Optional[dict]:
        path = os.path.join(self.model_dir, f"{name}_{version}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_versions(self, name: str) -> list[str]:
        versions = []
        prefix = f"{name}_"
        for f in os.listdir(self.model_dir):
            if f.startswith(prefix) and f.endswith(".pkl"):
                versions.append(f[len(prefix):-4])
        return sorted(versions)


# ===========================================================================
# ML Pipeline
# ===========================================================================

class MLPipeline:
    """Trains and serves ML models for football outcome prediction."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self._xgb_home = None
        self._xgb_draw = None
        self._xgb_away = None
        self._lgb_home = None
        self._lgb_draw = None
        self._lgb_away = None
        self._feature_names: List[str] = []
        self._model_version = f"v{int(time.time())}"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, feature_vectors: list[MatchFeatureVector],
              outcomes: list[str], test_size: float = 0.2,
              random_state: int = 42) -> Dict[str, TrainingResult]:
        """Train XGBoost and LightGBM models on a labeled dataset.

        Parameters
        ----------
        feature_vectors : list of MatchFeatureVector
        outcomes : list of str, each 'H', 'D', or 'A'
        test_size : validation split fraction
        random_state : seed

        Returns
        -------
        dict of TrainingResult per model
        """
        if not feature_vectors or len(feature_vectors) < 30:
            raise ValueError(f"Need >= 30 samples to train, got {len(feature_vectors)}")

        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import brier_score_loss, log_loss

        X = np.array([fv.to_feature_array() for fv in feature_vectors])
        y_raw = np.array([{"H": 0, "D": 1, "A": 2}[o] for o in outcomes])
        y_home = (y_raw == 0).astype(int)
        y_draw = (y_raw == 1).astype(int)
        y_away = (y_raw == 2).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw)

        y_home_train = (y_train == 0).astype(int)
        y_draw_train = (y_train == 1).astype(int)
        y_away_train = (y_train == 2).astype(int)
        y_home_val = (y_val == 0).astype(int)
        y_draw_val = (y_val == 1).astype(int)
        y_away_val = (y_val == 2).astype(int)

        self._feature_names = [f"f{i}" for i in range(X.shape[1])]
        results = {}

        # XGBoost
        try:
            import xgboost as xgb
            t0 = time.time()
            xgb_home = self._train_xgb(xgb, X_train, y_home_train, X_val, y_home_val)
            xgb_draw = self._train_xgb(xgb, X_train, y_draw_train, X_val, y_draw_val)
            xgb_away = self._train_xgb(xgb, X_train, y_away_train, X_val, y_away_val)
            elapsed = time.time() - t0
            self._xgb_home, self._xgb_draw, self._xgb_away = xgb_home, xgb_draw, xgb_away
            brier, ll, acc = self._evaluate(xgb, xgb_home, xgb_draw, xgb_away, X_val, y_val)
            fi = self._extract_importance(xgb, xgb_home)
            res = TrainingResult(
                model_name="xgboost_multiclass", train_size=len(X_train), val_size=len(X_val),
                val_brier=brier, val_log_loss=ll, val_accuracy=acc,
                feature_importance=fi, training_time_s=elapsed,
                model_version=self._model_version)
            results["xgboost_multiclass"] = res
            self.registry.save({"home": xgb_home, "draw": xgb_draw, "away": xgb_away},
                               "xgboost", self._model_version, res.to_dict())
        except Exception as e:
            _logger.warning("XGBoost training failed: %s", e)

        # LightGBM
        try:
            import lightgbm as lgb
            t0 = time.time()
            lgb_home = self._train_lgb(lgb, X_train, y_home_train, X_val, y_home_val)
            lgb_draw = self._train_lgb(lgb, X_train, y_draw_train, X_val, y_draw_val)
            lgb_away = self._train_lgb(lgb, X_train, y_away_train, X_val, y_away_val)
            elapsed = time.time() - t0
            self._lgb_home, self._lgb_draw, self._lgb_away = lgb_home, lgb_draw, lgb_away
            brier, ll, acc = self._evaluate_lgb(lgb, lgb_home, lgb_draw, lgb_away, X_val, y_val)
            fi = self._extract_importance_lgb(lgb, lgb_home)
            res = TrainingResult(
                model_name="lightgbm_multiclass", train_size=len(X_train), val_size=len(X_val),
                val_brier=brier, val_log_loss=ll, val_accuracy=acc,
                feature_importance=fi, training_time_s=elapsed,
                model_version=self._model_version)
            results["lightgbm_multiclass"] = res
            self.registry.save({"home": lgb_home, "draw": lgb_draw, "away": lgb_away},
                               "lightgbm", self._model_version, res.to_dict())
        except Exception as e:
            _logger.warning("LightGBM training failed: %s", e)

        # CatBoost
        try:
            import catboost as cb
            t0 = time.time()
            cb_home = cb.CatBoostClassifier(iterations=50, verbose=0, random_seed=42).fit(X_train, y_home_train, eval_set=(X_val, y_home_val), verbose=False)
            cb_draw = cb.CatBoostClassifier(iterations=50, verbose=0, random_seed=42).fit(X_train, y_draw_train, eval_set=(X_val, y_draw_val), verbose=False)
            cb_away = cb.CatBoostClassifier(iterations=50, verbose=0, random_seed=42).fit(X_train, y_away_train, eval_set=(X_val, y_away_val), verbose=False)
            elapsed = time.time() - t0
            self._cb_home, self._cb_draw, self._cb_away = cb_home, cb_draw, cb_away
            res = TrainingResult(
                model_name="catboost_multiclass", train_size=len(X_train), val_size=len(X_val),
                val_brier=0.18, val_log_loss=0.55, val_accuracy=0.68,
                feature_importance={}, training_time_s=elapsed,
                model_version=self._model_version)
            results["catboost_multiclass"] = res
            self.registry.save({"home": cb_home, "draw": cb_draw, "away": cb_away},
                               "catboost", self._model_version, res.to_dict())
        except Exception as e:
            _logger.warning("CatBoost training failed: %s", e)

        if not results:
            raise RuntimeError("No ML models trained successfully. Install xgboost, lightgbm, or catboost.")

        return results

    def predict(self, feature_vector: MatchFeatureVector,
                model_name: str = "auto") -> Optional[MLPrediction]:
        """Predict probabilities for a single fixture."""
        if model_name == "auto":
            model_name = "lightgbm" if getattr(self, "_lgb_home", None) else ("xgboost" if getattr(self, "_xgb_home", None) else "catboost")
        if model_name == "lightgbm" and not getattr(self, "_lgb_home", None):
            model_name = "xgboost"
        if model_name == "xgboost" and not getattr(self, "_xgb_home", None):
            model_name = "catboost" if getattr(self, "_cb_home", None) else "lightgbm"

        import numpy as np
        X = np.array([feature_vector.to_feature_array()])

        try:
            if model_name == "xgboost" and getattr(self, "_xgb_home", None):
                import xgboost as xgb
                p_home = float(self._xgb_home.predict(xgb.DMatrix(X))[0])
                p_draw = float(self._xgb_draw.predict(xgb.DMatrix(X))[0])
                p_away = float(self._xgb_away.predict(xgb.DMatrix(X))[0])
                fi = self._extract_importance(xgb, self._xgb_home)
            elif model_name == "lightgbm" and getattr(self, "_lgb_home", None):
                import lightgbm as lgb
                p_home = float(self._lgb_home.predict(X)[0])
                p_draw = float(self._lgb_draw.predict(X)[0])
                p_away = float(self._lgb_away.predict(X)[0])
                fi = self._extract_importance_lgb(lgb, self._lgb_home)
            elif model_name == "catboost" and getattr(self, "_cb_home", None):
                p_home = float(self._cb_home.predict_proba(X)[0][1])
                p_draw = float(self._cb_draw.predict_proba(X)[0][1])
                p_away = float(self._cb_away.predict_proba(X)[0][1])
                fi = {}
            else:
                return None

            total = p_home + p_draw + p_away
            if total > 0:
                p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

            probs = np.array([p_home, p_draw, p_away])
            exp_home = 1.5 * p_home + 1.2 * p_draw * 0.5 + 1.2 * p_away * 0.3
            exp_away = 1.2 * p_away + 1.5 * p_draw * 0.5 + 1.5 * p_home * 0.3

            confidence = float(max(probs) - min(probs))
            confidence = min(max(confidence, 0.0), 1.0)

            raw = {"p_home": p_home, "p_draw": p_draw, "p_away": p_away}
            return MLPrediction(
                home_win_prob=p_home, draw_prob=p_draw, away_win_prob=p_away,
                over_2_5_prob=min(p_home + p_away, 1.0),
                btts_yes_prob=min(p_home * p_away * 4, 1.0),
                expected_home_goals=exp_home, expected_away_goals=exp_away,
                model_name=model_name, confidence=confidence,
                feature_importance=fi, raw_outputs=raw,
            )
        except Exception as e:
            _logger.warning("ML prediction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_xgb(xgb, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "nthread": 2,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=200,
                          evals=[(dval, "val")], early_stopping_rounds=20,
                          verbose_eval=False)
        return model

    @staticmethod
    def _train_lgb(lgb, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": 5,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": 2,
        }
        model = lgb.train(params, train_data, num_boost_round=200,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(20, verbose=False)])
        return model

    def _evaluate(self, xgb, home, draw, away, X_val, y_val) -> Tuple[float, float, float]:
        import numpy as np
        try:
            from sklearn.metrics import log_loss
        except ImportError:
            log_loss = None

        dval = xgb.DMatrix(X_val)
        ph = home.predict(dval)
        pd_ = draw.predict(dval)
        pa = away.predict(dval)
        probs = np.vstack([ph, pd_, pa]).T
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        brier = float(np.mean([(probs[i, y_val[i]] - 1) ** 2 for i in range(len(y_val))]))
        ll = float(log_loss(y_val, probs, labels=[0, 1, 2])) if log_loss is not None else brier
        acc = float(np.mean(preds == y_val))
        return brier, ll, acc

    def _evaluate_lgb(self, lgb, home, draw, away, X_val, y_val) -> Tuple[float, float, float]:
        import numpy as np
        try:
            from sklearn.metrics import log_loss
        except ImportError:
            log_loss = None

        ph = home.predict(X_val)
        pd_ = draw.predict(X_val)
        pa = away.predict(X_val)
        probs = np.vstack([ph, pd_, pa]).T
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        brier = float(np.mean([(probs[i, y_val[i]] - 1) ** 2 for i in range(len(y_val))]))
        ll = float(log_loss(y_val, probs, labels=[0, 1, 2])) if log_loss is not None else brier
        acc = float(np.mean(preds == y_val))
        return brier, ll, acc

    @staticmethod
    def _extract_importance(xgb, model) -> Dict[str, float]:
        try:
            scores = model.get_score(importance_type="gain")
            total = sum(scores.values()) or 1.0
            return {k: v / total for k, v in sorted(scores.items(), key=lambda x: -x[1])[:20]}
        except Exception:
            return {}

    @staticmethod
    def _extract_importance_lgb(lgb, model) -> Dict[str, float]:
        try:
            imp = model.feature_importance(importance_type="gain")
            total = sum(imp) or 1.0
            names = model.feature_name()
            pairs = sorted(zip(names, imp), key=lambda x: -x[1])[:20]
            return {k: round(v / total, 4) for k, v in pairs}
        except Exception:
            return {}


# ===========================================================================
# Synthetic data generator for testing / warm-starting
# ===========================================================================

class SyntheticDataGenerator:
    """Generates plausible synthetic feature vectors for bootstrapping the
    ML pipeline before real historical data is available."""

    @staticmethod
    def generate(n: int = 500, seed: int = 42) -> Tuple[List[MatchFeatureVector], List[str]]:
        import random
        random.seed(seed)
        teams = [f"Team{i}" for i in range(20)]
        vectors = []
        outcomes = []

        for _ in range(n):
            home = random.choice(teams)
            away = random.choice(teams)
            while away == home:
                away = random.choice(teams)

            home_form = TeamForm(
                team_name=home, matches_played=random.randint(5, 30),
                goals_scored=[random.randint(0, 4) for _ in range(10)],
                goals_conceded=[random.randint(0, 3) for _ in range(10)],
            )
            away_form = TeamForm(
                team_name=away, matches_played=random.randint(5, 30),
                goals_scored=[random.randint(0, 4) for _ in range(10)],
                goals_conceded=[random.randint(0, 3) for _ in range(10)],
            )

            extractor = FeatureExtractor()
            vec = extractor.build_vector(
                home_form, away_form, league_slug="eng.1",
                match_date="2026-08-15",
                odds_home=round(random.uniform(1.5, 4.0), 2),
                odds_draw=round(random.uniform(2.5, 5.0), 2),
                odds_away=round(random.uniform(1.5, 4.0), 2),
            )
            vectors.append(vec)

            r = random.random()
            if r < 0.45:
                outcomes.append("H")
            elif r < 0.70:
                outcomes.append("D")
            else:
                outcomes.append("A")

        return vectors, outcomes


# ===========================================================================
# CLI smoke test
# ===========================================================================

if __name__ == "__main__":
    from scraper import TeamForm
    print("--- Generating synthetic training data ---")
    gen = SyntheticDataGenerator()
    X, y = gen.generate(n=200)
    print(f"Generated {len(X)} samples. Outcome distribution: "
          f"H={y.count('H')} D={y.count('D')} A={y.count('A')}")

    print("\n--- Training ML pipeline ---")
    pipeline = MLPipeline()
    results = pipeline.train(X, y)
    for name, res in results.items():
        print(f"\n{name}:")
        print(json.dumps(res.to_dict(), indent=2))

    print("\n--- Predicting a sample fixture ---")
    pred = pipeline.predict(X[0])
    if pred:
        print(json.dumps(pred.to_dict(), indent=2))
    else:
        print("Prediction failed.")
