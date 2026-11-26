"""Step 14: risk models P(e = 1 | features) and their calibration.

Two nested feature sets (H1 compares them):
  base = u, g, tau, q              (uncertainty-only comparator)
  full = u, g, v, rho, tau, q      (adds verifier verdicts and residuals)
Each is fitted as an L2 logistic regression and as LightGBM on synthetic train;
hyperparameters are chosen on dev by log loss; probabilities are then calibrated
on dev (Platt scaling for logistic, isotonic for LightGBM). Calibration fitted on
dev is optimistic on dev itself, so syn_test and adv_test are the honest scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GROUPS = {"base": ("u_", "g_", "tau_", "q_"), "full": ("u_", "g_", "v_", "rho_", "tau_", "q_")}
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
LGBM_GRID = tuple({"num_leaves": leaves, "min_child_samples": child}
                  for leaves in (7, 15, 31) for child in (20, 50))
EPS = 1e-6


def columns(table, feature_set: str) -> list[str]:
    prefixes = GROUPS[feature_set]
    return [c for c in table.columns if c.startswith(prefixes)]


def ece(y, p, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def metrics(y, p) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    both = len(set(y)) == 2
    return {"n": int(len(y)), "positives": int(y.sum()), "ece": ece(y, p), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
            "auprc": float(average_precision_score(y, p)) if both else float("nan")}


@dataclass
class RiskModel:
    """A fitted, calibrated model: predict(X) returns calibrated P(error)."""
    name: str
    feature_set: str
    kind: str
    columns: list[str]
    estimator: object
    calibrator: object
    params: dict = field(default_factory=dict)

    def raw(self, table) -> np.ndarray:
        X = table[self.columns].to_numpy(float)
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, table) -> np.ndarray:
        raw = np.clip(self.raw(table), EPS, 1 - EPS)
        if self.kind == "logistic":  # Platt scaling on the logit
            logit = np.log(raw / (1 - raw)).reshape(-1, 1)
            return self.calibrator.predict_proba(logit)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


def _logistic(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _lgbm(params: dict, seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1, **params)


def fit(train, dev, feature_set: str, kind: str, seed: int = 20261026) -> RiskModel:
    """Fit on train, pick hyperparameters on dev by log loss, calibrate on dev."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    cols = columns(train, feature_set)
    X_tr, y_tr = train[cols].to_numpy(float), train["e"].to_numpy(int)
    X_dev, y_dev = dev[cols].to_numpy(float), dev["e"].to_numpy(int)
    if len(set(y_tr)) < 2:
        raise ValueError("train split has a single class: no errors to learn from")
    grid = [{"C": C} for C in LOGISTIC_C] if kind == "logistic" else list(LGBM_GRID)
    best, best_loss, best_params = None, float("inf"), None
    for params in grid:
        model = _logistic(params["C"]) if kind == "logistic" else _lgbm(params, seed)
        model.fit(X_tr, y_tr)
        loss = log_loss(y_dev, np.clip(model.predict_proba(X_dev)[:, 1], EPS, 1 - EPS), labels=[0, 1])
        if loss < best_loss:
            best, best_loss, best_params = model, loss, params
    raw_dev = np.clip(best.predict_proba(X_dev)[:, 1], EPS, 1 - EPS)
    if kind == "logistic":
        calibrator = LogisticRegression(C=1e6, max_iter=1000)
        calibrator.fit(np.log(raw_dev / (1 - raw_dev)).reshape(-1, 1), y_dev)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_dev, y_dev)
    return RiskModel(name=f"{feature_set}_{kind}", feature_set=feature_set, kind=kind, columns=cols,
                     estimator=best, calibrator=calibrator, params=best_params)


def fit_all(table, seed: int = 20261026) -> dict[str, RiskModel]:
    train, dev = table[table.split == "train"], table[table.split == "dev"]
    return {f"{fs}_{kind}": fit(train, dev, fs, kind, seed) for fs in ("base", "full")
            for kind in ("logistic", "lightgbm")}


def evaluate(models: dict[str, RiskModel], table, splits=("dev", "syn_test", "adv_test")) -> dict:
    """{split: {model: metrics}} using calibrated probabilities."""
    result = {}
    for split in splits:
        part = table[table.split == split]
        if len(part):
            result[split] = {name: metrics(part["e"], model.predict(part)) for name, model in models.items()}
    return result


def report(results: dict, model_key: str, models: dict[str, RiskModel]) -> str:
    lines = [f"# Risk models ({model_key})", "",
             "Fitted on synthetic train; hyperparameters and calibration on dev (Platt for logistic,",
             "isotonic for LightGBM). Dev scores are optimistic (calibration fitted there);",
             "syn_test and adv_test are honest. base = uncertainty-only comparator (H1); full adds",
             "verifier verdicts and residuals. ECE uses 15 equal-width bins.", ""]
    for split, by_model in results.items():
        n = next(iter(by_model.values()))
        lines += [f"## {split} ({n['n']:,} fields, {n['positives']:,} errors)", "",
                  "| model | ECE | Brier | log loss | AUROC | AUPRC |", "| --- | --- | --- | --- | --- | --- |"]
        for name, m in by_model.items():
            lines.append(f"| {name} | {m['ece']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | "
                         f"{m['auroc']:.3f} | {m['auprc']:.3f} |")
        lines.append("")
    lines += ["Chosen hyperparameters: " + "; ".join(f"{n}: {m.params}" for n, m in models.items()), ""]
    return "\n".join(lines)
