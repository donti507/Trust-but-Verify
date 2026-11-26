"""Step 15: cost model and accept/escalate decision rule.

Per field: ESCALATE if p_hat * c_bar > lambda * r, else ACCEPT, where c_bar is the
expected cost of accepting a wrong value (configs/costs.yaml, "likely") and r the
review cost. Larger lambda automates more.

Two modes:
* field: only escalated fields are reviewed; accepted wrong fields cost c.
* trade: a document goes straight through only if every field is accepted;
  otherwise the whole document is reviewed (no loss, review cost for every field).

Document loss at lambda: L_d = min(B, sum over accepted fields of e * c) / B, in
[0, 1]. It never decreases as lambda grows, because a larger lambda only turns
escalations into acceptances.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COSTS_FILE = ROOT / "configs" / "costs.yaml"
SEVERITY_ORDER = ("critical", "major", "minor")


@dataclass(frozen=True)
class Costs:
    severity: dict              # {severity: {low, likely, high}}
    review: float               # default review cost r
    B: float                    # loss bound per document
    fields: dict                # per field-name overrides {name: {likely, r}}
    status: str = "placeholder"

    def c_bar(self, field_name: str) -> float:
        from trustswap.score.features import SEVERITY

        override = self.fields.get(field_name, {})
        return float(override.get("likely", self.severity[SEVERITY[field_name]]["likely"]))

    def r(self, field_name: str) -> float:
        return float(self.fields.get(field_name, {}).get("r", self.review))

    def scaled(self, factor: float) -> "Costs":
        """Sensitivity: every accept cost multiplied by factor (review cost unchanged)."""
        severity = {s: {k: v * factor for k, v in row.items()} for s, row in self.severity.items()}
        fields = {f: {**o, **({"likely": o["likely"] * factor} if "likely" in o else {})} for f, o in self.fields.items()}
        return replace(self, severity=severity, fields=fields)

    def reordered(self, order: tuple[str, str, str]) -> "Costs":
        """Sensitivity: give severity classes each other's costs, e.g. ("major", "critical", "minor")
        makes critical fields cost what major fields cost and vice versa."""
        rows = [self.severity[s] for s in SEVERITY_ORDER]
        return replace(self, severity=dict(zip(order, rows)))


def load_costs(path: Path = COSTS_FILE) -> Costs:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Costs(severity=data["severity"], review=float(data["review_cost"]["default"]), B=float(data["B"]),
                 fields=data.get("fields") or {}, status=data.get("status", "placeholder"))


def field_costs(table, costs: Costs):
    """Arrays c_bar and r aligned with the table's rows."""
    names = table["field"].str.rsplit(".", n=1).str[-1]
    c = names.map(costs.c_bar).to_numpy(float)
    r = names.map(costs.r).to_numpy(float)
    return c, r


def escalate(p, c, r, lam: float) -> np.ndarray:
    """True where the field goes to a human: p * c > lambda * r."""
    return np.asarray(p, float) * np.asarray(c, float) > lam * np.asarray(r, float)


def evaluate_lambda(table, p_col: str, lam: float, costs: Costs, mode: str = "field") -> dict:
    """Automation, review cost and mean normalised document loss at one lambda."""
    c, r = field_costs(table, costs)
    esc = escalate(table[p_col].to_numpy(float), c, r, lam)
    e = table["e"].to_numpy(int)
    frame = table[["doc_id"]].copy()
    frame["esc"], frame["err_cost"], frame["r"] = esc, e * c, r
    if mode == "field":
        frame["loss"] = np.where(esc, 0.0, frame["err_cost"])
        frame["review"] = np.where(esc, frame["r"], 0.0)
        per_doc = frame.groupby("doc_id").agg(loss=("loss", "sum"), review=("review", "sum"), esc=("esc", "sum"))
        automated = 1.0 - esc.mean()
    elif mode == "trade":
        per_doc = frame.groupby("doc_id").agg(err=("err_cost", "sum"), any_esc=("esc", "any"), r=("r", "sum"))
        per_doc["loss"] = np.where(per_doc["any_esc"], 0.0, per_doc["err"])
        per_doc["review"] = np.where(per_doc["any_esc"], per_doc["r"], 0.0)
        automated = 1.0 - per_doc["any_esc"].mean()
    else:
        raise ValueError("mode must be 'field' or 'trade'")
    doc_loss = np.minimum(costs.B, per_doc["loss"]) / costs.B
    return {"lambda": float(lam), "mode": mode, "automated": float(automated),
            "mean_doc_loss": float(doc_loss.mean()), "max_doc_loss": float(doc_loss.max()),
            "review_cost_per_doc": float(per_doc["review"].mean()),
            "errors_accepted": int(((~esc) & (e == 1)).sum()), "errors_total": int(e.sum()), "n_docs": int(len(per_doc))}


def lambda_grid(n: int = 200, low: float = 1e-3, high: float = 1e3) -> np.ndarray:
    """Runbook grid: 0 plus n log-spaced values from 1e-3 to 1e3."""
    return np.concatenate([[0.0], np.logspace(np.log10(low), np.log10(high), n)])


def frontier(table, p_col: str, costs: Costs, mode: str = "field", grid=None) -> list[dict]:
    return [evaluate_lambda(table, p_col, lam, costs, mode) for lam in (lambda_grid() if grid is None else grid)]
