"""Audit cached study artifacts and write results/audit/report.md.

Optional source evidence belongs in cache/audit_inputs/<collection>/:
docs.jsonl, gold.jsonl and text/<doc_id>.txt. The audit uses only generated
collections and never follows text_path from a manifest. Saved joblib files
must be trusted local Step 14 outputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd

from trustswap.decide.crc import _document_grid, calibrate_lambda, floor
from trustswap.decide.policy import COSTS_FILE, escalate, field_costs, lambda_grid, load_costs
from trustswap.extract.evaluate import load_gold, outcome
from trustswap.score.features import META, SEVERITY
from trustswap.score.risk import metrics
from trustswap.synth.render import trade_fields
from trustswap.verify.evaluate import verifier_hash

if __package__:
    from . import build_demo
    from .crc import ALPHAS
else:
    import build_demo
    from crc import ALPHAS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("frontier", "open_weights")
SPLITS = {"train": "synthetic", "dev": "synthetic", "syn_test": "synthetic", "adv_test": "adversarial"}
VARIANTS = ("base_logistic", "full_logistic", "base_lightgbm", "full_lightgbm")
KEYS = ["doc_id", "field"]
FOLDS = 1000  # Step 16 report default.
ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def check_freeze(path):
    record = json_file(path)
    require(record["verifier_hash"] == verifier_hash(), "Verifier hash differs from the recorded hash.")
    require(record["status"] == "frozen", "Verifier hash matches, but the verifier record is still draft.")
    return "Verifier source hash matches the frozen record."


def read_table(path, *, required=True):
    if not required and not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow", filters=[("split", "in", tuple(SPLITS))])


def checked_inventory(*tables):
    inventory = {}
    for table in tables:
        if table.empty:
            continue
        require(table["split"].map(SPLITS).eq(table["collection"]).all(),
                "A split contains rows from the wrong collection.")
        require(not table.duplicated(KEYS).any(), "Repeated document and field keys in a cache table.")
        for doc, split in table[["doc_id", "split"]].drop_duplicates().itertuples(index=False, name=None):
            require(isinstance(doc, str) and ID.fullmatch(doc), "Invalid document identifier.")
            require(inventory.setdefault(doc, split) == split, f"{doc} occurs in more than one split.")
    require(inventory, "No generated documents are identified by the cache tables.")
    return inventory


def cached_inventory(snapshot):
    inventory = {}
    for collection in sorted(set(SPLITS.values())):
        path = snapshot / collection / "docs.jsonl"
        if not path.is_file():
            continue
        for doc in json_lines(path):
            doc_id, split = doc["doc_id"], doc["split"]
            require(doc["collection"] == collection == SPLITS.get(split) and ID.fullmatch(doc_id),
                    f"Wrong collection, split or document ID in {path}.")
            require(inventory.setdefault(doc_id, split) == split, f"{doc_id} occurs in more than one split.")
    return inventory


def check_extractions(cache, model, inventory, required_docs):
    samples, counts = {}, defaultdict(lambda: [0, 0, 0, 0, 0])
    for doc, split in sorted(inventory.items()):
        path = cache / "extract" / model / f"{doc}.jsonl"
        if not path.is_file():
            continue
        rows = json_lines(path)
        ids = [sample["k"] for sample in rows]
        require(ids.count(0) == 1 and len(ids) == len(set(ids)), f"Invalid sample indices for {doc}.")
        require(all(sample.get("doc_id", doc) == doc for sample in rows), f"Wrong document in {path}.")
        samples[doc] = next(row for row in rows if row["k"] == 0)
        tally = counts[SPLITS[split]]
        tally[0] += 1
        tally[1] += len(rows)
        tally[2] += sum(row.get("schema_valid") is True for row in rows)
        tally[3] += sum(row.get("finish_reason") == "length" for row in rows)
        tally[4] += sum(row.get("retried_after_length") is True for row in rows)
    require(samples, "No extraction files for the identified documents.")
    require(required_docs <= samples.keys(), "Some feature or risk rows lack a cached k=0 extraction.")
    summary = " ".join(f"{collection}: {docs} documents, {valid}/{n} schema-valid samples "
                       f"({valid/n:.1%}), {truncated} final truncations, {retried} retried after truncation."
                       for collection, (docs, n, valid, truncated, retried) in sorted(counts.items()))
    return samples, summary


def check_injections(cache, model, inventory):
    counts = defaultdict(lambda: [0, 0])
    for doc, split in sorted(inventory.items()):
        path = cache / "verify_injected" / model / f"{doc}.jsonl"
        if not path.is_file():
            continue
        for row in json_lines(path):
            require(row.get("doc_id", doc) == doc, f"Wrong document in {path}.")
            kind = row["injection"]["kind"]
            fields = row["injection"]["fields"]
            detected = any(verdicts.get(field, {}).get("verdict") in {"HARD", "SOFT"}
                           for verdicts in row["rules"].values() for field in fields)
            counts[SPLITS[split], kind][0] += int(detected)
            counts[SPLITS[split], kind][1] += 1
    require(counts, "No cached injected-error results for identified documents.")
    return " ".join(f"{collection}, {kind}: {hit}/{n} caught ({hit/n:.1%})."
                    for (collection, kind), (hit, n) in sorted(counts.items()))


def check_features(features, samples, snapshot):
    require(len(features), "Feature cache is missing or empty.")
    columns = [col for col in features if col not in META and col != "e"]
    require(columns and np.isfinite(features[columns].to_numpy(float)).all(), "Feature columns contain NaN or infinity.")
    require(features["e"].isin([0, 1]).all(), "Feature labels must be binary.")
    summaries = []
    sampled = features.sort_values(KEYS).sample(n=min(50, len(features)), random_state=0)
    for collection, chosen in sampled.groupby("collection"):
        gold_path = snapshot / collection / "gold.jsonl"
        records = load_gold(gold_path)
        require(all(record.collection == collection for record in records.values()),
                f"{gold_path} contains another collection.")
        for row in chosen.itertuples(index=False):
            require(row.doc_id in records and row.doc_id in samples, f"Missing gold or extraction for {row.doc_id}.")
            gold = trade_fields(records[row.doc_id].trade)
            require(row.field in gold, f"No gold term for {row.doc_id} / {row.field}.")
            sample = samples[row.doc_id]
            predicted = sample.get("fields", {}).get(row.field, {}).get("value") if sample.get("schema_valid") else None
            expected = int(outcome(row.field, predicted, gold[row.field]) != "match")
            require(expected == row.e, f"Gold label disagrees for {row.doc_id} / {row.field}.")
        summaries.append(f"{collection}: {len(chosen)} sampled labels match gold.")
    return "All features are finite. " + " ".join(summaries)


def check_risk(features, risk, results, model):
    require(len(features) and len(risk), "Both feature and risk caches are required.")
    keys = ["collection", "split", *KEYS]
    labels = risk.set_index(keys).reindex(pd.MultiIndex.from_frame(features[keys]))
    require(len(features) == len(risk) and labels["e"].eq(features["e"].to_numpy()).all(),
            "Feature and risk keys or error labels differ.")
    root = results / "models" / model
    reported = json_file(root / "metrics.json")
    require(all(f"p_{variant}" in risk for variant in VARIANTS[:2]), "Base and full logistic predictions are required.")
    variants = [name for name in VARIANTS if f"p_{name}" in risk]
    summaries = []
    for variant in variants:
        scores = np.asarray(joblib.load(root / f"{variant}.joblib").predict(features), dtype=float)
        require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)),
                f"{variant} emitted invalid probabilities.")
        require(np.allclose(scores, labels[f"p_{variant}"], atol=1e-6, rtol=0),
                f"{variant} re-scores do not match cached probabilities.")
        for (collection, split), rows in features.assign(_prediction=scores).groupby(["collection", "split"]):
            if split == "train":
                continue
            found = metrics(rows["e"], rows["_prediction"])
            expected = reported[split][variant]
            require(found["n"] == expected["n"] and found["positives"] == expected["positives"],
                    f"{collection} {split} {variant}: reported counts differ.")
            require(np.isclose(found["auroc"], expected["auroc"], atol=1e-6, rtol=0, equal_nan=True),
                    f"{collection} {split} {variant}: reported AUROC differs.")
            if "ece" in expected:
                require(np.isclose(found["ece"], expected["ece"], atol=1e-6, rtol=0),
                        f"{collection} {split} {variant}: reported ECE differs.")
            auroc = f"{found['auroc']:.6f}" if np.isfinite(found["auroc"]) else "undefined (one label class)"
            summaries.append(f"{collection} {split} {variant}: AUROC {auroc}, ECE {found['ece']:.6f}.")
    require(summaries, "There are no evaluation rows to compare with published metrics.")
    return " ".join(summaries)


def check_decisions(risk, costs, results, model):
    require(len(risk), "Risk cache is missing or empty.")
    runs = []
    for split in SPLITS:
        for variant in VARIANTS:
            for path in results.glob(f"{model}_{variant}_lambda*_{split}/decisions.jsonl"):
                prefix = f"{model}_{variant}_lambda"
                rows = pd.DataFrame(json_lines(path))
                require(not rows.empty and rows["lambda"].nunique(dropna=False) == 1,
                        f"Inconsistent decision thresholds in {path}.")
                lam = float(rows["lambda"].iloc[0])
                # The run name formats lambda with six significant digits.
                require(np.isfinite(lam) and lam >= 0 and path.parent.name == f"{prefix}{lam:g}_{split}",
                        f"Invalid threshold or run name in {path}.")
                table = risk.loc[risk["split"].eq(split)].copy()
                c, r = field_costs(table, costs)
                p = table[f"p_{variant}"].to_numpy(float)
                table["c_bar"] = c
                table["p_hat"] = p
                table["lambda"] = lam
                table["action"] = np.where(escalate(p, c, r, lam), "ESCALATE", "ACCEPT")
                columns = ["p_hat", "c_bar", "lambda", "action", "e"]
                require(len(rows) == len(table), f"Wrong decision count in {path}.")
                joined = table[KEYS + columns].merge(rows[KEYS + columns], on=KEYS, suffixes=("_exp", "_log"),
                                                     validate="one_to_one")
                require(len(joined) == len(table), f"Wrong decision keys in {path}.")
                for column in columns:
                    expected, logged = joined[f"{column}_exp"], joined[f"{column}_log"]
                    same = (expected == logged).all() if column in ("action", "e") else np.allclose(
                        expected, logged, atol=1e-12, rtol=1e-12)
                    require(same, f"{column} disagrees with cached risks and policy in {path}.")
                runs.append(f"{SPLITS[split]} {split}: {len(rows)} {variant} decisions at λ={lam:g}.")
    require(runs, "No Step 15 decision logs for the allowed splits.")
    return " ".join(runs)


def check_crc(risk, costs, results, model):
    require(len(risk), "Risk cache is missing or empty.")
    path = results / f"crc_{model}.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = re.fullmatch(r"# Conformal risk control \(([^,]+), (\w+), (field|trade) mode\)",
                          lines[0] if lines else "")
    require(header is not None and header[1] == model, "CRC report header does not identify this model.")
    _, variant, mode = header.groups()
    require(variant in VARIANTS, "CRC report has an unsupported risk variant.")
    observed = [line.strip("| ").split(" | ") for line in lines if re.match(r"^\| [0-9]", line)]
    require(len(observed) == len(ALPHAS), "CRC report is missing one or more target levels.")
    table = risk.loc[risk["split"].eq("syn_test")]
    grid = lambda_grid()
    ids, losses, automated, totals = _document_grid(table, f"p_{variant}", costs, mode, grid)
    n = len(ids) // 2
    require(n, "CRC requires at least two synthetic test documents.")
    notes = []
    for alpha, row in zip(ALPHAS, observed):
        rng = np.random.default_rng(0)
        thresholds, held_losses, coverage = [], [], []
        certified_count = 0
        for _ in range(FOLDS):
            order = rng.permutation(len(ids))
            cal, test = order[:n], order[n:]
            bounds = (np.sum(losses[cal], axis=0) + 1) / (n + 1)
            first_failure = np.flatnonzero(bounds > alpha)
            prefix = int(first_failure[0]) if len(first_failure) else len(grid)
            col = max(0, prefix - 1)
            chosen, certified = calibrate_lambda(losses[cal], grid, alpha)
            require(chosen == grid[col] and certified == (prefix > 0),
                    f"CRC did not choose the largest passing prefix at alpha {alpha:g}.")
            if certified:
                require(np.all(bounds[:col + 1] <= alpha), f"CRC bound failed at alpha {alpha:g}.")
            certified_count += int(certified)
            thresholds.append(chosen)
            held_losses.append(losses[test, col].mean())
            coverage.append(automated[test, col].sum() / totals[test].sum())
        mean_loss = np.mean(held_losses)
        se = np.std(held_losses, ddof=1) / np.sqrt(FOLDS)
        expected = [
            f"{alpha:g}", f"{floor(n):.6g}", f"{np.mean(thresholds):.6g}", f"{mean_loss:.6g} ± {se:.6g}",
            f"{np.mean(coverage):.2%}", f"{certified_count/FOLDS:.2%}", "Pass" if mean_loss <= alpha else "Fail",
        ]
        require(row == expected, f"Cached CRC report differs from replay at alpha {alpha:g}.")
        notes.append(f"α={alpha:g}: {certified_count}/{FOLDS} certified, with fallback verified on the others.")
    return " ".join(notes)


def check_demo(risk, samples, costs, cache, snapshot, results, model):
    require(len(risk), "Risk cache is missing or empty.")
    documents = build_demo.load_documents(snapshot, cache, model, costs, 30, cached_text_only=True)
    require(documents, "No buildable demo documents in cache/audit_inputs.")
    html = build_demo.render_page(documents, costs.B)
    payload = json.loads(re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', html, re.S)[1])
    require(payload["bound"] == costs.B and not re.search(r"https?://", html, re.I) and "default-src 'none'" in html,
            "Demo bound or offline restrictions do not match the build.")
    indexed = risk.set_index(KEYS)
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for doc in payload["documents"]:
        subset = indexed.loc[doc["id"]]
        require({field["field"] for field in doc["fields"]} == set(subset.index)
                and len(doc["fields"]) == len(subset), f"Missing or repeated demo fields for {doc['id']}.")
        verify = json_file(cache / "verify" / model / f"{doc['id']}.json")["rules"]
        sample = samples[doc["id"]]
        original = sample.get("fields", {}) if sample.get("schema_valid", True) else {}
        for field in doc["fields"]:
            name = field["field"]
            row = subset.loc[name]
            term = name.rsplit(".", 1)[-1]
            truth = (row.p_full_logistic, costs.c_bar(term), costs.r(term), row.e, SEVERITY[term])
            require(tuple(field[k] for k in ("p", "c_bar", "r", "e", "severity")) == truth,
                    f"Demo embedded numbers differ for {doc['id']} / {name}.")
            collection = "adversarial" if doc["category"] == "Adversarial" else "synthetic"
            require(collection == row.collection and row.split in build_demo.COLLECTIONS[collection],
                    f"Demo collection differs for {doc['id']}.")
            item = original.get(name, {})
            raw, value = item.get("raw_value"), item.get("value")
            require(field["raw_value"] == raw and field["value"] == (raw if value is None else value),
                    f"Demo extraction differs for {doc['id']} / {name}.")
            expected_rules = {rule for rule, fields in verify.items() if rule in build_demo.RULE_TEXT
                              and fields.get(name, {}).get("verdict") in {"HARD", "SOFT"}}
            require(set(field["rules_fired"]) == expected_rules, f"Demo rules differ for {doc['id']} / {name}.")
            require(field["span"] == item.get("span"), f"Demo offsets differ for {doc['id']} / {name}.")
            if field["span"] is not None:
                start, end = field["span"]
                # Step 10 locates evidence case-insensitively and tolerates spacing.
                normalize = lambda text: " ".join(text.casefold().split())
                require(isinstance(raw, str) and normalize(doc["text"][start:end]) == normalize(raw),
                        f"Demo span differs from the extraction for {doc['id']} / {name}.")
            require(0 <= truth[0] <= 1 and truth[3] in (0, 1), f"Invalid demo risk or label for {doc['id']}.")
            review = bool(escalate(truth[0], truth[1], truth[2], 1.0))
            tally = counts[collection]
            tally[0] += not review
            tally[1] += review
            tally[2] += review * int(truth[3])
            tally[3] += (not review) * int(truth[3])
    out = results / "audit" / model / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return "Rebuilt and checked embedded evidence at " + str(out) + ". " + " ".join(
        f"{collection} at λ=1: {accepted} accepted, {reviewed} reviewed, {caught} caught, {missed} missed."
        for collection, (accepted, reviewed, caught, missed) in sorted(counts.items())
    )


def audit(cache, results, costs_path, freeze, models):
    results_out = []
    costs = load_costs(costs_path)
    snapshot = cache / "audit_inputs"

    def check(name, operation):
        try:
            result = operation()
        except Exception as error:
            results_out.append((False, name, " ".join(str(error).splitlines())))
            return None
        results_out.append((True, name, result))
        return result

    check("Verifier freeze", lambda: check_freeze(freeze))
    for model in models:
        features, risk, inventory, required, samples = pd.DataFrame(), pd.DataFrame(), {}, set(), {}

        def collections():
            nonlocal features, risk, inventory, required
            candidate_features = read_table(cache / "features" / f"{model}.parquet", required=False)
            candidate_risk = read_table(cache / "risk" / f"{model}.parquet", required=False)
            from_snapshot = cached_inventory(snapshot)
            if not candidate_features.empty or not candidate_risk.empty:
                table_inventory = checked_inventory(candidate_features, candidate_risk)
                require(all(from_snapshot.get(doc, split) == split for doc, split in table_inventory.items()),
                        "Cached manifests and tables disagree on document splits.")
            else:
                table_inventory = {}
            features, risk = candidate_features, candidate_risk
            required = set(table_inventory)
            inventory = {**from_snapshot, **table_inventory}
            require(required, "No generated feature or risk rows were available to audit for collection separation.")
            return "Allowed split and collection pairs and unique document-field keys checked."

        def extraction():
            nonlocal samples
            samples, summary = check_extractions(cache, model, inventory, required)
            return summary

        checks = [
            ("Collections", collections),
            ("Extraction", extraction),
            ("Injected-error detection", lambda: check_injections(cache, model, inventory)),
            ("Features and gold", lambda: check_features(features, samples, snapshot)),
            ("Risk and calibration", lambda: check_risk(features, risk, results, model)),
            ("Decision logs", lambda: check_decisions(risk, costs, results, model)),
            ("Conformal risk control", lambda: check_crc(risk, costs, results, model)),
            ("Demo rebuild", lambda: check_demo(risk, samples, costs, cache, snapshot, results, model)),
        ]
        for name, operation in checks:
            check(f"{model} · {name}", operation)

    out = results / "audit" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pipeline audit", "",
        "This report checks cached evidence. A missing input is a failure to verify that check.", "",
    ]
    lines += [f"- **{'PASS' if passed else 'FAIL'}** — {name}. {detail}" for passed, name, detail in results_out]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, all(passed for passed, _, _ in results_out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--freeze", type=Path, default=ROOT / "prereg" / "verifier-v1.0.json")
    parser.add_argument("--model", choices=MODELS, action="append")
    args = parser.parse_args(argv)
    out, passed = audit(args.cache, args.results, args.costs, args.freeze, args.model or MODELS)
    print(f"Wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
