"""
Clean raw fact tables — detect, report, and fix quality issues.

Pipeline steps:
  1. Detect & deduplicate duplicate shipment IDs (~1.5%)
  2. Normalize SKU typos via fuzzy matching (RECOVER, don't drop) (~3%)
  3. Fix negative / null quantities (~1%)
  4. Flag phantom shipments (non-existent store_id) (handful)
  5. Flag incomplete loads (qty_received < qty_ordered) (~6%)
  6. Flag late deliveries (actual > promised) (~9%)

Principle: typos are recovered via normalization, not deleted.
Target: retain ~98% of data.

Produces:
  - Cleaned fact tables (Parquet)
  - Audit report (JSON) with per-issue-type counts and actions taken

All randomness uses seed 42 for full reproducibility.
Output: data-pipeline/output/cleaned/ and data-pipeline/output/audit/
"""

import pandas as pd
from pathlib import Path
from collections import Counter

CLEANED_DIR = Path(__file__).parent.parent / "output" / "cleaned"
AUDIT_DIR = Path(__file__).parent.parent / "output" / "audit"


def _normalize_sku(value) -> str:
    """
    Recover a SKU code from casing / OCR-style typos back to canonical form.
    Canonical SKUs are 'SKU-0NN' (uppercase, digits in the tail), so after
    upper-casing we map the classic look-alikes: O→0, I→1, L→1. A clean code
    passes through unchanged (it contains no O/I/L).
    """
    s = str(value).strip().upper()
    return s.replace("O", "0").replace("I", "1").replace("L", "1")


def run_cleaning_pipeline(
    facts: dict[str, pd.DataFrame],
    dimensions: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Clean the raw shipments fact table: detect, report and resolve each injected
    data-quality issue, then return (cleaned_facts, audit_report).

    Resolution philosophy (DATA_SPEC): recover what we can (typos, sign errors),
    drop only what is genuinely unusable (duplicates, phantom stores, missing
    receipts). Late / incomplete are real service events — flag, never drop.
    Demand and inventory carry no injected issues and pass through unchanged.

    Deterministic: cleaning involves no randomness, so it is fully reproducible.
    """
    raw = facts["fact_shipments"]
    raw_n = len(raw)
    df = raw.copy()

    valid_skus = set(dimensions["dim_products"]["sku"])
    valid_stores = set(dimensions["dim_stores"]["store_id"])

    steps: list[dict] = []

    # 1 ── Recover SKU typos (normalize, don't drop) ───────────────────────
    typo_mask = ~df["sku"].isin(valid_skus)
    n_typo = int(typo_mask.sum())
    df["sku"] = df["sku"].map(_normalize_sku)
    n_unrecovered = int((~df["sku"].isin(valid_skus)).sum())
    steps.append({
        "issue": "sku_typo",
        "label": "SKU typos — casing / O-vs-0 / l-vs-1",
        "teaches": "normalization",
        "detected": n_typo,
        "action": "recovered",
        "rows_dropped": 0,
        "resolution": "Standardized to canonical SKU (uppercase; O→0, I/L→1). Rows retained.",
        "unrecovered": n_unrecovered,
    })

    # 2 ── Remove duplicate shipment lines (same shipment_id + SKU) ─────────
    #     (SKUs are normalized first so a typo'd copy still matches its original.)
    dup_mask = df.duplicated(subset=["shipment_id", "sku"], keep="first")
    n_dup = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()
    steps.append({
        "issue": "duplicate_id",
        "label": "Duplicate shipment lines",
        "teaches": "de-duplication",
        "detected": n_dup,
        "action": "removed",
        "rows_dropped": n_dup,
        "resolution": "Dropped exact duplicate (shipment_id, SKU) lines; kept first occurrence.",
    })

    # 3 ── Remove phantom shipments (store not in master) ───────────────────
    phantom_mask = ~df["store_id"].isin(valid_stores)
    n_phantom = int(phantom_mask.sum())
    df = df.loc[~phantom_mask].copy()
    steps.append({
        "issue": "phantom_store",
        "label": "Phantom shipments — unknown store",
        "teaches": "referential integrity",
        "detected": n_phantom,
        "action": "removed",
        "rows_dropped": n_phantom,
        "resolution": "Dropped — store_id absent from the store master.",
    })

    # 4 ── Recover negative quantities (sign error → absolute value) ────────
    neg_mask = (df["qty_ordered"] < 0) | (df["qty_received"] < 0)
    n_neg = int(neg_mask.sum())
    df["qty_ordered"] = df["qty_ordered"].abs()
    df["qty_received"] = df["qty_received"].abs()  # NaN stays NaN
    steps.append({
        "issue": "negative_qty",
        "label": "Negative quantities",
        "teaches": "glitch handling",
        "detected": n_neg,
        "action": "recovered",
        "rows_dropped": 0,
        "resolution": "Sign error corrected to absolute value. Rows retained.",
    })

    # 5 ── Drop rows with missing received quantity (unrecoverable) ─────────
    null_mask = df["qty_received"].isna()
    n_null = int(null_mask.sum())
    df = df.loc[~null_mask].copy()
    steps.append({
        "issue": "null_qty",
        "label": "Missing received quantity",
        "teaches": "glitch handling",
        "detected": n_null,
        "action": "removed",
        "rows_dropped": n_null,
        "resolution": "Dropped — received quantity missing and cannot be honestly imputed.",
    })

    # Quantities are now clean, non-null integers (pin width for cross-platform repro)
    df["qty_ordered"] = df["qty_ordered"].astype("int64")
    df["qty_received"] = df["qty_received"].astype("int64")

    # 6 ── Flag incomplete loads (keep — a real service event) ──────────────
    df["is_incomplete"] = df["qty_received"] < df["qty_ordered"]
    n_incomplete = int(df["is_incomplete"].sum())
    steps.append({
        "issue": "incomplete_load",
        "label": "Incomplete loads — under-delivery",
        "teaches": "exception investigation",
        "detected": n_incomplete,
        "action": "flagged",
        "rows_dropped": 0,
        "resolution": "Flagged is_incomplete=True for follow-up. Rows retained.",
    })

    # 7 ── Flag late deliveries (keep — a real service event) ───────────────
    df["is_late"] = (
        pd.to_datetime(df["actual_delivery"]) > pd.to_datetime(df["promised_delivery"])
    )
    n_late = int(df["is_late"].sum())
    steps.append({
        "issue": "late_delivery",
        "label": "Late deliveries",
        "teaches": "shipment follow-up",
        "detected": n_late,
        "action": "flagged",
        "rows_dropped": 0,
        "resolution": "Flagged is_late=True for follow-up. Rows retained.",
    })

    # Drop the synthetic ground-truth column from the production-clean table
    cleaned_ship = df.drop(columns=["_injected_issues"]).reset_index(drop=True)
    cleaned_n = len(cleaned_ship)

    # Validation: detection from data vs injected ground truth (rigor check).
    # Computed on the RAW frame so it measures detection accuracy independent
    # of pipeline ordering. Expect exact matches.
    injected = Counter(issue for lst in raw["_injected_issues"] for issue in lst)
    raw_phantom = int((~raw["store_id"].isin(valid_stores)).sum())
    raw_negnull = int(
        ((raw["qty_ordered"] < 0) | (raw["qty_received"] < 0) | raw["qty_received"].isna()).sum()
    )

    def _check(name: str, detected: int) -> dict:
        inj = int(injected[name])
        return {"injected": inj, "detected": int(detected), "match": inj == int(detected)}

    validation = {
        "sku_typo": _check("sku_typo", n_typo),
        "duplicate_id": _check("duplicate_id", n_dup),
        "phantom_store": _check("phantom_store", raw_phantom),
        "negative_null_qty": _check("negative_null_qty", raw_negnull),
    }

    rows_with_issues = int((raw["_injected_issues"].map(len) > 0).sum())

    audit = {
        "summary": {
            "raw_rows": int(raw_n),
            "rows_with_issues": rows_with_issues,
            "cleaned_rows": int(cleaned_n),
            "rows_removed": int(raw_n - cleaned_n),
            "rows_recovered_in_place": int(n_typo + n_neg),
            "retention_pct": round(cleaned_n / raw_n, 4),
            "flagged_incomplete": n_incomplete,
            "flagged_late": n_late,
        },
        "steps": steps,
        "validation": validation,
    }

    cleaned_facts = {
        "fact_demand": facts["fact_demand"].copy(),
        "fact_inventory": facts["fact_inventory"].copy(),
        "fact_shipments": cleaned_ship,
    }
    return cleaned_facts, audit


if __name__ == "__main__":
    import json
    from generate_dimensions import generate_all_dimensions
    from generate_facts import generate_all_facts

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    dims = generate_all_dimensions()
    facts = generate_all_facts(dims)
    cleaned, audit = run_cleaning_pipeline(facts, dims)

    for name, df in cleaned.items():
        path = CLEANED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  ✓ {name}: {len(df)} rows → {path}")

    audit_path = AUDIT_DIR / "cleaning_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  ✓ Audit report → {audit_path}")
