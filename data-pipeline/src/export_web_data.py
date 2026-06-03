"""
Export pipeline outputs to static JSON for the web app.

THE BOUNDARY: This script is the ONLY bridge between Python and the web app.
It writes JSON files into web/public/data/. The web app reads ONLY those files.

Exports (per DATA_SPEC.md filter-aggregation contract):
  - daily_aggregates.json  — daily aggregates by (date × region × brand)
                             for live dashboard recomputation (NOT raw rows)
  - dimensions.json        — all dimension tables (products, CEDIS, stores, calendar)
  - kpis.json              — headline KPI values
  - risk_scores.json       — fact_stockout_risk for Section 6
  - cleaning_audit.json    — audit report for Section 3
  - raw_sample.json        — sample dirty rows for Section 2
  - findings.json          — Pareto, day-of-week, MAPE cuts for Section 4

Target: web payload in tens of KB, not megabytes.
"""

import json
import pandas as pd
from pathlib import Path

WEB_DATA_DIR = Path(__file__).parent.parent.parent / "web" / "public" / "data"


# Daily-aggregate columns. Short names keep the largest payload compact; these
# are the sufficient statistics to recompute every dashboard KPI under any
# region/brand/date filter by simple summation:
#   act/fcst         → demand vs forecast (time series)
#   ape_sum/ape_n    → MAPE = Σape_sum / Σape_n   (exact under any filter)
#   ordered/received → fill rate = Σreceived / Σordered
#   lines/late       → on-time % = 1 - Σlate / Σlines
#   stockouts        → stockout events;  lost → lost-sales (MXN, price-weighted)
# Stockout-rate denominators are derivable from dims (#CEDIS × #SKUs per group).
_AGG_COLS = [
    "date", "region", "brand", "act", "fcst", "ape_sum", "ape_n",
    "ordered", "received", "lines", "late", "stockouts", "lost",
]


def _write_json(path: Path, obj, compact: bool = False) -> int:
    """Write obj as JSON; return bytes written."""
    seps = (",", ":") if compact else (", ", ": ")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=seps,
                  indent=None if compact else 2)
    return path.stat().st_size


def _records(df: pd.DataFrame) -> list:
    """DataFrame → list of dicts, with numpy types coerced to JSON-native."""
    return json.loads(df.to_json(orient="records"))


def _build_daily_aggregates(
    cleaned: dict[str, pd.DataFrame], dims: dict[str, pd.DataFrame]
) -> dict:
    """Daily aggregate by (date × region × brand) per the DATA_SPEC contract."""
    cd_region = dims["dim_distribution_centers"].set_index("cd_id")["region"]
    sku_brand = dims["dim_products"].set_index("sku")["brand"]
    sku_price = dims["dim_products"].set_index("sku")["unit_price"]

    # Demand: levels + the exact MAPE numerator/denominator
    d = cleaned["fact_demand"].copy()
    d["region"] = d["cd_id"].map(cd_region)
    d["brand"] = d["sku"].map(sku_brand)
    nz = d["actual_demand"] > 0
    d["ape"] = 0.0
    d.loc[nz, "ape"] = (
        (d.loc[nz, "actual_demand"] - d.loc[nz, "forecast_demand"]).abs()
        / d.loc[nz, "actual_demand"]
    )
    d["ape_n"] = nz.astype(int)
    dem = d.groupby(["date", "region", "brand"]).agg(
        act=("actual_demand", "sum"), fcst=("forecast_demand", "sum"),
        ape_sum=("ape", "sum"), ape_n=("ape_n", "sum"),
    )

    # Shipments (keyed by order_date): fill rate + on-time components
    s = cleaned["fact_shipments"].copy()
    s["region"] = s["cd_id"].map(cd_region)
    s["brand"] = s["sku"].map(sku_brand)
    ship = s.groupby([s["order_date"].rename("date"), "region", "brand"]).agg(
        ordered=("qty_ordered", "sum"), received=("qty_received", "sum"),
        lines=("is_late", "size"), late=("is_late", "sum"),
    )

    # Inventory: stockout rate + lost sales (price-weighted estimate)
    inv = cleaned["fact_inventory"].copy()
    inv["region"] = inv["cd_id"].map(cd_region)
    inv["brand"] = inv["sku"].map(sku_brand)
    inv["lost"] = inv["unmet_units"] * inv["sku"].map(sku_price)
    inva = inv.groupby(["date", "region", "brand"]).agg(
        stockouts=("is_stockout", "sum"), lost=("lost", "sum"),
    )

    agg = dem.join([ship, inva], how="outer").fillna(0).reset_index()
    for c in ["act", "fcst", "ape_n", "ordered", "received", "lines", "late",
              "stockouts"]:
        agg[c] = agg[c].astype(int)
    agg["ape_sum"] = agg["ape_sum"].round(4)
    agg["lost"] = agg["lost"].round(2)
    agg = agg.sort_values(["date", "region", "brand"])

    return {"columns": _AGG_COLS,
            "rows": json.loads(agg[_AGG_COLS].to_json(orient="values"))}


def _build_raw_sample(raw_shipments: pd.DataFrame, per_issue: int = 2) -> list:
    """A few real dirty raw rows (one+ of each issue) for Beat 2."""
    cols = ["shipment_id", "order_date", "cd_id", "store_id", "sku",
            "qty_ordered", "qty_received", "promised_delivery",
            "actual_delivery", "status"]
    df = raw_shipments.reset_index(drop=True)
    picked: list[int] = []
    for issue in ["sku_typo", "incomplete_load", "phantom_store", "late_delivery"]:
        hit = df.index[df["_injected_issues"].map(lambda lst: issue in lst)]
        picked.extend(list(hit[:per_issue]))
    # Show BOTH quantity-glitch variants explicitly (negative order, missing receipt)
    neg = df.index[df["qty_ordered"] < 0]
    if len(neg):
        picked.append(int(neg[0]))
    nul = df.index[df["qty_received"].isna()]
    if len(nul):
        picked.append(int(nul[0]))
    # include a full duplicate pair so the duplication is visible
    dup = df.index[df["_injected_issues"].map(lambda lst: "duplicate_id" in lst)]
    if len(dup):
        sid, sku = df.at[dup[0], "shipment_id"], df.at[dup[0], "sku"]
        pair = df.index[(df["shipment_id"] == sid) & (df["sku"] == sku)]
        picked.extend(list(pair))

    out = []
    for i in dict.fromkeys(picked):  # de-dup, preserve order
        row = {c: df.at[i, c] for c in cols}
        for q in ["qty_ordered", "qty_received"]:
            v = df.at[i, q]
            row[q] = None if pd.isna(v) else int(v)
        row["issues"] = list(df.at[i, "_injected_issues"])
        out.append(row)
    return out


def export_all(
    dimensions: dict[str, pd.DataFrame],
    raw_facts: dict[str, pd.DataFrame],
    cleaned_facts: dict[str, pd.DataFrame],
    kpis: dict,
    findings: dict,
    risk_scores: pd.DataFrame,
    audit: dict,
) -> dict:
    """Export all data to static JSON for the web app. Returns {file: bytes}."""
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    stores = dimensions["dim_stores"][
        ["store_id", "channel", "demand_tier", "assigned_cd_id", "region"]
    ]
    calendar = dimensions["dim_calendar"][
        ["date", "weekday", "is_weekend", "is_payday", "is_demand_event"]
    ]
    dims_out = {
        "products": _records(dimensions["dim_products"]),
        "distribution_centers": _records(dimensions["dim_distribution_centers"]),
        "stores": _records(stores),
        "calendar": _records(calendar),
    }

    sizes["dimensions.json"] = _write_json(WEB_DATA_DIR / "dimensions.json", dims_out, compact=True)
    sizes["kpis.json"] = _write_json(WEB_DATA_DIR / "kpis.json", kpis)
    sizes["findings.json"] = _write_json(WEB_DATA_DIR / "findings.json", findings)
    sizes["cleaning_audit.json"] = _write_json(WEB_DATA_DIR / "cleaning_audit.json", audit)
    sizes["risk_scores.json"] = _write_json(WEB_DATA_DIR / "risk_scores.json", _records(risk_scores), compact=True)
    sizes["raw_sample.json"] = _write_json(WEB_DATA_DIR / "raw_sample.json", _build_raw_sample(raw_facts["fact_shipments"]))
    sizes["daily_aggregates.json"] = _write_json(WEB_DATA_DIR / "daily_aggregates.json", _build_daily_aggregates(cleaned_facts, dimensions), compact=True)

    return sizes


if __name__ == "__main__":
    from generate_dimensions import generate_all_dimensions
    from generate_facts import generate_all_facts
    from clean_pipeline import run_cleaning_pipeline
    from analytics import compute_kpis, compute_findings, compute_risk_scores

    dims = generate_all_dimensions()
    raw = generate_all_facts(dims)
    cleaned, audit = run_cleaning_pipeline(raw, dims)
    kpis = compute_kpis(cleaned, dims)
    findings = compute_findings(cleaned, dims)
    risk = compute_risk_scores(cleaned, dims)

    sizes = export_all(dims, raw, cleaned, kpis, findings, risk, audit)

    total = sum(sizes.values())
    print(f"  ✓ Exported to {WEB_DATA_DIR}")
    for name, nbytes in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<24} {nbytes / 1024:7.1f} KB")
    print(f"    {'TOTAL':<24} {total / 1024:7.1f} KB")
