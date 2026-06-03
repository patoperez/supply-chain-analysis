"""
Compute analytics and the stock-out risk model from cleaned data.

Headline KPIs (per METRICS_SPEC.md):
  1. Fill Rate = SUM(qty_received) / SUM(qty_ordered)
  2. On-Time Delivery % = COUNT(actual <= promised) / COUNT(all)
  3. Stockout Events & Lost Sales (MXN) = unmet_demand × unit_price
  4. MAPE = AVG(|actual - forecast| / actual) where actual > 0

Derived cuts:
  - Pareto by CEDIS (lost-sales concentration)
  - Late-delivery day-of-week pattern
  - MAPE by category
  - Days of cover (stock / avg daily demand)

Risk model (transparent heuristic, 0-100 score per SKU × CEDIS):
  - Days of cover (dominant driver)
  - Position vs safety stock
  - Demand CV (std / mean)
  - SKU rotation bonus
  → Bucketed: Low / Medium / High

Output: data-pipeline/output/analytics/
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "analytics"


_WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def _lost_sales_by_row(
    inventory: pd.DataFrame, products: pd.DataFrame
) -> pd.Series:
    """
    Lost sales (MXN) per inventory row = unmet_units × unit_price.
    A single source of truth so every consumer (KPI total, Pareto) agrees.
    Labeled an ESTIMATE: it assumes each unmet unit would have sold at list price.
    """
    price = products.set_index("sku")["unit_price"]
    return inventory["unmet_units"] * inventory["sku"].map(price)


def _mape(actual: pd.Series, forecast: pd.Series) -> float:
    """MAPE per METRICS_SPEC: mean(|actual - forecast| / actual) over actual > 0."""
    mask = actual > 0
    return float(((actual[mask] - forecast[mask]).abs() / actual[mask]).mean())


def compute_kpis(
    cleaned_facts: dict[str, pd.DataFrame],
    dimensions: dict[str, pd.DataFrame],
) -> dict:
    """Compute the headline KPIs from CLEANED data, per METRICS_SPEC.md."""
    shipments = cleaned_facts["fact_shipments"]
    demand = cleaned_facts["fact_demand"]
    inventory = cleaned_facts["fact_inventory"]
    products = dimensions["dim_products"]

    # 1. Fill rate = received / ordered
    fill_rate = shipments["qty_received"].sum() / shipments["qty_ordered"].sum()

    # 2. On-time delivery = share of lines delivered on/before promise
    on_time = (~shipments["is_late"]).mean()

    # 3. Stockout events & lost sales (modeled estimate)
    stockout_events = int(inventory["is_stockout"].sum())
    lost_sales = float(_lost_sales_by_row(inventory, products).sum())

    # 4. Forecast accuracy — MAPE (days with actual > 0)
    mape = _mape(demand["actual_demand"], demand["forecast_demand"])

    return {
        "fill_rate": round(float(fill_rate), 4),
        "on_time_delivery": round(float(on_time), 4),
        "stockout_events": stockout_events,
        "stockout_rate": round(stockout_events / len(inventory), 4),
        "unmet_units_total": int(inventory["unmet_units"].sum()),
        "lost_sales_mxn": round(lost_sales, 2),
        "lost_sales_is_estimate": True,
        "mape": round(mape, 4),
        # supporting context (so downstream copy can cite consistent counts)
        "shipment_lines": int(len(shipments)),
        "late_lines": int(shipments["is_late"].sum()),
        "incomplete_lines": int(shipments["is_incomplete"].sum()),
        "inventory_rows": int(len(inventory)),
    }


def compute_findings(
    cleaned_facts: dict[str, pd.DataFrame],
    dimensions: dict[str, pd.DataFrame],
) -> dict:
    """The derived analytical cuts for Beat 4, per METRICS_SPEC.md."""
    shipments = cleaned_facts["fact_shipments"]
    demand = cleaned_facts["fact_demand"]
    inventory = cleaned_facts["fact_inventory"]
    products = dimensions["dim_products"]
    cedis = dimensions["dim_distribution_centers"]

    # ── Pareto by CEDIS: where the lost-sales cost concentrates ──
    inv = inventory.copy()
    inv["lost_sales"] = _lost_sales_by_row(inv, products)
    by_cd = inv.groupby("cd_id")["lost_sales"].sum().sort_values(ascending=False)
    total_ls = float(by_cd.sum())
    name = cedis.set_index("cd_id")["name"]
    city = cedis.set_index("cd_id")["city"]
    pareto, cum = [], 0.0
    for cd_id, ls in by_cd.items():
        share = float(ls) / total_ls if total_ls else 0.0
        cum += share
        pareto.append({
            "cd_id": cd_id,
            "name": name.get(cd_id),
            "city": city.get(cd_id),
            "lost_sales_mxn": round(float(ls), 2),
            "share": round(share, 4),
            "cumulative_share": round(cum, 4),
        })

    # ── Late deliveries by order weekday: the operational bottleneck signal ──
    sh = shipments.copy()
    sh["order_weekday"] = pd.to_datetime(sh["order_date"]).dt.day_name()
    grp = sh.groupby("order_weekday")["is_late"].agg(["size", "sum"])
    late_by_weekday = []
    for d in _WEEKDAY_ORDER:
        if d in grp.index:
            n, late = int(grp.loc[d, "size"]), int(grp.loc[d, "sum"])
            late_by_weekday.append({
                "weekday": d,
                "shipments": n,
                "late": late,
                "late_rate": round(late / n, 4) if n else 0.0,
            })

    # ── MAPE by category: where planning hurts most ──
    dm = demand.merge(products[["sku", "category"]], on="sku")
    mape_by_category = [
        {"category": cat, "mape": round(_mape(g["actual_demand"], g["forecast_demand"]), 4)}
        for cat, g in dm.groupby("category")
    ]
    mape_by_category.sort(key=lambda r: r["mape"], reverse=True)

    return {
        "pareto_by_cedis": pareto,
        "late_by_weekday": late_by_weekday,
        "mape_by_category": mape_by_category,
    }


_ROTATION_PTS = {"High": 10, "Med": 5, "Low": 0}


def _cover_points(days: float) -> int:
    """Low days-of-cover is the dominant driver, tiered down from <1 day."""
    if pd.isna(days):
        return 0
    if days < 1:
        return 50
    if days < 2:
        return 40
    if days < 3:
        return 30
    if days < 5:
        return 18
    if days < 7:
        return 8
    return 0


def _volatility_points(cv: float) -> int:
    """
    Higher demand volatility (CV) raises uncertainty, and thus risk. Thresholds
    are calibrated to CEDIS-level demand CV, which aggregation smooths into the
    ~0.15-0.26 range; the high tier flags the event/payday-sensitive SKUs.
    """
    if pd.isna(cv):
        return 0
    if cv >= 0.24:
        return 20
    if cv >= 0.20:
        return 12
    if cv >= 0.16:
        return 6
    return 0


def compute_risk_scores(
    cleaned_facts: dict[str, pd.DataFrame],
    dimensions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Explainable stock-out risk model — a transparent, point-based scoring
    heuristic (NOT machine learning). Every point traces to a named driver, so a
    planner can see *why* a SKU x CEDIS is flagged:

      - days of cover (stock / avg daily demand) — dominant driver, up to 50 pts
      - position vs safety stock (below the buffer?)            — up to 20 pts
      - demand volatility (CV = std / mean)                     — up to 20 pts
      - SKU rotation (costlier to run out)                      — up to 10 pts

    Score is capped at 100 and bucketed Low (<35) / Medium (35-59) / High (60+).
    Returns fact_stockout_risk (one row per SKU x CEDIS), including the component
    points so the score is fully auditable.
    """
    demand = cleaned_facts["fact_demand"]
    inventory = cleaned_facts["fact_inventory"]
    products = dimensions["dim_products"]

    # Daily CEDIS-level demand per (sku, cd_id): mean level and volatility
    daily = (
        demand.groupby(["date", "sku", "cd_id"])["actual_demand"].sum().reset_index()
    )
    stats = (
        daily.groupby(["sku", "cd_id"])["actual_demand"]
        .agg(avg_daily_demand="mean", demand_std="std")
        .reset_index()
    )

    # Current snapshot: latest date's stock + safety stock per (sku, cd_id)
    last_date = inventory["date"].max()
    snap = inventory.loc[
        inventory["date"] == last_date,
        ["sku", "cd_id", "stock_units", "safety_stock"],
    ].rename(columns={"stock_units": "current_stock"})

    df = snap.merge(stats, on=["sku", "cd_id"], how="left")
    df = df.merge(products[["sku", "rotation"]], on="sku", how="left")
    df["avg_daily_demand"] = df["avg_daily_demand"].fillna(0.0)
    df["demand_std"] = df["demand_std"].fillna(0.0)

    # Risk inputs (guard against any zero-demand combo → NaN, scored as no risk)
    safe = df["avg_daily_demand"].where(df["avg_daily_demand"] > 0)
    df["days_of_cover"] = df["current_stock"] / safe
    df["demand_cv"] = (df["demand_std"] / safe).fillna(0.0)
    df["below_safety"] = df["current_stock"] < df["safety_stock"]

    # Transparent point allocation
    df["cover_pts"] = df["days_of_cover"].map(_cover_points)
    df["safety_pts"] = df["below_safety"].map({True: 20, False: 0})
    df["volatility_pts"] = df["demand_cv"].map(_volatility_points)
    df["rotation_pts"] = df["rotation"].map(_ROTATION_PTS).fillna(0).astype(int)

    df["risk_score"] = (
        df["cover_pts"] + df["safety_pts"] + df["volatility_pts"] + df["rotation_pts"]
    ).clip(upper=100).astype(int)
    df["risk_level"] = pd.cut(
        df["risk_score"], bins=[-1, 34, 59, 100], labels=["Low", "Medium", "High"]
    ).astype(str)

    # Round for presentation
    df["days_of_cover"] = df["days_of_cover"].round(2)
    df["demand_cv"] = df["demand_cv"].round(3)
    df["avg_daily_demand"] = df["avg_daily_demand"].round(2)

    cols = [
        "sku", "cd_id", "rotation", "current_stock", "avg_daily_demand",
        "days_of_cover", "safety_stock", "below_safety", "demand_cv",
        "cover_pts", "safety_pts", "volatility_pts", "rotation_pts",
        "risk_score", "risk_level",
    ]
    return df[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import json
    from generate_dimensions import generate_all_dimensions
    from generate_facts import generate_all_facts
    from clean_pipeline import run_cleaning_pipeline

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dims = generate_all_dimensions()
    facts = generate_all_facts(dims)
    cleaned, _ = run_cleaning_pipeline(facts, dims)

    kpis = compute_kpis(cleaned, dims)
    findings = compute_findings(cleaned, dims)
    risk = compute_risk_scores(cleaned, dims)

    for name, payload in [("kpis", kpis), ("findings", findings)]:
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {name} → {path}")

    risk_path = OUTPUT_DIR / "fact_stockout_risk.parquet"
    risk.to_parquet(risk_path, index=False)
    print(f"  ✓ risk: {len(risk)} rows → {risk_path}")
