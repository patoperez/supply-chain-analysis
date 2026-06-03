"""
Generate fact tables for the supply chain network.

Produces:
  - fact_demand (daily demand + forecast per SKU × store, ~124k rows)
  - fact_inventory (daily inventory snapshots per SKU × CEDIS, ~10.8k rows)
  - fact_shipments (shipment records with deliberate quality issues, ~12k+ rows)

Quality issues injected per DATA_SPEC.md:
  ~6% incomplete loads, ~9% late deliveries, ~1.5% duplicate IDs,
  ~3% SKU typos, ~1% negative/null qty, handful of phantom shipments.

Seasonality: weekday shape, payday spikes, sporting-event surge,
  ~20% MAPE forecast error.

All randomness uses seed 42 for full reproducibility.
Output: Parquet files in data-pipeline/output/facts/
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "facts"

# ── Demand configuration ──
BASE_DEMAND = {"High": 22, "Med": 12, "Low": 5}
TIER_MULT = {"A": 1.3, "B": 1.0, "C": 0.7}
WEEKDAY_MULT = {
    "Monday": 0.85, "Tuesday": 0.90, "Wednesday": 0.95,
    "Thursday": 1.00, "Friday": 1.15, "Saturday": 1.25, "Sunday": 1.20,
}
PAYDAY_MULT = 1.35
EVENT_MULT = 1.50  # for Beverages & Snacks during sporting event

# ── SKU portfolio sizes by channel ──
PORTFOLIO_SIZE = {
    "Mass": (14, 16),
    "Convenience": (8, 10),
    "Wholesale": (12, 14),
    "Traditional": (10, 12),
}

# ── Shipment cadence (days between orders) by demand tier ──
SHIPMENT_CADENCE = {"A": 3, "B": 5, "C": 7}
SKUS_PER_SHIPMENT = {"A": (5, 8), "B": (4, 6), "C": (3, 5)}

# ── Late-delivery probability by ORDER weekday ──
# Overall ~9%, but weekend dispatch backlog makes orders placed Thu–Sat likelier
# to slip than mid-week orders. Modest, plausible spread — not a dramatic split.
WEEKDAY_LATE_PROB = {
    "Monday": 0.075, "Tuesday": 0.070, "Wednesday": 0.075, "Thursday": 0.095,
    "Friday": 0.120, "Saturday": 0.115, "Sunday": 0.085,
}

# ── CEDIS inventory policy (all in DAYS of average daily demand) ──
# Sized from a textbook reorder-point policy so the network runs at a
# defensible ~3-5% stockout rate, coherent with the ~97-98% fill rate.
# Reorder point ≈ demand over the lead time + safety stock.
INV_SAFETY_DAYS = 2.5        # buffer below which a SKU is "at risk" (used by risk model)
INV_REORDER_DAYS = 6.5       # ≈ mean lead time (~4d) + safety stock (2.5d): textbook ROP
INV_REPLENISH_DAYS = 8.0     # order size, in days of demand
INV_START_DAYS = {"High": 9, "_default": 11}  # opening stock; high-rotation turns leaner
INV_FAIL_PROB = 0.12         # chance a due replenishment slips (supplier shortfall)
INV_FAIL_RESCHEDULE = (2, 6)  # rng.integers low/high (days) when a replenishment slips
INV_LEAD_NORMAL = (3, 6)      # rng.integers low/high (days) normal lead time
INV_LEAD_EVENT = (5, 8)       # rng.integers low/high (days) during events/paydays


# ---------------------------------------------------------------------------
# Step 1: SKU portfolio assignment
# ---------------------------------------------------------------------------
def _assign_portfolios(
    stores: pd.DataFrame, products: pd.DataFrame, rng: np.random.Generator
) -> dict[str, list[str]]:
    """Assign which SKUs each store carries based on channel type."""
    all_skus = products["sku"].values
    portfolios: dict[str, list[str]] = {}

    for _, store in stores.iterrows():
        lo, hi = PORTFOLIO_SIZE[store["channel"]]
        n = int(rng.integers(lo, hi + 1))
        chosen = rng.choice(all_skus, size=n, replace=False)
        portfolios[store["store_id"]] = sorted(chosen.tolist())

    return portfolios


# ---------------------------------------------------------------------------
# Step 2: Demand generation (vectorized)
# ---------------------------------------------------------------------------
def _generate_demand(
    dims: dict[str, pd.DataFrame], rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Generate fact_demand: daily demand per active (store × SKU) combination.
    Returns (demand_df, portfolios).

    Forecast misses payday and event effects → natural ~20% MAPE.
    """
    products = dims["dim_products"]
    stores = dims["dim_stores"]
    calendar = dims["dim_calendar"]

    portfolios = _assign_portfolios(stores, products, rng)

    # Build active (store_id, sku) pairs
    pairs = []
    for store_id, skus in portfolios.items():
        for sku in skus:
            pairs.append({"store_id": store_id, "sku": sku})
    pairs_df = pd.DataFrame(pairs)

    # Cross-join with calendar
    pairs_df["_k"] = 1
    cal = calendar[["date", "weekday", "is_payday", "is_demand_event"]].copy()
    cal["_k"] = 1
    demand = pairs_df.merge(cal, on="_k").drop("_k", axis=1)

    # Merge store info
    demand = demand.merge(
        stores[["store_id", "assigned_cd_id", "demand_tier"]], on="store_id"
    )
    demand.rename(columns={"assigned_cd_id": "cd_id"}, inplace=True)

    # Merge product info
    demand = demand.merge(products[["sku", "rotation", "category"]], on="sku")

    # ── Compute expected demand (full seasonal) ──
    demand["base"] = (
        demand["rotation"].map(BASE_DEMAND).astype(float)
        * demand["demand_tier"].map(TIER_MULT).astype(float)
    )
    demand["wday_m"] = demand["weekday"].map(WEEKDAY_MULT).astype(float)
    demand["pay_m"] = np.where(demand["is_payday"], PAYDAY_MULT, 1.0)
    demand["evt_m"] = np.where(
        demand["is_demand_event"]
        & demand["category"].isin(["Beverages", "Snacks"]),
        EVENT_MULT,
        1.0,
    )

    expected = demand["base"] * demand["wday_m"] * demand["pay_m"] * demand["evt_m"]

    # Forecast sees weekday pattern but NOT payday/event effects
    expected_smooth = demand["base"] * demand["wday_m"]

    # Generate noise
    n = len(demand)
    actual_noise = rng.normal(0, 0.18, size=n)
    forecast_noise = rng.normal(0.02, 0.08, size=n)

    demand["actual_demand"] = np.maximum(
        0, np.round(expected * (1 + actual_noise))
    ).astype(int)
    demand["forecast_demand"] = np.maximum(
        0, np.round(expected_smooth * (1 + forecast_noise))
    ).astype(int)

    result = demand[
        ["date", "sku", "store_id", "cd_id", "forecast_demand", "actual_demand"]
    ].copy()

    return result, portfolios


# ---------------------------------------------------------------------------
# Step 3: Shipment generation (clean base)
# ---------------------------------------------------------------------------
def _generate_clean_shipments(
    dims: dict[str, pd.DataFrame],
    portfolios: dict[str, list[str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate clean CEDIS→store shipments with realistic cadence."""
    stores = dims["dim_stores"]
    calendar = dims["dim_calendar"]

    dates = pd.to_datetime(calendar["date"])
    start_date = dates.min()
    end_date = dates.max()

    records = []
    shipment_counter = 1

    for _, store in stores.iterrows():
        tier = store["demand_tier"]
        cadence = SHIPMENT_CADENCE[tier]
        store_skus = portfolios[store["store_id"]]

        # First order offset (stagger across stores)
        offset = int(rng.integers(0, cadence))
        current = start_date + pd.Timedelta(days=offset)

        while current <= end_date:
            # Select SKUs for this shipment
            lo, hi = SKUS_PER_SHIPMENT[tier]
            n_skus = min(int(rng.integers(lo, hi + 1)), len(store_skus))
            selected = rng.choice(store_skus, size=n_skus, replace=False)

            sid = f"SH-{shipment_counter:06d}"

            dispatch = current + pd.Timedelta(days=int(rng.integers(1, 3)))
            promised = dispatch + pd.Timedelta(days=int(rng.integers(1, 4)))

            for sku in selected:
                qty = int(rng.integers(20, 100))
                records.append(
                    {
                        "shipment_id": sid,
                        "order_date": current.strftime("%Y-%m-%d"),
                        "cd_id": store["assigned_cd_id"],
                        "store_id": store["store_id"],
                        "sku": sku,
                        "qty_ordered": qty,
                        "qty_received": qty,
                        "dispatch_date": dispatch.strftime("%Y-%m-%d"),
                        "promised_delivery": promised.strftime("%Y-%m-%d"),
                        "actual_delivery": promised.strftime("%Y-%m-%d"),
                        "status": "Delivered",
                    }
                )

            shipment_counter += 1

            # Next order: cadence ± 1 day of jitter
            jitter = int(rng.integers(-1, 2))
            current += pd.Timedelta(days=max(1, cadence + jitter))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 4: Inject quality issues
# ---------------------------------------------------------------------------
_SKU_TYPO_MAP = {
    "SKU-001": "SKu-001",
    "SKU-003": "SKU-0O3",   # O instead of 0
    "SKU-005": "sku-005",   # lowercase
    "SKU-008": "SKU-O08",   # O instead of 0
    "SKU-010": "SKU-0l0",   # l instead of 1
    "SKU-012": "sku-012",   # lowercase
    "SKU-015": "SKU-O15",   # O instead of 0
    "SKU-017": "SKU-0I7",   # I instead of 1
    "SKU-019": "sku-019",   # lowercase
    "SKU-020": "SKU-O20",   # O instead of 0
}


def _inject_quality_issues(
    shipments: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject deliberate data-quality issues per DATA_SPEC.md rates.
    Each issue is tagged in '_injected_issues' column for audit.
    """
    df = shipments.copy()
    n = len(df)
    df["_injected_issues"] = [[] for _ in range(n)]

    # 1. Incomplete loads (~6%): qty_received < qty_ordered
    mask = rng.random(n) < 0.06
    fill_frac = rng.uniform(0.50, 0.90, size=n)
    df.loc[mask, "qty_received"] = (
        (df.loc[mask, "qty_ordered"] * fill_frac[mask]).round().astype(int)
    )
    for idx in df.index[mask]:
        df.at[idx, "_injected_issues"] = df.at[idx, "_injected_issues"] + [
            "incomplete_load"
        ]

    # 2. Late deliveries (~9% overall, weekday-dependent): weekend dispatch
    #    backlog makes Thu–Sat orders likelier to slip than mid-week orders.
    #    Keeping rng.random(n) here preserves every downstream RNG draw.
    late_prob = pd.to_datetime(df["order_date"]).dt.day_name().map(WEEKDAY_LATE_PROB).to_numpy()
    mask = rng.random(n) < late_prob
    late_days = rng.integers(1, 6, size=n)
    promised_dt = pd.to_datetime(df["promised_delivery"])
    late_dt = promised_dt + pd.to_timedelta(late_days, unit="D")
    df.loc[mask, "actual_delivery"] = late_dt[mask].dt.strftime("%Y-%m-%d")
    for idx in df.index[mask]:
        df.at[idx, "_injected_issues"] = df.at[idx, "_injected_issues"] + [
            "late_delivery"
        ]

    # 3. Duplicate shipment IDs (~1.5% of rows duplicated)
    n_dupes = max(1, int(round(n * 0.015)))
    dupe_indices = rng.choice(df.index, size=n_dupes, replace=False)
    dupes = df.loc[dupe_indices].copy()
    dupes["_injected_issues"] = [["duplicate_id"] for _ in range(len(dupes))]
    df = pd.concat([df, dupes], ignore_index=True)

    # 4. SKU typos (~3% of original rows)
    n_current = len(df)
    mask = rng.random(n_current) < 0.03
    typo_candidates = df.loc[mask, "sku"].values
    typo_applied = []
    for sku in typo_candidates:
        if sku in _SKU_TYPO_MAP:
            typo_applied.append(_SKU_TYPO_MAP[sku])
        else:
            # Generic: lowercase the whole thing
            typo_applied.append(sku.lower())
    df.loc[mask, "sku"] = typo_applied
    for idx in df.index[mask]:
        df.at[idx, "_injected_issues"] = df.at[idx, "_injected_issues"] + [
            "sku_typo"
        ]

    # 5. Negative / null quantities (~1%)
    mask = rng.random(len(df)) < 0.01
    for idx in df.index[mask]:
        if rng.random() < 0.5:
            df.at[idx, "qty_ordered"] = -abs(df.at[idx, "qty_ordered"])
        else:
            df.at[idx, "qty_received"] = np.nan
        df.at[idx, "_injected_issues"] = df.at[idx, "_injected_issues"] + [
            "negative_null_qty"
        ]

    # 6. Phantom shipments (handful: ~12 rows with non-existent store_id)
    n_phantoms = 12
    phantom_indices = rng.choice(df.index, size=n_phantoms, replace=False)
    phantom_ids = [f"ST-9{i:03d}" for i in range(n_phantoms)]
    df.loc[phantom_indices, "store_id"] = phantom_ids
    for idx in phantom_indices:
        df.at[idx, "_injected_issues"] = df.at[idx, "_injected_issues"] + [
            "phantom_store"
        ]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5: Inventory simulation
# ---------------------------------------------------------------------------
def _generate_inventory(
    dims: dict[str, pd.DataFrame], demand: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Simulate daily CEDIS-level inventory per SKU.

    Sized from a textbook reorder-point policy (see the INV_* constants at the
    top of this module) so the network runs at a defensible ~3-5% stockout rate,
    coherent with the ~98% fill rate: reorder point = lead-time demand + safety
    stock, with occasional supplier slips and longer event/payday lead times.

    Records 'unmet_units' per row — demand the available stock could not cover on
    a stockout day — so lost sales can be estimated honestly downstream.
    """
    products = dims["dim_products"]
    cedis = dims["dim_distribution_centers"]
    calendar = dims["dim_calendar"]
    dates_sorted = sorted(calendar["date"].unique())

    # Build a set of demand-event dates for lead-time inflation
    event_dates = set(
        calendar.loc[calendar["is_demand_event"], "date"].values
    )
    payday_dates = set(
        calendar.loc[calendar["is_payday"], "date"].values
    )

    # Aggregate demand to CEDIS level: daily demand per (date, sku, cd_id)
    agg_demand = (
        demand.groupby(["date", "sku", "cd_id"])["actual_demand"]
        .sum()
        .reset_index()
        .rename(columns={"actual_demand": "daily_demand"})
    )
    demand_lookup = agg_demand.set_index(["date", "sku", "cd_id"])["daily_demand"]

    records = []

    for _, cd in cedis.iterrows():
        cd_id = cd["cd_id"]
        for _, prod in products.iterrows():
            sku = prod["sku"]
            rotation = prod["rotation"]

            # Calculate average daily demand for this SKU at this CEDIS
            cd_sku_demand = agg_demand[
                (agg_demand["cd_id"] == cd_id) & (agg_demand["sku"] == sku)
            ]
            if cd_sku_demand.empty:
                avg_daily = 0
            else:
                avg_daily = cd_sku_demand["daily_demand"].mean()

            if avg_daily == 0:
                for date in dates_sorted:
                    records.append(
                        {
                            "date": date,
                            "sku": sku,
                            "cd_id": cd_id,
                            "stock_units": 0,
                            "safety_stock": 0,
                            "reorder_point": 0,
                            "is_stockout": False,
                            "unmet_units": 0,
                        }
                    )
                continue

            # Reorder point ≈ lead-time demand + safety stock, so the cycle
            # troughs at the safety buffer, not through it (see INV_* constants).
            safety_stock = round(avg_daily * INV_SAFETY_DAYS)
            reorder_point = round(avg_daily * INV_REORDER_DAYS)
            start_days = (
                INV_START_DAYS["High"] if rotation == "High"
                else INV_START_DAYS["_default"]
            )
            stock = round(avg_daily * start_days)
            replenish_amount = round(avg_daily * INV_REPLENISH_DAYS)

            # Replenishment state
            replenish_pending = False
            replenish_day = -1

            for day_idx, date in enumerate(dates_sorted):
                # Check for replenishment arrival
                if replenish_pending and day_idx >= replenish_day:
                    # Chance a due replenishment slips (supplier shortfall)
                    if rng.random() < INV_FAIL_PROB:
                        # Slipped — reschedule a few days out
                        replenish_day = day_idx + int(rng.integers(*INV_FAIL_RESCHEDULE))
                    else:
                        # Arrives, but sometimes partial (80-100% of order)
                        fill = rng.uniform(0.80, 1.0)
                        stock += round(replenish_amount * fill)
                        replenish_pending = False

                # Get today's demand
                try:
                    today_demand = int(demand_lookup.get((date, sku, cd_id), 0))
                except (ValueError, TypeError):
                    today_demand = 0

                # Determine stockout and the unmet quantity (demand the
                # available stock could not cover) BEFORE depleting.
                is_stockout = stock < today_demand
                unmet_units = today_demand - stock if is_stockout else 0

                # Deplete stock (can't go below 0)
                stock = max(0, stock - today_demand)

                records.append(
                    {
                        "date": date,
                        "sku": sku,
                        "cd_id": cd_id,
                        "stock_units": stock,
                        "safety_stock": safety_stock,
                        "reorder_point": reorder_point,
                        "is_stockout": is_stockout,
                        "unmet_units": int(unmet_units),
                    }
                )

                # Trigger replenishment if below reorder point
                if stock <= reorder_point and not replenish_pending:
                    # Longer lead time during events/paydays
                    if date in event_dates or date in payday_dates:
                        lead_time = int(rng.integers(*INV_LEAD_EVENT))
                    else:
                        lead_time = int(rng.integers(*INV_LEAD_NORMAL))
                    replenish_day = day_idx + lead_time
                    replenish_pending = True

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_all_facts(
    dimensions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Generate all fact tables given dimension tables.
    Returns dict of name -> DataFrame.
    Uses SeedSequence to ensure independence between generation steps.
    """
    master = np.random.SeedSequence(SEED)
    seeds = master.spawn(4)

    rng_demand = np.random.default_rng(seeds[0])
    rng_shipments = np.random.default_rng(seeds[1])
    rng_issues = np.random.default_rng(seeds[2])
    rng_inventory = np.random.default_rng(seeds[3])

    print("  Generating demand...")
    demand, portfolios = _generate_demand(dimensions, rng_demand)
    print(f"    → {len(demand):,} rows")

    print("  Generating shipments...")
    shipments = _generate_clean_shipments(dimensions, portfolios, rng_shipments)
    print(f"    → {len(shipments):,} clean rows")

    print("  Injecting quality issues...")
    shipments = _inject_quality_issues(shipments, rng_issues)
    n_issues = sum(1 for issues in shipments["_injected_issues"] if len(issues) > 0)
    print(f"    → {len(shipments):,} total rows, {n_issues:,} with issues")

    print("  Simulating inventory...")
    inventory = _generate_inventory(dimensions, demand, rng_inventory)
    n_stockouts = inventory["is_stockout"].sum()
    print(f"    → {len(inventory):,} rows, {n_stockouts:,} stockout events")

    return {
        "fact_demand": demand,
        "fact_inventory": inventory,
        "fact_shipments": shipments,
    }


if __name__ == "__main__":
    from generate_dimensions import generate_all_dimensions

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dims = generate_all_dimensions()
    facts = generate_all_facts(dims)

    for name, df in facts.items():
        path = OUTPUT_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  ✓ {name}: {len(df):,} rows → {path}")

    # ── Quick validation ──
    demand = facts["fact_demand"]
    shipments = facts["fact_shipments"]
    inventory = facts["fact_inventory"]

    # MAPE check (exclude zero-actual rows per METRICS_SPEC)
    nonzero = demand[demand["actual_demand"] > 0]
    mape = (
        (nonzero["actual_demand"] - nonzero["forecast_demand"]).abs()
        / nonzero["actual_demand"]
    ).mean()
    print(f"\n  MAPE: {mape:.1%}")

    # Issue breakdown
    from collections import Counter

    issue_counter: Counter[str] = Counter()
    for issues in shipments["_injected_issues"]:
        for issue in issues:
            issue_counter[issue] += 1
    print("\n  Quality issues injected:")
    for issue, count in issue_counter.most_common():
        pct = count / len(shipments) * 100
        print(f"    {issue}: {count} ({pct:.1f}%)")

    # Stockout summary
    print(f"\n  Stockout events: {inventory['is_stockout'].sum()}")
    print(f"  Stockout rate: {inventory['is_stockout'].mean():.1%}")
