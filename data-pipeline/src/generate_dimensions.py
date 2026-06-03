"""
Generate dimension tables for the supply chain network.

Produces:
  - dim_products (20 SKUs: snacks, cookies, cereals, beverages)
  - dim_distribution_centers (6 CEDIS in real Mexican cities)
  - dim_stores (~126 retail stores across 4 channels)
  - dim_calendar (90 days with weekday, payday, event flags)

All randomness uses seed 42 for full reproducibility.
Output: Parquet files in data-pipeline/output/dimensions/
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "dimensions"


# ---------------------------------------------------------------------------
# dim_products — 20 SKUs across 4 categories
# ---------------------------------------------------------------------------
def _generate_products() -> pd.DataFrame:
    """
    20 fictional-but-recognizable SKUs for a Mexican beverage & snack company.
    Fields: sku, brand, category, presentation, unit_cost, unit_price, rotation.
    """
    products = [
        # Snacks (5) — high-rotation staples
        ("SKU-001", "Crujitos",   "Snacks",    "150g Bag",       8.50,  18.90, "High"),
        ("SKU-002", "Crujitos",   "Snacks",    "300g Family",   14.20,  32.50, "Med"),
        ("SKU-003", "Tostiricas", "Snacks",    "180g Bag",       9.80,  21.50, "High"),
        ("SKU-004", "Tostiricas", "Snacks",    "62g Personal",   4.10,   9.90, "High"),
        ("SKU-005", "Salabritas", "Snacks",    "200g Bag",      10.50,  23.90, "Med"),
        # Cookies (5)
        ("SKU-006", "Marianitas", "Cookies",   "180g Pack",      7.30,  16.50, "Med"),
        ("SKU-007", "Marianitas", "Cookies",   "400g Family",   13.90,  31.90, "Low"),
        ("SKU-008", "Chocorel",   "Cookies",   "120g Pack",      8.90,  19.90, "Med"),
        ("SKU-009", "Chocorel",   "Cookies",   "240g Box",      15.60,  34.90, "Low"),
        ("SKU-010", "Polvorones", "Cookies",   "200g Tray",      9.20,  20.50, "Med"),
        # Cereals (4)
        ("SKU-011", "Granolitas", "Cereals",   "450g Box",      18.50,  42.90, "Med"),
        ("SKU-012", "Granolitas", "Cereals",   "250g Box",      11.20,  25.90, "Low"),
        ("SKU-013", "Chocobolt",  "Cereals",   "380g Box",      16.80,  38.50, "Med"),
        ("SKU-014", "AvenaSol",   "Cereals",   "500g Bag",      12.40,  27.90, "Low"),
        # Beverages (6) — high-rotation, surge-sensitive
        ("SKU-015", "FrutaViva",  "Beverages", "600ml Bottle",   6.20,  14.90, "High"),
        ("SKU-016", "FrutaViva",  "Beverages", "1.5L Bottle",   11.50,  26.90, "High"),
        ("SKU-017", "AguaPura",   "Beverages", "1L Bottle",      3.80,   9.50, "High"),
        ("SKU-018", "AguaPura",   "Beverages", "500ml 6-Pack",  18.90,  42.00, "Med"),
        ("SKU-019", "Energiza",   "Beverages", "473ml Can",      9.50,  22.90, "Med"),
        ("SKU-020", "Citronela",  "Beverages", "355ml Can",      5.90,  13.90, "High"),
    ]

    cols = ["sku", "brand", "category", "presentation",
            "unit_cost", "unit_price", "rotation"]
    return pd.DataFrame(products, columns=cols)


# ---------------------------------------------------------------------------
# dim_distribution_centers — 6 CEDIS in real Mexican cities
# ---------------------------------------------------------------------------
def _generate_distribution_centers() -> pd.DataFrame:
    """
    6 distribution centers in real Mexican cities across 4 regions.
    Fields: cd_id, name, city, region, lat, lon, capacity.
    """
    centers = [
        ("CD-MX",  "CEDIS Central",    "Mexico City", "Central",   19.4326, -99.1332, 45000),
        ("CD-GDL", "CEDIS Occidente",  "Guadalajara", "West",      20.6597, -103.3496, 32000),
        ("CD-MTY", "CEDIS Norte",      "Monterrey",   "North",     25.6866, -100.3161, 38000),
        ("CD-PUE", "CEDIS Puebla",     "Puebla",      "Central",   19.0414, -98.2063,  22000),
        ("CD-MER", "CEDIS Sureste",    "Merida",      "Southeast", 20.9674, -89.5926,  18000),
        ("CD-TIJ", "CEDIS Noroeste",   "Tijuana",     "North",     32.5149, -117.0382, 25000),
    ]

    cols = ["cd_id", "name", "city", "region", "lat", "lon", "capacity"]
    return pd.DataFrame(centers, columns=cols)


# ---------------------------------------------------------------------------
# dim_stores — ~126 retail stores across 4 channels
# ---------------------------------------------------------------------------
def _generate_stores(cedis: pd.DataFrame) -> pd.DataFrame:
    """
    ~126 stores distributed across CEDIS and channels.
    Channel mix: Mass (~20%), Convenience (~35%), Wholesale (~15%), Traditional (~30%).
    Each store assigned a demand tier (A/B/C) and a CEDIS.
    """
    rng = np.random.default_rng(SEED)

    # Store counts per CEDIS (proportional to capacity, totaling ~126)
    # MX=35, GDL=22, MTY=26, PUE=16, MER=13, TIJ=14  → 126
    cedis_store_counts = {
        "CD-MX": 35, "CD-GDL": 22, "CD-MTY": 26,
        "CD-PUE": 16, "CD-MER": 13, "CD-TIJ": 14,
    }

    channels = ["Mass", "Convenience", "Wholesale", "Traditional"]
    channel_weights = [0.20, 0.35, 0.15, 0.30]
    tiers = ["A", "B", "C"]
    tier_weights = [0.25, 0.45, 0.30]  # fewer high-demand stores

    # Name prefixes by channel for realistic store names
    channel_prefixes = {
        "Mass":        ["SuperMax", "MegaMart", "HiperCompra", "TiendaGrande"],
        "Convenience": ["MiniPronto", "RapidStop", "Express24", "CerquiTa"],
        "Wholesale":   ["MayoreoPlus", "BodegaCentral", "DistribuMax"],
        "Traditional": ["Abarrotes", "Tienda", "Miscelanea", "LaEsquina"],
    }

    # Build region lookup from CEDIS
    cd_region = dict(zip(cedis["cd_id"], cedis["region"]))

    stores = []
    store_counter = 1

    for cd_id, count in cedis_store_counts.items():
        # Assign channels proportionally per CEDIS
        assigned_channels = rng.choice(channels, size=count, p=channel_weights)

        for channel in assigned_channels:
            tier = rng.choice(tiers, p=tier_weights)
            prefix = rng.choice(channel_prefixes[channel])
            store_id = f"ST-{store_counter:04d}"
            name = f"{prefix} {store_id[-3:]}"

            stores.append({
                "store_id": store_id,
                "name": name,
                "channel": channel,
                "demand_tier": tier,
                "assigned_cd_id": cd_id,
                "region": cd_region[cd_id],
            })
            store_counter += 1

    return pd.DataFrame(stores)


# ---------------------------------------------------------------------------
# dim_calendar — 90 days
# ---------------------------------------------------------------------------
def _generate_calendar() -> pd.DataFrame:
    """
    90-day calendar starting 2024-01-01.
    Flags: is_weekend, is_payday (15th, last day of month), is_demand_event
    (2-week sporting-event window boosting beverages & snacks).
    """
    start = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start, periods=90, freq="D")

    # The ~2-week sporting event: Feb 5–18 (a fictional tournament window)
    event_start = pd.Timestamp("2024-02-05")
    event_end = pd.Timestamp("2024-02-18")

    records = []
    for d in dates:
        # Payday: 15th of any month, or last calendar day of the month
        is_last_day = d == d + pd.offsets.MonthEnd(0)
        is_payday = (d.day == 15) or is_last_day

        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "weekday": d.strftime("%A"),
            "month": d.strftime("%Y-%m"),
            "is_weekend": d.weekday() >= 5,
            "is_payday": is_payday,
            "is_demand_event": event_start <= d <= event_end,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_all_dimensions() -> dict[str, pd.DataFrame]:
    """Generate all four dimension tables. Returns dict of name -> DataFrame."""
    cedis = _generate_distribution_centers()
    return {
        "dim_products": _generate_products(),
        "dim_distribution_centers": cedis,
        "dim_stores": _generate_stores(cedis),
        "dim_calendar": _generate_calendar(),
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dims = generate_all_dimensions()
    for name, df in dims.items():
        path = OUTPUT_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  ✓ {name}: {len(df)} rows → {path}")

    # Quick sanity print
    print("\n--- dim_products ---")
    print(dims["dim_products"].to_string(index=False))
    print(f"\n--- dim_distribution_centers ---")
    print(dims["dim_distribution_centers"].to_string(index=False))
    print(f"\n--- dim_stores: {len(dims['dim_stores'])} rows ---")
    print(dims["dim_stores"].head(10).to_string(index=False))
    print(f"\n--- dim_calendar: {len(dims['dim_calendar'])} rows ---")
    print(dims["dim_calendar"].head(10).to_string(index=False))
