# Supply Chain Analytics — Distribution Network Stock-Out Analysis

End-to-end analysis of a distribution network's service failures: **where sales were
lost to stock-outs, why, and how to see the next one coming.** Built on a realistically
defective dataset and fully reproducible from a fixed seed.

**🔗 Live interactive case study:** https://perezfajardo.com/supply-chain-control-tower/
&nbsp;·&nbsp; presentation source: [supply-chain-control-tower](https://github.com/patoperez/supply-chain-control-tower)

> **All data is 100% synthetic** — no real, proprietary, or confidential information
> appears anywhere. The companies, SKUs, prices, and quantities are fabricated to
> demonstrate analytical method on realistically messy data.

---

## The scenario

A fictional Mexican beverage & snack company's distribution network over 90 days:
**6 distribution centers (CEDIS)**, **~126 stores** across four channels, **20 SKUs**.
The realism is in the *structure and the defects* of the data, not the numbers.

## What this analysis does

1. **Generate** a star-schema dataset (demand · inventory · shipments) with realistic
   seasonality — weekday shape, payday spikes, a two-week sporting-event demand surge,
   and a built-in ~20% forecast error so forecast-vs-actual is real.
2. **Inject and then detect deliberate data-quality issues** at designed rates —
   incomplete loads, late deliveries, duplicate records, SKU typos, negative/null
   quantities, and phantom (unknown-store) shipments.
3. **Clean** by the cheapest honest means: recover what's recoverable (normalize SKU
   typos, sign-correct negatives), flag the real service events (late / incomplete),
   and drop only the genuinely unusable (duplicates, phantom stores, missing receipts).
   **~98% of rows retained** — fix it, don't throw it away.
4. **Analyze** root causes and quantify the cost to the business.
5. **Score stock-out risk** with a transparent, explainable model.

## Key findings

- **~$1.5M in estimated lost sales** to stock-outs *(a labeled modeled estimate, not a
  precise figure)*.
- **Three CEDIS drive ~62% of the loss** — a sharp Pareto concentration that says where
  to act first.
- **Late deliveries cluster Thursday–Saturday** (~12% late on Friday vs ~6% on Tuesday)
  — a weekend dispatch-backlog signal, fixable by moving the cutoff, not by capital.
- **Forecast error ~19% (MAPE)**, worst for the event-driven categories the forecast
  doesn't anticipate — and stock-outs spike in exactly that window.

## The stock-out risk model

A transparent, **explainable scoring heuristic** (0–100) — deliberately not a black
box. Every point traces to a named driver a planner can act on:

- **days of cover** (the dominant driver), **position vs. safety stock**, **demand
  volatility** (coefficient of variation), and **SKU rotation**.

Scores bucket into Low / Medium / High. Notably, the forward-looking risk view
independently flags the **same** distribution center the historical loss analysis did
— two independent methods pointing at one answer.

## Reproducibility

Every random draw uses a **fixed seed (42)**, so re-running the pipeline yields a
byte-identical dataset.

```bash
cd data-pipeline
pip install -r requirements.txt

python src/generate_facts.py     # synthetic network (dimensions + facts, with defects)
python src/clean_pipeline.py     # detect → recover / flag / drop → audit report
python src/analytics.py          # KPIs, root-cause findings, risk model
python src/export_web_data.py    # aggregate + export JSON for the presentation layer
```

Generated outputs are **committed for inspection** under `data-pipeline/output/`.

## Repository structure

```
data-pipeline/
├── src/
│   ├── generate_dimensions.py   # products, CEDIS, stores, calendar
│   ├── generate_facts.py        # demand, inventory, shipments (+ injected defects)
│   ├── clean_pipeline.py        # detect → recover / flag / drop, with an audit
│   ├── analytics.py             # KPIs, findings, explainable stock-out risk model
│   └── export_web_data.py       # static JSON for the web presentation
├── output/                      # generated data (parquet) + audit & analytics JSON
└── requirements.txt
```

## Methods & metrics

- **Fill rate**, **on-time delivery**, **stock-out events & lost sales (est.)**, and
  **forecast accuracy (MAPE)** — each computed from a defensible, stated formula.
- **Lost sales is always a modeled estimate**, never presented as a hard figure.
- Stock and service levels are sized from a textbook reorder-point policy (reorder
  point ≈ lead-time demand + safety stock).

*Stack: Python · pandas · numpy. A Power BI view of the same data is planned (not uploaded yet.*
