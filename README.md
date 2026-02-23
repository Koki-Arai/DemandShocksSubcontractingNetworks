# DemandShocksSubcontractingNetworks
# Demand Shocks and Subcontracting Networks

Replication code for:

> **"Demand Shocks and Subcontracting Networks: Capacity Constraints, Vertical Organization, and Competition in Japanese Public Procurement"**
> *[Journal name and volume to be updated upon acceptance]*

---

## Overview

This repository contains all code used to produce the empirical results in the paper. The analysis examines how demand shocks propagate through subcontracting networks in capacity-constrained procurement markets, using Japanese construction data and exogenous variation from earthquakes and fiscal instruments.

**Three main findings:**
1. Prime contractors absorb ~75% of demand shocks internally (transmission elasticity ε = 0.258)
2. Winning bid rates rise by 7.2 pp for capacity-intensive projects after the 2011 earthquake, persisting 14+ years
3. Institutional procurement rules and labor market tightness jointly amplify both patterns

---

## Repository Structure

```
.
├── code/
│   ├── 01_main/
│   │   └── main_analysis.R              # Sections 5–7, Appendices A–E
│   ├── 02_robustness/
│   │   ├── robustness_analyses.R        # Reviewer response: κ endogeneity (A1–A3), ¥17T sensitivity (B)
│   │   ├── robustness_colab.py          # Google Colab version of robustness checks
│   │   └── generate_paper_text.R        # Auto-generate LaTeX table text
│   └── 03_structural_change/
│       └── structural_change_robustness.py  # Response to structural change concern (Appendix C)
├── data/
│   └── DATA_DESCRIPTION.md             # Variable definitions and data sources
├── docs/
│   └── (generated outputs: tables, figures)
└── README.md
```

---

## Code Description

### `code/01_main/main_analysis.R`

Main empirical analysis. Covers all results reported in Sections 5–7 and Appendices A–E.

| Section | Content | Key output |
|---------|---------|------------|
| Section 5 | Demand transmission elasticity, Bartik IV | Table 5-1 to 5-4 |
| Section 6 | Earthquake DiD, event study, decomposition | Table 6-1, 6-2, Figure 6.1 |
| Section 7 | Institutional amplification | Table 7-1 to 7-4 |
| Appendix A | Kumamoto replication | Table A.1, A.2 |
| Appendix B | FE specification robustness, forest plot | Table B.1, Figure B.1–B.2 |
| Appendix C | Capacity vs. market power decomposition | Table C.1, Figure C.1 |
| Appendix D | Selection bias, backlog, extensive/intensive margin | Table D.1, Figure D.1–D.2 |
| Appendix E | Economic significance: tightness vs. bidder competition | Table E.1, Figure E.1 |

**Required R packages:**
```r
install.packages(c("tidyverse", "fixest", "modelsummary", "sandwich", "lmtest", "broom"))
```

---

### `code/02_robustness/robustness_analyses.R`

R implementation of robustness checks addressing two main reviewer concerns:

**Concern ②: κ endogeneity**
- `A1`: Labor market tightness (job-offer/applicant ratio) as alternative capacity proxy
- `A2`: Lagged κ (t−1, t−12) to mitigate simultaneity bias
- `A3`: Demographic aging (age55_share) as structural IV for κ

**Concern ③: ¥17 trillion overestimation risk**
- `B1`: Entry response scenarios (bidder recovery over 5–14 years)
- `B2`: Estimated price (denominator) endogeneity correction (0–5 pp adjustment)
- `B3`: Persistence profile scenarios (flat, event-study path, exponential decay)

Outputs: `docs/appendix_tables_A1_A3_B1_B3.xlsx`, `docs/appendix_f_sensitivity.png`

---

### `code/02_robustness/robustness_colab.py`

Google Colab-compatible Python version of the robustness checks. Produces identical results to `robustness_analyses.R`. Useful for researchers without an R environment.

**Usage:**
1. Upload `monthly_prefecture_panel.csv` to `/content/` (or mount Google Drive)
2. Run all cells sequentially
3. Download `robustness_results.xlsx` and `appendix_f_sensitivity.png`

**Required packages** (auto-installed in first cell):
```
linearmodels, statsmodels, pandas, numpy, matplotlib, seaborn, openpyxl
```

**Key outputs:**

| Sheet / File | Content |
|-------------|---------|
| `A1_LaborProxy` | Table A1: alternative proxy results |
| `A2_LaggedKappa` | Table A2: lagged κ results |
| `A3_AgingIV` | Table A3: structural IV results |
| `B1_EntryResponse` | Sensitivity: bidder entry scenarios |
| `B2_PriceAdj` | Sensitivity: price denominator scenarios |
| `B3_Persistence` | Sensitivity: persistence profiles |
| `AppendixF_TableF1` | Combined sensitivity (Table F.1) |
| `appendix_f_sensitivity.png` | Figure F.1: three-panel sensitivity figure |

**Sensitivity analysis results summary (Table F.1):**

| Scenario | Cumulative cost (¥T) | % of baseline |
|---------|-------------------|--------------|
| Upper bound (no adjustment) | 16.8 | 99% |
| Event study path only | 9.9 | 58% |
| Moderate: 10yr entry + 2pp price adj | ~11 | ~65% |
| Conservative: event 10yr + fast entry | ~6 | ~35% |
| Lower bound: flat path + fast entry + 5pp adj | ~3.7 | ~22% |

Even the most conservative scenario exceeds ¥3 trillion, confirming fiscal significance across all plausible assumptions.

---

### `code/03_structural_change/structural_change_robustness.py`

Python (Google Colab) analysis addressing the concern that post-2018 labor reforms and government-led wage increases ("official spring offensive") may confound the tightness and competition results.

**Four analyses:**

| Analysis | Method | Purpose |
|---------|--------|---------|
| Table C1 | 2SLS, Pre/Post-2021 subsamples | β₁ sign stable before and after reform |
| Table C2 | DiD, wage control + period subsamples | Post×Large stable; wage index does not confound |
| Placebo A | DiD, FY2006–2010 only, pseudo-treatment = FY2009 | No pre-earthquake Large/Small divergence |
| Placebo B | DiD, FY2016–2019 window | No discontinuous jump at FY2018 reform date |

**Additional output:**
- Event study figure spanning FY2007–2024 with earthquake (2011) and reform (2018) markers
- Year-by-year Large vs. Small win rate differential table

**Required data columns beyond baseline:**
- `wage_index`: MLIT construction labor cost index (prefecture × year, FY2013=100)

---

## Data

All data are derived from publicly available Japanese government administrative sources. Processed datasets will be deposited in the journal data repository upon acceptance.

See [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md) for:
- Complete variable definitions for all six datasets
- Source URLs and access instructions
- Construction notes for derived variables (κ, Bartik instrument)
- Replication environment specifications

**Data files required** (place in `data/` folder):

| File | Source | N |
|------|--------|---|
| `tohoku_bid_panel.csv` | Tohoku RDB (MLIT) | 6,190 |
| `monthly_prefecture_panel.csv` | Juchū Dōtai (MLIT) | 6,335 |
| `annual_prefecture_panel.csv` | Kōji Chōsa (MLIT) | 1,551 |
| `sna_public_investment.csv` | Cabinet Office SNA | 24 |
| `construction_labor_survey.csv` | MLIT Labor Survey | 517 |
| `kumamoto_prefecture_panel.csv` | Kyushu RDB (MLIT) | 77 |

---

## Replication

### Full replication (R)

```r
# Install packages
install.packages(c("tidyverse", "fixest", "modelsummary",
                   "sandwich", "lmtest", "broom"))

# Run main analysis
source("code/01_main/main_analysis.R")

# Run robustness checks (reviewer response)
source("code/02_robustness/robustness_analyses.R")
```

### Robustness checks only (Python / Google Colab)

1. Open `code/02_robustness/robustness_colab.py` in Google Colab
2. Upload data files to `/content/`
3. Run all cells → download `robustness_results.xlsx`

### Structural change robustness (Python / Google Colab)

1. Open `code/03_structural_change/structural_change_robustness.py` in Google Colab
2. Upload data files (including `wage_index` column in monthly panel)
3. Run all cells → download `comment5_results.xlsx` and figure

---

## Citation

```
@article{arai2026demand,
  title   = {Demand Shocks and Subcontracting Networks: Capacity Constraints,
             Vertical Organization, and Competition in Japanese Public Procurement},
  author  = {Arai, [First Name]},
  journal = {[Journal]},
  year    = {2026},
  note    = {Forthcoming}
}
```

---

## License

Code: MIT License. See `LICENSE` for details.

Data: Subject to MLIT data use terms. Processed datasets available via journal data repository upon acceptance.

---

## Contact

For questions about replication, please open a GitHub issue or contact the corresponding author.
