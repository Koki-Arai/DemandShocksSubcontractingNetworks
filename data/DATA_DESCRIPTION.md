# Data Description

This document lists all datasets used in the paper. Raw data cannot be redistributed due to data use agreements and government data policies. Researchers wishing to replicate the analysis should obtain the data directly from the sources listed below.

---

## Dataset 1: Tohoku Regional Development Bureau Bid Records

| Item | Detail |
|------|--------|
| **File name** | `tohoku_bid_panel.csv` |
| **Source** | Tohoku Regional Development Bureau (MLIT) |
| **URL** | https://www.thr.mlit.go.jp/bumon/b00097/k00910/nyusatsu/ |
| **Coverage** | FY2006–2024 |
| **Unit of observation** | Project-grade × month cell |
| **N** | 6,190 |
| **Used in** | Section 6 (competition effects), Appendix B, C, D, E |

**Key variables:**

| Variable | Description |
|----------|-------------|
| `grade` | Contractor rank (A/B/C/D) based on project contract value |
| `year` | Fiscal year |
| `month` | Month (1–12) |
| `win_rate` | Winning bid rate (%) = winning contract price / estimated price × 100 |
| `n_bidders` | Number of participating bidders |
| `single_bidder` | Indicator: single bidder auction (1/0) |
| `contract_value` | Average contract value in cell (¥ million) |
| `tightness_z` | Capacity tightness index κ (standardized, mean=0, SD=1) |
| `pseudo_cost` | Estimated cost component of winning bid |
| `markup` | win_rate − pseudo_cost |

**Construction notes:**
- Aggregated from project-level records to project-grade × month cells
- `tightness_z` constructed from Construction Order Dynamics Survey (Dataset 4) and merged by regional bureau × month
- FY2013+ subset used for tightness specifications due to κ data availability

---

## Dataset 2: Annual Prefecture Panel (Construction Work Survey / Kōji Chōsa)

| Item | Detail |
|------|--------|
| **File name** | `annual_prefecture_panel.csv` |
| **Source** | Ministry of Land, Infrastructure, Transport and Tourism (MLIT) |
| **URL** | https://www.mlit.go.jp/sogoseisaku/jouhouka/sosei_jouhouka_tk4_000004.html |
| **Coverage** | FY2013–2023 |
| **Unit of observation** | Prefecture × fiscal year |
| **N** | 1,551 (47 prefectures × 33 periods, unbalanced) |
| **Used in** | Section 5 (annual transmission elasticity), Section 7 |

**Key variables:**

| Variable | Description |
|----------|-------------|
| `pref_id` | Prefecture identifier (1–47) |
| `year` | Fiscal year |
| `sub_orders` | Subcontract completed work (¥ million) |
| `prime_orders` | Prime contractor completed work (¥ million) |
| `sub_ratio` | Subcontracting ratio = sub_orders / prime_orders |
| `pub_ratio` | Public works ratio = public prime work / total prime work (institutional rigidity θ proxy) |
| `workers` | Number of construction workers (prefecture × year) |
| `age55_share` | Share of construction workers aged 55+ |
| `outsourcing_ratio` | Outsourcing cost / labor cost ratio |
| `bartik` | Bartik shift-share instrument = pub_ratio_{i,2013} × ΔNationalPublicInvestment_t |

---

## Dataset 3: Monthly Prefecture Panel (Construction Order Dynamics Survey / Juchū Dōtai Tōkei)

| Item | Detail |
|------|--------|
| **File name** | `monthly_prefecture_panel.csv` |
| **Source** | Ministry of Land, Infrastructure, Transport and Tourism (MLIT) |
| **URL** | https://www.mlit.go.jp/sogoseisaku/jouhouka/sosei_jouhouka_tk4_000002.html |
| **Coverage** | FY2013–2024 |
| **Unit of observation** | Prefecture × month |
| **N** | 6,335 (47 prefectures × 135 months, unbalanced) |
| **Used in** | Section 5 (monthly transmission, tightness interaction), Section 6 (κ construction) |

**Key variables:**

| Variable | Description |
|----------|-------------|
| `pref_id` | Prefecture identifier (1–47) |
| `year` | Fiscal year |
| `month` | Month (1–12) |
| `sub_orders` | Subcontract orders received (¥ million) |
| `prime_orders` | Prime contractor orders received (¥ million) |
| `kappa_z` | Capacity tightness index (standardized): (sub/prime) − E[sub/prime]_month, scaled to mean=0, SD=1 |
| `pub_ratio` | Public works ratio (prefecture-level annual, merged) |
| `bartik` | Bartik instrument (same construction as Dataset 2) |
| `labor_tightness` | Effective job-offer-to-applicant ratio for construction workers |
| `age55_share` | Share of construction workers aged 55+ (annual, merged to monthly) |
| `wage_index` | MLIT construction labor cost index (FY2013=100); used in Appendix C robustness |

**Construction notes for κ:**
```
κ_{mt} = (sub_orders_{mt} / prime_orders_{mt}) − E[sub/prime]_m
```
where E[sub/prime]_m is the prefecture-month mean over the full sample. Standardized to mean=0, SD=1. The FY2021 structural break (shift of +1.25 SD) reflects the combined effect of accelerating labor shortages and post-COVID infrastructure stimulus.

---

## Dataset 4: National Accounts (SNA) — Public Investment

| Item | Detail |
|------|--------|
| **File name** | `sna_public_investment.csv` |
| **Source** | Cabinet Office, System of National Accounts |
| **URL** | https://www.esri.cao.go.jp/jp/sna/data/data_list/kakuhou/files/files_kakuhou.html |
| **Coverage** | FY2001–2024 |
| **Unit of observation** | National × fiscal year |
| **N** | 24 |
| **Used in** | Bartik shift component (Section 4.2, Section 5) |

**Key variables:**

| Variable | Description |
|----------|-------------|
| `year` | Fiscal year |
| `public_investment` | Total public investment = central + local government capital formation (¥ trillion) |
| `delta_pub_inv` | Year-on-year change (shift component of Bartik instrument) |

---

## Dataset 5: Construction Labor Force Survey (MLIT)

| Item | Detail |
|------|--------|
| **File name** | `construction_labor_survey.csv` |
| **Source** | Ministry of Land, Infrastructure, Transport and Tourism (MLIT) |
| **URL** | https://www.mlit.go.jp/totikensangyo/const/totikensangyo_const_tk2_000080.html |
| **Coverage** | FY2013–2023 |
| **Unit of observation** | Prefecture × fiscal year |
| **N** | 517 (47 prefectures × 11 years) |
| **Used in** | Section 7 (labor constraint, outsourcing elasticity), Appendix C (structural change robustness) |

**Key variables:**

| Variable | Description |
|----------|-------------|
| `pref_id` | Prefecture identifier (1–47) |
| `year` | Fiscal year |
| `workers` | Total construction workers |
| `age55_share` | Share of workers aged 55 or older |
| `job_offer_ratio` | Effective job-offer-to-applicant ratio for construction workers |
| `wage_index` | Construction labor cost index (FY2013=100); proxy for "official spring offensive" wage increases |

---

## Dataset 6: Kumamoto Prefecture Bid Records (Appendix A)

| Item | Detail |
|------|--------|
| **File name** | `kumamoto_prefecture_panel.csv` |
| **Source** | Kyushu Regional Development Bureau / Kumamoto Prefecture (MLIT) |
| **Coverage** | FY2013–2024 |
| **Unit of observation** | Prefecture × year |
| **N** | 77 |
| **Used in** | Appendix A (external validity replication) |

**Key variables:** Same structure as Dataset 1, aggregated to prefecture-year level. `treated` = 1 for Kumamoto prefecture after FY2016.

---

## Data Availability and Access

All six datasets are derived from publicly available administrative sources maintained by the Japanese government. They can be obtained as follows:

1. **MLIT Construction Surveys (Datasets 2, 3, 5)**: Available for download from the MLIT statistics portal at the URLs listed above. No registration required.

2. **Bid Records (Datasets 1, 6)**: Individual bid records are published by each Regional Development Bureau. Tohoku RDB records are available at the URL listed above. Prefecture-level records require contact with the respective bureau.

3. **SNA (Dataset 4)**: Downloadable from the Cabinet Office statistics portal.

Processed panel datasets as used in the analysis (after aggregation, merging, and variable construction) will be made available upon acceptance via the journal data repository, subject to MLIT data use terms.

---

## Replication Environment

| Software | Version | Purpose |
|----------|---------|---------|
| R | ≥ 4.3.0 | Main analysis (Sections 5–7, Appendices) |
| Python | ≥ 3.10 | Robustness checks (Google Colab compatible) |
| fixest | ≥ 0.11 | 2SLS with fixed effects |
| linearmodels | ≥ 4.30 | IV2SLS (Python) |
| modelsummary | ≥ 1.4 | Table output |
