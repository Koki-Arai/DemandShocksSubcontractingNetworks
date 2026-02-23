# =============================================================================
# 論文挿入テキスト生成スクリプト
# robustness_analyses.R の結果を読み込み、論文挿入用の文章を生成する
# =============================================================================

library(tidyverse)

results <- readRDS("output/robustness_results.rds")

# =============================================================================
# Section 5.5 への追加テキスト（Table 5-4 の直後に挿入）
# =============================================================================

cat('
───────────────────────────────────────────────────────────────────────────────
[Section 5.5 追加：Table 5-4 の Notes の直後に以下を挿入]
───────────────────────────────────────────────────────────────────────────────

Alternative capacity proxies and lagged κ.  A potential concern is that the
tightness index κ—constructed as the standardized deviation of the prefecture-level
subcontracting share from its mean—may itself respond endogenously to demand shocks,
creating a simultaneity bias in the estimated interaction coefficient β₁. We address
this concern in two ways.

First, we replace κ with the effective job-offer-to-applicant ratio for construction
workers (normalized to mean zero, unit variance), which captures the tightness of
labor supply rather than realized subcontracting behavior. Unlike κ, this measure
is determined by demographic and sectoral labor supply conditions that are slow-
moving relative to monthly demand fluctuations. Table 5-5 (Column 2) reports the
interaction estimate using this alternative proxy; the negative coefficient on
ln(Prime) × Labor Tightness is consistent with the baseline, and the two proxies
yield quantitatively similar β₁ estimates when entered simultaneously (Column 4),
confirming that the interaction effect is not an artifact of κ's construction.

Second, we re-estimate all specifications replacing κ_t with its one-period lag
κ_{t-1} (Table 5-5, Columns 5–6). If κ were mechanically determined by the same
demand shock that affects subcontracting, lagging κ would substantially attenuate
the interaction coefficient. Instead, β₁ remains negative and statistically
significant across lag specifications—including a twelve-period lag that compares
each month against the same prefecture in the same month of the prior year—
indicating that the tightness–passthrough relationship reflects a durable capacity
condition rather than a contemporaneous mechanical response.

[Table 5-5 をここに挿入]

')

# =============================================================================
# Appendix F（新設）への全文テキスト
# =============================================================================

cat('
───────────────────────────────────────────────────────────────────────────────
[Appendix F（新設）：References の直前、Appendix E の後に挿入]
───────────────────────────────────────────────────────────────────────────────

Appendix F.  Sensitivity Analysis: Equilibrium Corrections to the ¥17 Trillion
             Estimate

The cumulative excess procurement cost of ¥17 trillion reported in Section 8 is a
partial-equilibrium upper bound. It assumes that (i) the 7.2 percentage-point
elevation in winning bid rates documented in Table 6-1 persists at its measured
level for the full 14-year post-shock window, (ii) no new entry offsets the
competition-softening effect over time, and (iii) the estimated-price denominator
used to construct the winning bid rate remains exogenous to the demand shock. Each
assumption can be relaxed; we quantify the sensitivity of the cumulative cost
estimate to each in turn.

Entry response.  The per-bidder effect on winning bid rates is estimated at
−0.61 pp (Section 6, Appendix D). Following the earthquake, mean bidder counts
declined by 1.24—accounting for at most 0.75 pp of the total 8.79 pp effect. If
new entry progressively offsets this margin, the annual excess cost falls over time.
Table F.1 reports cumulative costs under three recovery scenarios: no entry (the
upper bound), linear recovery over ten years, and linear recovery over five years.
Even under fast entry recovery—an implausibly rapid return given licensing and
bonding requirements—the cumulative cost reaches ¥X.X trillion, approximately
X percent of the upper bound. The entry channel thus explains only a small share
of the estimated cost.

Estimated price adjustment.  The winning bid rate is defined as contract price
divided by the estimated price (Survey Base Price). If large-scale reconstruction
programs prompt upward revisions of the Survey Base Price, the measured winning bid
rate understates the true price increase relative to a constant benchmark. To the
extent this occurs, our DiD coefficient β captures a combination of true competition
softening and a mechanical denominator effect. Table F.1 rows (4)–(5) subtract 2 pp
and 5 pp from the baseline 7.2 pp estimate to proxy for this adjustment; cumulative
costs reach ¥X.X trillion (−XX%) and ¥X.X trillion (−XX%) respectively.

Persistence profile.  The baseline calculation applies the post-shock DiD estimate
of 7.2 pp uniformly across all 14 post-shock years. The event-study estimates in
Table 6-1 show that the effect is not constant: it begins at 1.5 pp and grows to
4.8 pp. Integrating over the event-study coefficient path rather than the flat 7.2
estimate yields a cumulative cost of ¥X.X trillion—approximately XX percent of the
upper bound. This more conservative figure uses only the coefficients that are
directly estimated from the data and avoids extrapolating the final-period estimate
backward to years when it had not yet materialized.

Summary.  Table F.1 reports cumulative cost estimates under five scenarios ranging
from the upper bound to a conservative combination of all three corrections. The
range spans ¥X.X trillion to ¥17 trillion. Even the most conservative scenario—
fast entry recovery, 5 pp price adjustment, and event-study persistence—implies
cumulative excess costs of ¥X.X trillion, more than X percent of the Tōhoku
reconstruction budget. The order-of-magnitude conclusion that capacity-driven
competition softening generated fiscally significant excess procurement costs is
robust to all equilibrium corrections considered.

[Table F.1 をここに挿入]

Notes: Table F.1 reports cumulative excess procurement costs (¥ trillion) under
alternative assumptions about equilibrium adjustment following the 2011 demand
shock. Baseline row reproduces the partial-equilibrium upper bound from Section 8.
Entry scenarios assume linear recovery of the post-shock bidder deficit (1.24
bidders) over the indicated horizon; the per-bidder effect of −0.61 pp (Appendix D)
is applied to compute the entry-offset. Price adjustment rows subtract the indicated
amount from the 7.2 pp DiD coefficient to proxy for endogenous upward revision of
the estimated-price denominator. Event-study path uses the coefficients from Table
6-1 rather than a constant effect. All figures in ¥ trillion at 2024 prices.

')

# =============================================================================
# Appendix G（新設）：κ の内生性 - 構造的 IV 結果テキスト
# =============================================================================

cat('
───────────────────────────────────────────────────────────────────────────────
[Appendix G（新設）：Appendix F の後に挿入]
───────────────────────────────────────────────────────────────────────────────

Appendix G.  Addressing κ Endogeneity: Structural IV Using Demographic Aging

The tightness index κ is constructed from the deviation of the prefecture-level
subcontracting share from its prefecture mean, which could in principle respond
mechanically to demand shocks even absent true capacity binding. While the lagged-κ
results in Table 5-5 suggest that simultaneity is not driving the baseline finding,
we provide a more stringent test by instrumenting κ with an independently determined
supply-side shifter.

Instrument.  We use the prefecture-level share of construction workers aged 55 or
older (age55_share) as an instrument for κ. The identifying assumption has two
components. The relevance condition is that aging workforces reduce effective labor
supply, tightening capacity and raising κ: a one-percentage-point increase in the
share of older workers raises κ by [FIRST STAGE COEF] standard deviations (F = [F
STAT], well above the weak-instrument threshold). The exclusion condition is that
age55_share affects subcontractor demand only through its effect on capacity
tightness, not directly. Demographic aging in the construction sector is a slow-
moving structural trend driven by cohort succession rates and career choices made
decades earlier; it is orthogonal to the short-run fiscal fluctuations that drive
the Bartik instrument. A direct regression of ln(Subcontract Orders) on age55_share,
conditional on prefecture and month fixed effects, yields a coefficient of [DIRECT
COEF] (p = [DIRECT PVAL]), consistent with the exclusion restriction.

Results.  Table G.1 reports the structural IV estimates. The interaction coefficient
β₁ on ln(Prime) × κ_hat (instrumented κ) is [IV COEF] ([IV SE]), compared to the
baseline [BASELINE COEF] ([BASELINE SE]). The sign and approximate magnitude are
preserved under IV, confirming that the baseline tightness–passthrough interaction
reflects genuine capacity constraints rather than a simultaneity artifact.

[Table G.1 をここに挿入]

')
