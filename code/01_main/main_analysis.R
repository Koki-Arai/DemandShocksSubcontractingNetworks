# =============================================================================
# Demand Shocks and Subcontracting Networks
# Main Empirical Analysis
#
# Paper sections covered:
#   Section 5  — Demand transmission elasticity (Bartik IV, Table 5-1 to 5-4)
#   Section 6  — Competition effects, DiD, event study (Table 6-1, 6-2)
#   Section 7  — Institutional amplification (Table 7-1 to 7-4)
#   Appendix A — Kumamoto replication
#   Appendix B — FE specification robustness
#   Appendix C — Capacity vs. market power decomposition
#   Appendix D — Selection bias, backlog exclusion, extensive/intensive margin
#   Appendix E — Economic significance
#
# Software: R 4.3+
# Key packages: fixest, modelsummary, tidyverse, sandwich
# =============================================================================

library(tidyverse)
library(fixest)        # feols, feiv — two-way FE and IV estimation
library(modelsummary)  # publication-quality tables
library(sandwich)      # clustered standard errors
library(lmtest)        # coeftest

# =============================================================================
# 0. Data loading
# =============================================================================
# See data/ folder and data/DATA_DESCRIPTION.md for variable definitions.
# All raw data are from administrative sources listed in Table 2 of the paper.

# Primary dataset: Tohoku Regional Development Bureau bid records
# Unit: project-grade × month cell
# Period: FY2006–2024, N = 6,190
df_bid <- read_csv("data/tohoku_bid_panel.csv")

# Monthly prefecture panel (Construction Order Dynamics Survey, Juchū Dōtai)
# Unit: prefecture × month
# Period: FY2013–2024, N = 6,335
df_monthly <- read_csv("data/monthly_prefecture_panel.csv") %>%
  mutate(
    ln_sub      = log(sub_orders),
    ln_prime    = log(prime_orders),
    lp_x_kappa  = ln_prime * kappa_z,
    bartik_x_kappa = bartik * kappa_z,
    ym          = year * 100 + month
  )

# Annual prefecture panel (Construction Work Survey, Kōji Chōsa)
# Unit: prefecture × year
# Period: FY2013–2023, N = 1,551
df_annual <- read_csv("data/annual_prefecture_panel.csv") %>%
  mutate(
    ln_sub   = log(sub_orders),
    ln_prime = log(prime_orders)
  )

# Kumamoto Prefecture bid records (for Appendix A replication)
# Unit: prefecture × year, N = 77
df_kumamoto <- read_csv("data/kumamoto_prefecture_panel.csv")

# =============================================================================
# 1. Section 5: Demand Transmission Elasticity
# =============================================================================

# --- Table 5-1: Baseline demand transmission (annual panel) ---
# OLS baseline
ols_baseline <- feols(ln_sub ~ ln_prime | pref_id + year,
                      data = df_annual, cluster = ~pref_id)

# Bartik IV
iv_baseline <- feols(ln_sub ~ 1 | pref_id + year |
                       ln_prime ~ bartik,
                     data = df_annual, cluster = ~pref_id)

# Monthly panel IV (preferred specification)
iv_monthly <- feols(ln_sub ~ 1 | pref_id + ym |
                      ln_prime ~ bartik,
                    data = df_monthly, cluster = ~pref_id)

# --- Table 5-2: Tightness–passthrough interaction ---
# Main interaction: ln(Prime) × κ
iv_interaction <- feols(ln_sub ~ 1 | pref_id + ym |
                          ln_prime + lp_x_kappa ~ bartik + bartik_x_kappa,
                        data = df_monthly, cluster = ~pref_id)

# High-θ / Low-θ subsample split (institutional rigidity)
iv_high_theta <- feols(ln_sub ~ 1 | pref_id + ym |
                         ln_prime + lp_x_kappa ~ bartik + bartik_x_kappa,
                       data = df_monthly %>% filter(pub_ratio > median(pub_ratio)),
                       cluster = ~pref_id)

iv_low_theta  <- feols(ln_sub ~ 1 | pref_id + ym |
                         ln_prime + lp_x_kappa ~ bartik + bartik_x_kappa,
                       data = df_monthly %>% filter(pub_ratio <= median(pub_ratio)),
                       cluster = ~pref_id)

# --- Table 5-3: Heterogeneity in transmission elasticity ---
# (Subsample by θ, market thickness, labor tightness)
# See robustness_analyses.R for full implementation

# --- Table 5-4: Robustness checks ---
# (Tokyo exclusion, alternative instruments, placebo)
# See robustness_analyses.R for full implementation

# Output Table 5-1
modelsummary(
  list("OLS" = ols_baseline, "IV (annual)" = iv_baseline, "IV (monthly)" = iv_monthly),
  output = "docs/table_5_1_transmission.tex",
  stars = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  coef_rename = c("fit_ln_prime" = "ln(Prime) [ε]"),
  gof_map = c("nobs", "r.squared")
)

# =============================================================================
# 2. Section 6: Competition Effects (Earthquake DiD)
# =============================================================================

# Prep: Large project indicator, post-earthquake indicator
df_bid <- df_bid %>%
  mutate(
    large = as.integer(grade %in% c("A", "B")),
    post  = as.integer(year >= 2011),
    post_large = post * large
  )

# --- Table 6-1: Persistence of competition softening ---
# Separate DiD estimates for each post-shock window

did_windows <- list(
  "0-1yr"  = feols(win_rate ~ post_large | grade_id + ym,
                   data = df_bid %>% filter(year <= 2012 | year < 2011),
                   cluster = ~grade_id),
  "2-4yr"  = feols(win_rate ~ post_large | grade_id + ym,
                   data = df_bid %>% filter(year %in% c(2008:2010, 2013:2015)),
                   cluster = ~grade_id),
  "5-9yr"  = feols(win_rate ~ post_large | grade_id + ym,
                   data = df_bid %>% filter(year %in% c(2008:2010, 2016:2020)),
                   cluster = ~grade_id),
  "10+yr"  = feols(win_rate ~ post_large | grade_id + ym,
                   data = df_bid %>% filter(year %in% c(2008:2010, 2021:2024)),
                   cluster = ~grade_id)
)

# --- Table 6-2: Robustness and decomposition ---
# Baseline
did_baseline <- feols(win_rate ~ post_large | grade_id + ym,
                      data = df_bid, cluster = ~grade_id)

# Controlling for number of bidders (extensive margin)
did_bidders  <- feols(win_rate ~ post_large + log(n_bidders) | grade_id + ym,
                      data = df_bid, cluster = ~grade_id)

# Macro controls (Nikkei, FX, rates, PPI)
did_macro    <- feols(win_rate ~ post_large + nikkei_z + fx_z + jgb_z + ppi_z |
                        grade_id + ym,
                      data = df_bid, cluster = ~grade_id)

# Pre-shock parallel trends: F-test on pre-period year × Large interactions
df_pre <- df_bid %>% filter(year < 2011) %>%
  mutate(yr_large = factor(year) : factor(large))
did_pretest <- feols(win_rate ~ yr_large | grade_id + ym,
                     data = df_pre, cluster = ~grade_id)
# joint F-test for all yr_large coefficients
# wald(did_pretest, ...)

# --- Event study (Figure 6.1 + Table 6-1 dynamic) ---
df_bid <- df_bid %>%
  mutate(
    yr_rel = year - 2010,  # base year = FY2010
    yr_rel_clamp = pmax(-4, pmin(13, yr_rel))
  )
for (y in -4:13) {
  df_bid[[paste0("D_", y)]] <- as.integer(df_bid$yr_rel == y & df_bid$large == 1)
}
event_dummies <- paste0("D_", -4:13) %>% setdiff("D_0")  # base = yr_rel 0
event_formula <- as.formula(
  paste("win_rate ~", paste(event_dummies, collapse = " + "), "| grade_id + ym")
)
did_event <- feols(event_formula, data = df_bid, cluster = ~grade_id)

# Export event study coefficients
event_coefs <- broom::tidy(did_event, conf.int = TRUE) %>%
  filter(str_starts(term, "D_")) %>%
  mutate(yr_rel = as.integer(str_remove(term, "D_"))) %>%
  bind_rows(tibble(term = "D_0", estimate = 0, std.error = 0,
                   conf.low = 0, conf.high = 0, yr_rel = 0)) %>%
  arrange(yr_rel)
write_csv(event_coefs, "docs/event_study_coefs.csv")

# =============================================================================
# 3. Section 7: Institutional Amplification
# =============================================================================

# --- Table 7-1: Procurement rigidity and subcontracting intensity ---
ols_inst_1 <- feols(sub_ratio ~ pub_ratio + ln_prime | pref_id,
                    data = df_annual, cluster = ~pref_id)
ols_inst_2 <- feols(sub_ratio ~ pub_ratio + ln_prime | pref_id + year,
                    data = df_annual, cluster = ~pref_id)
ols_inst_3 <- feols(sub_ratio ~ pub_ratio + ln_prime + log(workers) | pref_id + year,
                    data = df_annual, cluster = ~pref_id)

# --- Table 7-2: Labor constraints and outsourcing ---
ols_labor  <- feols(outsourcing_ratio ~ log(workers) + ln_prime | pref_id + year,
                    data = df_annual, cluster = ~pref_id)

# =============================================================================
# 4. Appendix A: Kumamoto Replication
# =============================================================================

did_kumamoto <- feols(
  cbind(ln_prime, ln_sub, sub_ratio) ~
    i(year, treated, ref = 2015) | pref_id + year,
  data = df_kumamoto, cluster = ~pref_id
)

# =============================================================================
# 5. Appendix C: Capacity vs. Market Power Decomposition
# =============================================================================

# BidRate, PseudoCost, and Markup regressions
ols_bid_rate   <- feols(win_rate   ~ tightness_z | firm_id + ym, data = df_bid)
ols_pseudocost <- feols(pseudo_cost ~ tightness_z | firm_id + ym, data = df_bid)
ols_markup     <- feols(markup     ~ tightness_z | firm_id + ym, data = df_bid)

# =============================================================================
# 6. Export
# =============================================================================
cat("\n====== Main Analysis Complete ======\n")
cat("Key estimates:\n")
cat(sprintf("  ε (demand transmission, IV monthly): %.3f (SE = %.3f)\n",
            coef(iv_monthly)["fit_ln_prime"],
            se(iv_monthly)["fit_ln_prime"]))
cat(sprintf("  β (competition DiD, baseline):       %.3f (SE = %.3f)\n",
            coef(did_baseline)["post_large"],
            se(did_baseline)["post_large"]))
cat(sprintf("  δ (procurement rigidity):            %.3f (SE = %.3f)\n",
            coef(ols_inst_2)["pub_ratio"],
            se(ols_inst_2)["pub_ratio"]))
