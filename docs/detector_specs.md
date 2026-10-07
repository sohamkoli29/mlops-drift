# Drift Detector Specifications

Reference = training set (reference.parquet). Current = one window of 1,000 production samples.
All detectors expose the same interface (implemented Day 6):

    score(reference: DataFrame, current: DataFrame) -> {feature: (score, flagged)}

## Feature scope

| Detector     | Numeric (5)            | Categorical (8)         |
|--------------|------------------------|-------------------------|
| PSI          | quantile bins          | category shares         |
| KS           | yes                    | not applicable          |
| Wasserstein  | yes (normalised)       | not applicable          |

KS and Wasserstein are defined for ordered numeric data, so they cover the 5 numeric
features only. When comparing detectors, drift scenarios must shift numeric features.
Categorical-only drift is a PSI-only case and is reported separately.

## 1. PSI (Population Stability Index)

For bins i with reference share p_i and current share q_i:

    PSI = sum_i (q_i - p_i) * ln(q_i / p_i)

- Numeric bins: 10 quantile bins from the reference; use the unique edges, outer edges -inf and +inf.
  Zero-heavy features (capital_gain, capital_loss) collapse to few bins; that is expected.
- Categorical bins: each reference category, plus one "__OTHER__" bin for unseen categories.
- Smoothing: clip every share to at least 1e-4 before the log.
- Thresholds (flag if >= 0.10):

| PSI          | Level  |
|--------------|--------|
| < 0.10       | none   |
| 0.10 - 0.25  | medium |
| > 0.25       | high   |

## 2. Kolmogorov-Smirnov

    D = sup_x |F_ref(x) - F_cur(x)|     (scipy.stats.ks_2samp)

- Flag if p-value < 0.05.
- Also record D as the score (the p-value alone is hard to compare across features).
- Caution: with 19.5k reference vs 1,000 current samples, small harmless shifts can reach
  significance. Day 11 checks the false-alarm rate on the no-drift control and tightens
  (for example p < 0.01 or D > cutoff) if needed.

## 3. Wasserstein (Earth Mover's distance), normalised

    W = scipy.stats.wasserstein_distance(ref, cur) / std(ref)

- Dividing by the reference std makes scores comparable across features with different units.
- Starting threshold: W > 0.10 (placeholder). The real cutoff is calibrated on the
  no-drift control: for example the 95th or 99th percentile of W over no-drift windows.

## Window-level aggregation (built Day 8)

- share_flagged = flagged features / evaluated features
- Severity: none if share_flagged < 0.15 and max PSI < 0.10; high if share_flagged >= 0.40
  or max PSI > 0.25; otherwise medium. (Starting values, tuned Day 11.)
- Top-k: features ranked by score, with k = 3.

## Known-answer tests (Day 6)

| Case                                  | Expected                          |
|---------------------------------------|-----------------------------------|
| ref vs. a fresh sample of itself      | PSI < 0.05, KS not flagged, W < 0.05 |
| N(0,1) vs N(0.5,1)                    | PSI medium or high, KS flagged    |
| N(0,1) vs N(2,1)                      | PSI high, W about 2.0 before normalising |
| categorical shares shifted 50 -> 90%  | PSI high                          |
