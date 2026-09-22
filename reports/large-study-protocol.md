# Large study v1: protocol fixed before return evaluation

This experiment tests whether the four already-audited Jev disclosure concepts add
information about subsequent stock returns. It does not ask Jev to forecast prices.

## Data and target

- Fixed, current convenience universe of 149 companies in `config/large_study.json`.
  This is not a point-in-time constituent list and has survivorship/identity bias.
- Original 2018–2025 SEC 8-Ks containing any of items 1.01, 1.02, 2.03, 2.04 or 3.02.
  At most 375 hash-selected filings per year, without inspecting returns. Discovery
  produced 1,619 candidates, below every annual cap. BK lacked a current SEC mapping
  and is logged as a discovery exclusion, not silently remapped.
- Full primary text up to 24,000 characters, with larger documents excluded and
  counted. No exhibit text, article summaries or later documents are supplied.
- Entry is the first observed SPY session strictly after the filing's acceptance
  date in New York, even for a premarket filing. This provides a conservative delay.
- Target: whether the stock's five-session adjusted open-to-open return exceeds
  SPY's return over those exact sessions. This is a binary excess-return target.
- Alpaca consolidated SIP daily bars, all corporate-action adjustments, automatic
  symbol remapping disabled. Use ratio features rather than historical dollar-price
  levels. Historical vendor corrections, adjustments and symbol identity still
  limit point-in-time correctness. Daily opens are price proxies, not verified fills.

## Comparison

All models use exactly the same eligible events:

1. Stock and SPY return, trend, volatility, volume and range features from the last
   completed session; filing item flags, source-text length, elapsed time since
   acceptance, sector, and issuer controls.
2. The same inputs plus four fixed rule-based text flags, as a diagnostic control.
3. The same inputs plus Jev's disclosed/unclear probabilities for refinancing,
   executed maturity extension, covenant waiver, and capex funding. The redundant
   third probability is omitted.

Primary learner: scikit-learn histogram gradient boosting, with fixed parameters.
Secondary learner: scaled logistic regression with fixed regularization. This uses
the installed gradient boosting implementation, not the XGBoost package.
Training-class-prior and 50/50 predictions are additional probability baselines.

Train on 2018–2022; validation is 2023; test is 2024–2025. Remove any observation
whose label reaches the next split. Same-issuer filings mapping to the same entry
date are collapsed by retaining the latest available filing. Different overlapping
events remain in predictive metrics but are not assumed independent.

Primary comparison: test log loss with Jev minus test log loss without Jev. Negative
means improvement. Approximate uncertainty uses paired resampling of 20-session
calendar blocks, preserving all same-day events together, with 1,000 replicates.
Secondary metrics, year breakdowns and the rules/logistic comparisons are descriptive.
Do not select a winning horizon, prompt, feature list or learner after inspecting test
returns. The earlier extraction audits inspected some 2024–2025 source text but no
return labels; this is outcome-untouched, not completely source-blind evaluation.

## Costed economic diagnostic

Each learner's probability threshold is its 2023 validation 80th percentile. Freeze
it before applying it to 2024–2025. An eligible event takes 5% of current equity long
the stock and 5% short SPY for five sessions. No overlapping position in the same
issuer; new entries must keep gross exposure within 100%. A 20% drawdown blocks new
entries while that drawdown persists.

Charge 5 bps per leg on entry and exit and 100 bps/year for the SPY short. Mark
positions daily, keep idle capital at zero yield, and do not compound overlapping
event returns as if they were separate full-capital trades. Report net equity,
drawdown, number of trades, turnover, transaction costs and estimated borrow costs.
This is a portfolio diagnostic with approximate prices, no capacity/locate model,
and a dollar-neutral SPY hedge, not a beta-neutral or executable strategy.

## Reproducibility and scope

Source responses, hashes, exclusions and inference responses are cached under
`data/large_study_v1`. Both numerical baselines and Jev use the same final sample;
missing inference never results in a silent zero feature. Entire holding windows
and 61 prior sessions must have required prices. This can introduce coverage bias,
so exclusions and per-issuer/year counts are retained.

Jev receives only the filing text and unchanged factual questions. Its probability
output is not assumed calibrated as a trading forecast. Pretraining may include
historical text or later information; future paper observations are required to
test this limitation. A negative result rejects only this feature/data/horizon
combination, not the general value of language-model interpretation.

Before evaluating returns, a live response exposed two-decimal API rounding:
0.81 + 0.17 + 0.01 = 0.99. Raw responses and their hashes are preserved. The adapter
accepts only the narrow error bound consistent with independently rounding each
option to two decimals, then normalizes derived probabilities to sum to one. Larger
or unexplained discrepancies still fail validation. A benchmark outcome/input
column-prefix collision was also found and fixed before any predictive evaluation;
the regression test now mutates future stock and benchmark prices together and
verifies that neither can alter the actual selected model inputs.

Source access is free under the configured accounts. The initial usage-based
inference limit is $1, expected to be ample for this sample; the user has authorized
additional Jev use if needed. Actual billing and failed-request charges may differ
from estimates. Four inference workers and four download workers are used; a shared
SEC limiter allows at most one request start every 0.4 seconds, below the
[SEC's published limit](https://www.sec.gov/about/developer-resources).

See the [Alpaca adjustment definitions](https://docs.alpaca.markets/us/reference/stockbars)
and [Jev model/pricing documentation](https://docs.typesafe.ai/models).
