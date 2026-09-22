# Large study findings

**This experiment did not establish useful predictive value from the four Jev
features.** The small probability-score improvement was inconclusive, and the
costed portfolio diagnostic performed worse with Jev added.

## Scale and test design

- Searched 149 configured companies and found 1,619 original matching 8-Ks from
  2018–2025. BK lacked a current SEC mapping; some other issuers had no matching filings.
- Processed 1,560 eligible filings from 129 issuers, making 6,240 factual decisions.
  Excluded 51 oversized primary texts and eight source downloads above 32 MiB.
- Downloaded 273,210 adjusted daily price bars, with no price-download failures.
- Final matched sample: 1,088 training events (2018–2022), 138 validation events
  (2023), and 321 test events across 97 issuers (2024–2025).
- Thirteen further events were excluded: three lacked complete/finite market
  features or holding prices, eight crossed split boundaries or fell outside entry
  dates, and two duplicated an issuer/entry date.

The target was whether the stock beat SPY over five sessions, entering on the first
session strictly after the SEC acceptance date. All variants used the same events.
The baseline included market features, filing item flags, text length, time since
acceptance, issuer and sector controls. The Jev variant added probabilities for
refinancing, executed debt maturity extension, covenant waiver, and capex funding.

## Main result

| Gradient boosting variant | Test log loss ↓ | Brier ↓ | AUC ↑ | Accuracy |
|---|---:|---:|---:|---:|
| Market features + filing metadata | 0.72141 | 0.26374 | 0.428 | 47.0% |
| Same inputs + Jev facts | 0.71875 | 0.26231 | 0.440 | 48.3% |
| Constant training-class probability | **0.68976** | **0.24831** | 0.500 | **54.8%** |

Jev minus baseline log loss was **−0.00266**, with an approximate paired 95% interval
of **[−0.01542, +0.01022]**, using 20-session calendar blocks. That range includes no
improvement. Both fitted models were worse than the simple constant prediction.
The secondary logistic comparison also failed to show an improvement from Jev.
The fixed text rules did not establish a meaningful improvement either.

The result was not consistent across test years: Jev improved boosting log loss in
2024 but worsened it in 2025. These year breakdowns are descriptive, not additional
independent tests.

## Economic diagnostic

With the validation-selected probability threshold, a five-session stock-long/SPY-short
portfolio, fixed 5% equity per leg, position limits, and transaction/borrow costs:

| Gradient boosting variant | Trades | Gross return, same positions | Net return | Maximum drawdown |
|---|---:|---:|---:|---:|
| Market + metadata | 44 | −1.77% | −2.24% | −2.54% |
| Market + metadata + Jev | 62 | −4.01% | −4.68% | −4.81% |

These are two-year simulated portfolio returns using adjusted daily-open proxies,
not executable quotes or per-trade returns. Costs were 5 bps per leg per side plus
100 bps/year estimated short borrowing. The negative result already existed before
costs. No test-informed threshold changes or reversals of the signals were tried.

The highest Jev-augmented probability bin looks favorable in the chart, but contains
only **seven** test events. That is not grounds to select a new strategy after seeing
these returns. The saved confidence analysis includes every bin's sample count.

## Cost and verification

Jev returned **5,695,599 input tokens**, costing approximately **$0.2392** at the
configured published rate. This estimates successful-response usage; uncertain
charges for failed requests are not included. Extraction used four workers and
resumed cached responses after interruptions. Two answer distributions required
narrow rounding normalization; original responses remain unchanged.

All **24 local tests passed**. Post-run verification checked all 1,560 request and
response hashes, original SEC acceptance headers, text hashes, feature/label
separation, purged split boundaries, non-overlapping issuer positions, and portfolio
P&L accounting. The protocol and evaluation code were preserved before model fitting.
No model, feature, horizon or strategy threshold was changed after viewing results.

## Interpretation

Jev's cheap interpretation capability worked operationally at this scale. That did
not turn these particular disclosure facts into a useful five-session trading
signal. The result argues against further scaling this exact recipe solely to get
more calls or a better-looking backtest.

It does not test the value of every possible text source or every fact Jev could
extract. The next hypothesis should specify information that could change an
economic forecast and why it might not already be reflected in prices. Any new
feature/horizon search must treat this test period as inspected, with separate new
evaluation data. Historical pretraining contamination, current-universe survivorship,
vendor price revisions and only 321 out-of-time events further limit inference.

The baseline implementation is scikit-learn histogram gradient boosting, not the
XGBoost package. Hyperparameters were fixed rather than optimized on the test set.

## Artifacts

- [Protocol](large-study-protocol.md)
- [Full metrics, comparisons and chart](../results/large_study_v1/REPORT.md)
- [Every test prediction](../results/large_study_v1/test_predictions.csv)
- [Machine-readable summary](../results/large_study_v1/summary.json)
- [Comparison chart](../results/large_study_v1/comparison.png)

Raw sources and responses are in `data/large_study_v1`; inference interruptions are
preserved in `inference_attempts`. Evaluation refuses to overwrite the completed
test. The same final dataset is used for every model comparison.
