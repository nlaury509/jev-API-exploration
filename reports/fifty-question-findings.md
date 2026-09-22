# Fifty questions: completed exploratory results

Expanding from four to fifty questions did not establish better predictions in
this setup. The larger model lost less money in the costed simulation than the
four-question version, but all five gradient-boosting portfolios lost money.
These findings apply to these question bundles and this dataset, not to every
possible use of text features.

Jev processed 1,560 filings from 129 issuers, answering 50 questions per request:
78,000 answers, 21,482,799 input tokens, and approximately **$0.9023** in successful
response usage. Median recorded request latency was **0.625 seconds**. Failed
requests might incur additional charges; this is not an account billing receipt.
The published price used was $0.042 per million input tokens, with free outputs:
https://docs.typesafe.ai/models (checked September 21, 2026).

## What was compared

All versions use the same market data, filing metadata, gradient-boosting
parameters, five-session target, trading rules and costs. The bundles add 0, 4,
10, 25 or 50 questions in an order fixed before this run. The original four
cached answers stay identical across versions; 46 new questions supply extra
probabilities. Fifty questions produce 115 text-derived numeric inputs, compared
with eight from four questions, on top of 193 market/metadata inputs.

Training: 1,088 events from 2018–2022. Validation: 138 events from 2023.
Previously inspected test: 321 events from 2024–2025. The same 1,547 matched
events were retained, with the same exclusions and boundary purges.

| Questions | Prediction error (log loss; lower is better) | Net simulated return | Trades |
|---|---:|---:|---:|
| 0 | 0.72141 | -2.24% | 44 |
| 4 | 0.71875 | -4.68% | 62 |
| 10 | 0.72288 | -1.18% | 53 |
| 25 | 0.71992 | -1.11% | 47 |
| 50 | 0.72048 | -1.79% | 57 |

A constant probability based on the training class frequency had lower error
than all five models: **0.68976**. Fifty versus four questions increased log loss
by **0.00173**, with an approximate date-block interval **[-0.00903, +0.01276]**.
Fifty versus zero changed it by **-0.00093**, interval **[-0.01451, +0.01272]**.
Both comparisons include zero. They do not establish predictive improvement.
The 50-question model had slightly worse log loss than four in both test years.

Portfolio differences do not track prediction quality monotonically. The fixed
validation-derived trade thresholds select different opportunities, and these
are small, noisy samples of correlated trades. The least negative portfolio is
not a validated winner. Simulations buy the stock and short equal dollars of
SPY, use adjusted daily-open proxies, and include the original estimated costs.

## Does more information help?

Additional useful information can help. Additional columns can also be redundant,
rare, noisy or incorrectly extracted. The learner must estimate relationships
from limited examples; it cannot automatically know which apparent relationships
will persist.

The secondary logistic model illustrated overfitting: training log loss improved
from **0.59286** with four questions to **0.54568** with fifty, while later-period
loss deteriorated from **0.93355** to **1.02385**. Its unchanged regularization is
part of this test; this does not prove that every tuned wider model would fail.

For **27 of 50 questions**, the same selected answer appeared in at least 95% of
training filings; 25 of these were new questions. This selected-label summary
does not mean all their probabilities were constant, but it explains why fifty
questions need not provide fifty broadly varying independent signals. The corpus
was selected for financing-related filing items; many broader business questions
are rarely answered in this source text.

## Extraction and implementation checks

- 29 tests passed, including response validation, preservation of unknown answers,
  nested feature selection, exclusion of outcome columns and existing timing tests.
- Original baseline and four-question predictions reproduced. All ten learners'
  saved test log losses and simulated portfolios were independently reconciled.
- All original row values, targets and splits were retained; frozen hashes match.
- The original four questions were repeated within the wider API calls only for
  a batch-context stability check. Selected-label agreement was **99.76%**; this
  is not a factual-accuracy or calibration measurement.
- Eight of 78,000 API answers selected a label inconsistent with their largest
  reported probability. Raw outputs are retained. Only the validated probability
  vectors enter predictors; the protocol amendment predates expanded evaluation.
- A limited post-response source check found actual extraction mistakes, including
  missing a signed acquisition and treating a contingent undrawn facility maturity
  as an existing debt obligation. No comprehensive expert labeling audit was done,
  and the questions were not revised after those checks.

## Limits and artifacts

The 2024–2025 results had already been inspected before designing this follow-up.
This is exploratory reuse, not a new untouched test. All bundles are reported;
there was no selection of a favorable horizon, threshold, question subset or
hyperparameter after this run's outcomes. Approximate intervals are descriptive
and do not account for adaptive research or all multiple comparisons.

The original survivorship, adjusted-price, model-pretraining and execution-proxy
limitations remain. We added no prior filings, peer documents or agreement
exhibits, and the Jev training-data cutoff remains unverified.

- [Protocol](fifty-question-protocol.md)
- [Question definitions](../config/fifty_questions.json)
- [Complete model comparison](../results/fifty_questions_v1/REPORT.md)
- [Chart](../results/fifty_questions_v1/comparison.png)
- [Verification](../results/fifty_questions_v1/verification.json)
- [Training question frequencies](../results/fifty_questions_v1/training_question_frequencies.json)

Raw sources remain in data/large_study_v1. New responses, protocol history and
the limited source review are saved in data/fifty_questions_v1.
