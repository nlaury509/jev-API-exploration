# First Jev extraction audit

Run: `data/extractions/20260921T195920726092Z`  
Dataset: `data/runs/20260921T193450927191Z`  
Model: `jev-1.13.0`

Jev can run against this account and turn real SEC filing text into structured
features at low measured latency and estimated cost. This six-document development
audit does not establish predictive value, calibration, or production reliability.

| Measure | Result |
|---|---:|
| Filings completed | 6 / 6 |
| Factual answers | 24 |
| Agreement with prewritten assistant reference | 23 / 24 (95.8%) |
| Always answer “not disclosed” baseline | 19 / 24 (79.2%) |
| Positive reference answers | 5 |
| Positive reference answers detected | 5 / 5 |
| Additional positive prediction | 1 |
| Median sequential API latency per filing | 0.419 seconds |
| Returned input tokens | 21,823 |
| Estimated inference cost | $0.000916566 |

Cost uses returned usage and the [published model rate](https://docs.typesafe.ai/models)
of $0.042 per million input tokens, checked September 21, 2026. It is not a billing
receipt. This measures six sequential requests, not sustained or parallel throughput.

## What the answers show

- Ford: detected the explicit use of DOE financing to construct battery plants.
- United Airlines and Carnival: detected proceeds used to retire existing debt.
- Caterpillar: detected facility replacement and executed maturity extensions.
- Boeing: correctly distinguished a future extension option from an executed
  extension, but disagreed with the refinancing reference.
- Delta: correctly avoided treating an aircraft purchase as disclosure of financing
  specifically funding that purchase.

## The disagreement remains unresolved business interpretation

Boeing disclosed a new $4 billion facility and same-day termination of older
$0.8 billion and $3.2 billion facilities. Jev assigned refinancing probabilities
of 0.63 disclosed, 0.33 not disclosed, and 0.04 unclear. That is a plausible economic
interpretation. Our prewritten rubric required an explicit replacement/repayment
link and prohibited inference merely from old and new debt being mentioned, so the
reference was not disclosed. We retain that original answer and report disagreement;
we have not changed the label or prompt to increase the score.

## Limits and decision

The reference was written by the coding assistant before inference, not by an
independent financial expert. There are only six documents; the 24 questions are
not independent observations. Five reference answers were positive, with no positive
covenant-waiver or unclear examples. Several responses assign probability exactly
zero or one; this sample cannot validate probability calibration. Only primary
8-K text was supplied, not the agreement exhibits.

The always-negative baseline is a sanity check, not a competitive text baseline.
Neither a keyword/rules extractor nor a text-vector classifier has been evaluated,
so incremental value over inexpensive text processing is still unknown. No price
labels, XGBoost comparison, trades, or return estimates were produced.

Proceed to a larger extraction evaluation with fresh, reviewed examples, including
actual waivers and ambiguous cases, and a simple text baseline. Before claiming an
edge, test whether these features improve a numerical model on later dates under
realistic availability and execution assumptions. The existing 76 convenience-sample
filings should be treated as development material, not strong profitability evidence.

## Reproduce and inspect

`python3 jev.py audit` reuses the saved response cache for identical requests.
The run contains exact requests, responses, references, source hashes, failures,
features.csv, summary.json, and a document-by-document REPORT.md. The reference and
question configuration also live in `config/` for review. Thirteen local tests pass,
covering collection behavior, response validation, cache integrity, and scoring.
