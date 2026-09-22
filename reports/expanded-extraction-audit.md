# Expanded Jev extraction audit

## Result

Jev matched all 72 prewritten assistant-reference answers on 18 fresh filings,
versus 64 for the fixed text rules. A separate three-filing waiver challenge produced
11/12 agreement for Jev versus 9/12 for the rules. Jev identified all three actual
waivers or forbearances. This supports the feasibility of interpreting text into
features; it does not demonstrate that those features predict returns.

| Evaluation set | Documents | Jev agreement | Text rules | Always not disclosed |
|---|---:|---:|---:|---:|
| Fresh hash-selected filings | 18 | 72/72 (100%) | 64/72 (88.9%) | 63/72 (87.5%) |
| Targeted waiver challenge | 3 | 11/12 (91.7%) | 9/12 (75.0%) | 7/12 (58.3%) |

The original six filings are excluded from both rows. There are now 27 reviewed
filings in total. The targeted examples were selected because they contain waivers,
so the two sets should not be pooled into a population accuracy estimate.

## How the comparison was conducted

- Kept the original four questions, their criteria, and pinned `jev-1.13.0` unchanged.
- Selected the next 18 eligible filings by the original accession-hash ranking,
  excluding all six previously reviewed accessions. No return data informed selection.
- Wrote and hashed the sentence/keyword baseline before reading the new sample.
  Its rules were not changed after errors were observed.
- Reviewed complete primary document texts and saved reference labels, notes, and
  positive evidence quotes before each batch's Jev calls. The labeling assistant
  also designed the baseline, so this is not a blinded, independent expert evaluation.
- Required complete reference coverage, matching question/source hashes, and
  verified evidence quotes before making calls. Saved both models' outputs on the
  same source texts, alongside exact requests, probabilities, timing, and usage.
- Reported per-question and per-class metrics as well as overall accuracy. The
  rules return categorical labels, so their outputs are not treated as calibrated
  probabilities.

## Where Jev helped

The fixed rules made eight errors in the 18-document sample; Jev agreed with the
reference on each of those answers:

- Three false positives treated tax-plan expiration extensions as debt extensions.
- Three false positives treated generic capex/financing risk language as a stated
  use of a particular financing transaction.
- One missed a stated use of revolving borrowings to fund retirement of convertible
  notes, because the relationship was described across sentences.
- One missed financing for a cruise ship, a capital asset described without the
  rule's specific capex vocabulary.

There were only nine positive reference answers in that sample. Jev identified all
nine without additional positives. The rules identified seven, missed two, and
produced six false positives: positive precision 53.8%, recall 77.8%. Always answering
not disclosed achieves 87.5% accuracy but misses all nine positives.

The baseline is a small, untuned rules system. This result does not establish that
Jev beats a carefully engineered extractor, a trained text classifier, or another
language model. It does provide a concrete example of the semantic interpretation
the project is intended to test.

## The targeted cases and remaining disagreement

The challenge uses original SEC submissions from
[BIOLASE](https://www.sec.gov/Archives/edgar/data/811240/000095017024104387/biol-20240831.htm),
[iRobot](https://www.sec.gov/Archives/edgar/data/1159167/000119312525136125/d42654d8k.htm),
and [Audacy](https://www.sec.gov/Archives/edgar/data/1067837/000119312524201513/d883977d8k.htm).
These are historical issuer labels, not current tradeable-symbol assertions or an
addition to the liquid-stock universe. No prices were requested for this set.

Jev identified each actual waiver/forbearance. The rules identified two, missing
iRobot's wording, and incorrectly treated iRobot's discussion of an inability to
refinance as an intended refinancing.

Both methods disagreed with the BIOLASE refinancing reference. The filing describes
a specific contemplated bridge-loan roll-up into future bankruptcy financing. The
reference counts this as intended refinancing under the existing rubric. Jev chose
not disclosed, with P(disclosed) = 0.29. The funding is conditional and this is a
judgment-sensitive boundary, so retain the original score and flag the label for
independent review; do not silently relabel it to improve agreement.

## Cost, speed, and verification

The 21 new calls returned 74,607 input tokens, with estimated inference cost of
**$0.003133494**, using the configured [published price](https://docs.typesafe.ai/models)
of $0.042 per million input tokens. This is an estimate, not a billing receipt.
Median sequential request latency was **0.400 seconds**. No sustained parallel
throughput measurement was performed. Identical requests reuse the local cache.

Seventeen local tests pass, including chronological timestamp handling, credential
boundaries, previous-case exclusion, reference requirements, schema and cache
validation, positive-class scoring, and named-filing collection without price calls.

## Research limits and next decision

These remain small development samples with assistant-authored reference labels,
repeated issuers and similar document templates. The 72 answers are not 72 independent
observations. There are no unclear reference labels, and only three targeted positive
waiver examples. Perfect agreement cannot establish broad reliability or calibration.
Only primary 8-K text was supplied; exhibits were excluded. Historical model-training
contamination also remains possible.

The next useful question is whether the extracted information adds predictive value,
not whether another small extraction sample can produce a higher score. Define one
event-based return target and timing/cost assumptions before examining returns,
expand the number and diversity of events, and compare numerical features alone
against numerical features plus these text features on later dates. Include filing-item
metadata and a cheap text baseline so any improvement is attributable to semantic
features rather than simply knowing a filing occurred. A live paper observation
period would help address historical model contamination.

Do not use the 76 original convenience-sample filings as a convincing trading test.
The present result justifies that next research step; it provides no evidence yet
of profitable trades or superiority over XGBoost.

## Artifacts and reproduction

Fresh sample:

```bash
python3 jev.py audit --config config/expanded_extraction.json --reference config/expanded_reference.json
python3 compare_extraction.py data/extractions/20260921T200528694224Z
```

Targeted set:

```bash
python3 collect.py filings --config config/waiver_sources.json
python3 jev.py audit --dataset data/runs/20260921T200820502891Z --config config/waiver_extraction.json --reference config/waiver_reference.json
python3 compare_extraction.py data/extractions/20260921T200946235135Z
```

Each new collection/audit creates a new timestamp directory; use its printed path
to inspect that rerun. The paths above identify the runs underlying this report.
`comparison.json` contains per-class/per-question metrics and disagreements;
`comparison_rows.json` contains every paired answer. `baseline_predictions.json`
includes sentences matched by the rules. Source and response caches live under `data/`.
