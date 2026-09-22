# Frozen exploratory fifty-question comparison

The user requested a 50-question run to test whether broader extraction helps.
This design is written before the expanded extraction or new predictive results.
The 2024–2025 outcomes have already been inspected in the original experiment.
Every new result is exploratory; no fresh holdout is claimed.

Use all 1,560 existing eligible primary 8-K documents, unchanged. Ask 50 choice
questions in a single request per document using pinned Jev 1.13.0. The first
four questions are identical to the original questions. The remaining 46 cover
liquidity, debt amount/terms, dilution, transaction stage, stress, financing
structure, capital allocation and adverse or favorable business developments.
Multi-level judgments include unknown categories; fact questions distinguish
not disclosed from ambiguous. No external knowledge or future information is
requested. Full definitions and fixed order live in config/fifty_questions.json.

Compare the fixed nested bundles 0, 4, 10, 25, 50. Preserve the original four
cached probability inputs in every applicable bundle so their resampling does
not confound the additional-feature comparison. Repeated answers to those four
questions in the new batch are a descriptive batch-context stability check,
not an identical-request repeatability experiment. Preserve n-1 probabilities
for each additional question. Missing fields cause failure, never imputation
with outcomes. Retain all fixed features; no return-guided question selection.

Use the same matched 1,547 rows, chronological boundaries and purges as before.
Train 2018–2022, use 2023 only for each learner's pre-existing 80th percentile
trade cutoff, and compare on the already inspected 2024–2025 period. Keep both
original learners and all hyperparameters unchanged. Primary descriptive
comparison: boosting 50 minus boosting 4 in test log loss. Secondary: 50 minus
0, intermediate bundles, logistic regression, training-versus-later loss, and
the same costed stock-long/SPY-short portfolio diagnostic. Report all bundles,
including unfavorable results. Approximate date-block intervals are neither
fresh confirmatory tests nor adjusted for adaptive research/multiplicity.

The command refuses to overwrite completed results or change frozen source,
question, or implementation hashes. Cached responses are validated. A bounded
three-document API smoke check precedes bulk inference; no returns are used in
that check. Four workers and a $5 successful-usage guard are ample for this run;
failed requests may also be billed. No brokerage endpoints are used.

This tests question breadth on a financing-selected corpus, not arbitrary
financial text, cross-document comparisons or a new data source. Many expanded
questions may be absent in this corpus. There is no full expert labeling audit
of the new questions, no established task-specific calibration, and no verified
Jev training-data cutoff. More questions do not imply independent information.

API definitions/pricing verified against https://docs.typesafe.ai/api and
https://docs.typesafe.ai/models on September 21, 2026.

## Pre-evaluation response-handling amendment

After 98 documents, a raw API answer selected `unknown` with probability .47
despite assigning `increase` .48. No expanded predictive outcomes had been
computed. Because this experiment exclusively consumes the probability vector,
the new experiment permits selected-label/argmax mismatches while retaining
all schema, range, sum, model, usage and hash checks. Raw outputs are unchanged;
mismatches are separately counted and saved. The general adapter remains strict
by default. The original frozen manifest is archived, and the amendment records
the old/new implementation hashes. Questions, feature mapping, splits, learners,
costs and evaluation remain unchanged. This is an API integrity issue, not an
assumption that the chosen label is correct or that the probabilities calibrate.
