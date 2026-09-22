# Training-only audit of Jev features

This descriptive audit uses 1,088 training filings (2018–2022). It reads only
the split, Jev probabilities, and SEC item flags from the saved feature table.
It does not read return labels, rerun inference, refit predictors, or evaluate
the inspected validation/test periods.

The original extraction used three-way `choice` questions, not `noul`:
disclosed, not_disclosed, unclear. The prediction models received disclosed
and unclear probabilities for each question (eight numeric columns).

| Fact | Probability <=10% | Probability >=90% | Largest absolute correlation with an individual item flag |
|---|---:|---:|---:|
| Refinancing | 67.8% | 19.4% | 0.305 (1.02) |
| Maturity extension | 89.2% | 8.9% | 0.169 (8.01; negative) |
| Covenant waiver | 98.3% | 0.7% | 0.054 (2.02) |
| Capital expenditure funding | 95.0% | 1.0% | 0.044 (3.02; negative) |

The probabilities are indeed concentrated near zero or one. Waivers and capital
expenditure funding have particularly few high-positive examples. That limits
the evidence available for learning their relationship with returns. Binary
features can nevertheless be predictive; converting them to graded scores does
not by itself add useful information.

The simple hypothesis that Jev merely reproduced individual filing flags is
not supported by these correlations. This is not proof that the features add
information conditional on every baseline input.

For a descriptive nonlinear check, probabilities were binned at .1, .5, .9.
Mutual information between those bins and the joint pattern of all nine item
flags was 0.270, 0.100, 0.025, and 0.040 bits respectively. Mean shuffled
references were 0.098, 0.059, 0.031, and 0.047 bits. These finite-sample estimates
suggest some overlap for refinancing and maturity extensions, rather than a
complete re-encoding of the flags. Estimates depend on bins, and shuffled
references ignore issuer/date dependence: they are not significance tests.
Neither correlation nor this audit establishes incremental trading information
or explains the previous null result conclusively.

Reproduce with `python3 audit_feature_overlap.py`. Full distributions, every
pairwise association (including unclear probabilities), conditional summaries,
and source hash are in `results/feature_overlap_audit/`.

API documentation checked September 21, 2026:

- https://docs.typesafe.ai/api documents choice distributions and ordered score levels.
- https://docs.typesafe.ai/models lists $0.042/million input tokens, free output,
  and 64k total / 32k state-plus-longest-question token limits.
- https://docs.typesafe.ai/patterns/fan-out says parallel questions usually have
  little effect on latency. Additional question tokens still cost money.
- https://docs.typesafe.ai/concepts/system-one distinguishes typed decisions
  from generated explanations and notes that calibration does not guarantee
  individual correctness. Calibration on these financial questions remains
  an empirical matter. The checked pages do not establish a training-data cutoff.

The next experiment should select economically motivated axes and point-in-time
comparison documents before evaluating returns. Same-week peer context must
exclude filings not yet public at the decision timestamp. The already inspected
2024–2025 results cannot become an untouched test for the revised design.
