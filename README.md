# Jev trading research: data and extraction pilot

This first milestone collects SEC 8-K filings concerning agreements and financing,
their embedded EX-10/EX-99 text exhibits, and Alpaca historical stock bars. It does
not predict prices or place orders. A separate Jev adapter audits factual
extraction from primary filing text; there are no paid model calls in the collector.

Python 3.11+ is sufficient. Collection and extraction use the standard library.
The larger predictive study uses the packages pinned in `requirements.txt`.

## Run

Credentials are read from `.env` or environment variables. The local `.env` is
Git-ignored and should have permissions `600`. See `.env.example` for names.
SEC requests include the contact identity in `SEC_USER_AGENT`; Alpaca credentials
are only sent to `data.alpaca.markets`, never to SEC or a trading endpoint.

```bash
cd ~/Desktop/Projects/jev-trading-research
python3 collect.py check
python3 collect.py pilot --limit 10
python3 collect.py pilot
python3 -m unittest discover -s tests -v
```

`check` performs fresh, read-only requests and prints no credentials. `pilot`
defaults to at most 200 original filings. `config/pilot.json` controls symbols,
filing dates (start inclusive, end exclusive), item filters, sample size, sampling
seed, and bars. `1Day` keeps the first audit small; `5Min` is also supported.
Selection is reproducible by accession hash across all candidate issuers, not by
subsequent returns. Amendments (`8-K/A`) are excluded from the initial sample.
The configured 30-company universe is an engineering audit sample across several
industries, not a universe selected by trading performance. A request for 200
filings means **up to** 200; narrower filters can yield fewer. The first 12-company
trial yielded only three matching filings, so the default was broadened without
changing dates or item filters. XOM was omitted after the current SEC mapping
resolved to a new holding-company entity without the requested historical filings.

## Output

Each collection produces a new `data/runs/<UTC timestamp>/` directory:

- `config.json`: exact collection settings, without secrets.
- `selected_filings.json`: chosen filing metadata before downloading documents.
- `events.json`: normalized metadata and pointers to documents.
- `filings/<accession>/submission.txt`: original complete SEC submission.
- `filings/<accession>/document_*.txt`: readable 8-K and EX-10/EX-99 text.
- `filings/<accession>/metadata.json`: acceptance time, document types, source
  URL, retrieval time, original response hash, and unsupported exhibit flags.
- `bars.csv`: sorted, deduplicated UTC OHLCV, including SPY as a benchmark.
- `discovery_sources.json`, `price_sources.json`: response provenance.
- `failures.json`, `summary.json`, `REPORT.md`: collection and coverage audit.

Raw HTTP responses are cached with hashes in `data/cache/`. Re-running reuses
that snapshot. Requests are sequential and spaced by at least 0.4 seconds. No
authentication headers are written to the cache or report. A partial download is
reported as incomplete rather than silently treated as a complete sample.

## Interpretation and limits

- This is a convenience universe for checking the pipeline, not a representative
  or point-in-time universe. Current ticker-to-CIK mappings must be reviewed for
  historical identity changes. Alpaca automatic symbol remapping is disabled.
- A filing's acceptance time is neither necessarily its first public announcement
  nor its exact dissemination time. Never use the underlying event date as the
  time an investor could have read the filing. Model a publication/processing lag
  when predictions are added.
- Prices use consolidated `sip` data, with `adjustment=raw`. Historical requests
  end at least 20 minutes before now. Corporate actions, dividends, missing bars,
  exchange holidays, and session boundaries need handling before computing
  financial labels. Daily bar close and volume become known after the bar ends,
  not at its start timestamp. This collector intentionally generates no labels.
- Coverage reports count and date ranges; they are not a missing-bar audit against
  an exchange calendar. Unsupported PDF exhibits are flagged; other exhibit types
  and externally referenced agreements are not automatically fetched. Raw complete
  submissions are retained for later inspection.
- Historical vendor data can contain corrections made after the event. Jev's
  pretraining can also contain historical outcomes. This pilot cannot establish
  leakage-free predictive skill or profitability.

## Jev extraction audit

Set `TYPESAFE_API_KEY` in `.env`, then run:

```bash
python3 jev.py check
python3 jev.py prepare
python3 jev.py audit
```

The default dataset is the collected 76-filing run. Override it with `--dataset`.
`prepare` writes requests without calling the API. `audit` makes up to six paid
requests, four factual questions per filing, using `config/extraction.json`.
Questions cover refinancing, executed maturity extensions, covenant waivers, and
explicit financing for capital expenditures. Each answer preserves disclosed,
not disclosed, and unclear probabilities. No prices or future outcomes are sent.
Whole primary documents are supplied; oversized documents are skipped and logged,
never silently truncated. Exhibits are excluded from this first model audit.

Reference answers in `config/extraction_reference.json` were assigned by the coding
assistant from source text before inference. These are development labels, not
independent expert judgments. The six documents are hash-selected, with just five
positive reference answers and no positive covenant-waiver example. That limits
what even perfect agreement could show. Several rules deliberately require explicit
wording rather than financial inference; review the rubric alongside disagreements.

Results in `data/extractions/<timestamp>/` contain exact requests, responses,
source and reference hashes, features.csv, metrics, latency, cost estimates, and
REPORT.md. Responses are cached by request hash to avoid paying for identical
replays. Failed POSTs are not automatically retried. API keys are not logged.
Estimated costs use returned input tokens and published pricing, not billing receipts.

The [official API](https://docs.typesafe.ai/api) supports typed choices. We pin
`jev-1.13.0`; [model documentation](https://docs.typesafe.ai/models) says versioned
IDs are accepted even though the account model list currently exposes aliases.

## Next experiment

The [first extraction audit](reports/first-extraction-audit.md) completed six filings.
The [expanded comparison](reports/expanded-extraction-audit.md) adds 18 fresh filings
and three separately reported waiver cases. Jev matched 72/72 fresh-sample reference
answers versus 64/72 for fixed text rules, and 11/12 versus 9/12 on the targeted set.
These assistant-reference development results are not evidence of a trading edge.

```bash
python3 jev.py audit --config config/expanded_extraction.json --reference config/expanded_reference.json
python3 compare_extraction.py data/extractions/20260921T200528694224Z
```

The expanded configuration excludes the first six cases and requires complete
prewritten references. Its frozen baseline hash is checked before calls. The
comparison operates only on a complete saved run and writes per-class/per-question
metrics and every paired answer. Pass the printed audit path to compare a new run.
`collect.py filings --config config/waiver_sources.json` downloads the targeted
original SEC submissions without market prices; see the expanded report for details.

Next, define a return target and execution timing before examining returns, expand
the event dataset, and compare a numerical baseline against the same baseline plus
text features on later, held-out dates. Include filing metadata and cheap text
features as controls. Extraction accuracy and forecasting value are separate tests.

## Large predictive study

**Completed:** [findings](reports/large-study-findings.md). The study processed
1,560 filings and evaluated 321 later test events. It did not establish useful
predictive improvement from the four Jev features; the costed portfolio diagnostic
performed worse. Estimated successful-response inference cost was $0.2392.

The [frozen protocol](reports/large-study-protocol.md) defines an event-based test
across 149 configured companies and 2018–2025 filings. It compares fixed gradient
boosting and logistic models with market/filing metadata alone, plus text rules,
and plus the four audited Jev concepts. It uses 2018–2022 for training, 2023 for
validation and 2024–2025 for testing, purging labels across boundaries.

```bash
python3 large_study.py collect
python3 large_study.py extract
python3 event_study.py
```

Collection resumes from cached sources under `data/large_study_v1`. Extraction
resumes from saved answers, using four workers. Run extraction again after collection
finishes if an earlier extraction covered only a partial download snapshot. Evaluation
requires complete inference and refuses to overwrite a finished test. Results,
predictions, exclusions, model-input columns, costed portfolio diagnostics and charts
are written to `results/large_study_v1`. The portfolio uses adjusted daily-open
proxies and estimated costs; it is not an execution backtest. No brokerage calls occur.

The four audited concepts are refinancing, executed maturity extension, covenant
waiver, and financing of capex. This experiment does not test arbitrary text inputs.

## Fifty-question exploratory follow-up

**Completed:** [findings](reports/fifty-question-findings.md). The run produced
78,000 answers for approximately $0.90. Fifty questions did not establish better
prediction quality than four; all five boosting portfolios lost money after
estimated costs. More features did not produce a monotonic improvement.

The [frozen follow-up protocol](reports/fifty-question-protocol.md) compares
0, 4, 10, 25 and 50 questions on the same matched historical rows. The full
[question definitions](config/fifty_questions.json) include financing terms,
liquidity, dilution, transaction stage and business developments. Each new call
asks all 50 questions. Predictor inputs retain the original cached four answers
and add probabilities from the remaining questions, avoiding a resampling
confound. Both original learners, costs, horizon and trading rules stay fixed.

```bash
python3 feature_study.py prepare
python3 large_study.py extract50 --limit 3
python3 large_study.py extract50
python3 feature_study.py evaluate
```

The 2024–2025 outcomes were already inspected, so this is exploratory research,
not a new untouched test. Saved results go to `results/fifty_questions_v1`;
the evaluator refuses to overwrite a completed comparison. Sources are reused
without additional downloads. Raw responses, source/request hashes, request
costs and a limited post-response extraction spot check are retained under
`data/fifty_questions_v1`. The API occasionally returns a selected label that
does not match the largest probability; this experiment records those cases
and uses the validated probability vectors, preserving raw responses. The
general Jev adapter remains strict by default.

## Sources

- [SEC submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [SEC fair-access guidelines](https://www.sec.gov/about/developer-resources)
- [SEC guide to 8-Ks](https://www.sec.gov/files/readan8k.pdf)
- [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Alpaca SIP access and free-plan restriction](https://docs.alpaca.markets/us/docs/market-data-faq)

Data access is free under the documented options. Jev inference is a separate cost.
