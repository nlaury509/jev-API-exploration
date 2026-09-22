#!/usr/bin/env python3
"""Frozen, exploratory 0/4/10/25/50-question ablation on existing sources."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect import ROOT, load_env, now, write_json
from jev import JevClient, canonical, choice_probabilities, digest, payload_for, validate_response

SOURCE = ROOT / 'data/large_study_v1'
OLD = ROOT / 'results/large_study_v1'
STUDY = ROOT / 'data/fifty_questions_v1'
OUT = ROOT / 'results/fifty_questions_v1'
CONFIG = ROOT / 'config/fifty_questions.json'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration():
    cfg = json.loads(CONFIG.read_text())
    original = json.loads((ROOT / 'config/extraction.json').read_text())
    if len(cfg['questions']) != 50 or cfg['question_order'] != list(cfg['questions']):
        raise ValueError('Expected exactly 50 questions in the frozen order.')
    if {q: cfg['questions'][q] for q in cfg['question_order'][:4]} != original['questions']:
        raise ValueError('Original four questions changed.')
    if any(q['type'] != 'choice' for q in cfg['questions'].values()):
        raise ValueError('This experiment uses choice distributions, including ordered rubrics.')
    return cfg


def prepare():
    cfg = configuration()
    STUDY.mkdir(parents=True, exist_ok=True)
    paths = [CONFIG, OLD / 'events_with_features_and_labels.csv', OLD / 'summary.json',
             OLD / 'test_predictions.csv', SOURCE / 'events.json', SOURCE / 'bars.csv',
             ROOT / 'config/large_study.json', ROOT / 'event_study.py', ROOT / 'jev.py', Path(__file__)]
    manifest = {'config': cfg, 'source_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
                'interpretation': cfg['status'], 'original_four': 'Cached original probabilities, identical across all nested feature sets.',
                'repeated_four': 'Asked again inside the 50-question request only to report batch-context stability; not used by predictors.'}
    path = STUDY / 'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != 'created_at'} != manifest:
            raise ValueError('Frozen configuration, source or implementation changed. Do not overwrite this run.')
    else:
        write_json(path, manifest | {'created_at': now()})
        write_json(STUDY / 'config.json', cfg)
    return cfg


def checked_record(event, cfg, path):
    text = (SOURCE / event['text_path']).read_text()
    if hashlib.sha256(text.encode()).hexdigest() != event['text_sha256']:
        raise ValueError('Source text changed.')
    payload = payload_for(text, cfg)
    if path is None:
        return payload
    record = json.loads(path.read_text())
    if (record['accession'] != event['accession'] or record['text_sha256'] != event['text_sha256']
            or record['request_sha256'] != digest(payload)
            or record['response_sha256'] != digest(record['response'])):
        raise ValueError('Saved source, request or response integrity mismatch.')
    # These learners use probability distributions, not the selected label.
    # Preserve and separately report API label/argmax inconsistencies.
    validate_response(record['response'], payload, require_argmax=False)
    return record


def extract(limit=None):
    cfg = prepare()
    events = json.loads((SOURCE / 'events.json').read_text())
    destination = STUDY / 'jev'
    destination.mkdir(exist_ok=True)
    records, pending = [], []
    for event in events:
        path = destination / (event['accession'] + '.json')
        if path.exists():
            records.append(checked_record(event, cfg, path))
        else:
            pending.append((event, checked_record(event, cfg, None)))
    if limit is not None:
        if limit < 1:
            raise ValueError('Limit must be positive.')
        pending = pending[:limit]
    spent = sum(r['response']['usage']['input_tokens'] for r in records) / 1e6 * cfg['input_usd_per_million_tokens']
    start = time.perf_counter()
    initial = len(records)
    failures = []

    def work(item):
        event, payload = item
        record = JevClient().predict(payload, require_argmax=False) | {k: event[k] for k in ['accession', 'symbol', 'text_sha256']}
        write_json(destination / (event['accession'] + '.json'), record)
        return record

    def save():
        summary = {'created_at': now(), 'completed': len(records), 'requested': len(events),
                   'questions_per_document': 50, 'decisions': len(records) * 50,
                   'complete': len(records) == len(events), 'failures': failures,
                   'estimated_total_usd': spent,
                   'total_input_tokens': sum(r['response']['usage']['input_tokens'] for r in records),
                   'new_records_this_attempt': len(records) - initial,
                   'elapsed_seconds_this_attempt': time.perf_counter() - start,
                   'cost_note': 'Successful responses only; failed requests may also be billed.'}
        write_json(STUDY / 'inference_summary.json', summary)
        return summary

    previous = STUDY / 'inference_summary.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        write_json(STUDY / 'attempts' / (digest(old) + '.json'), old)
    with ThreadPoolExecutor(max_workers=cfg['inference_workers']) as pool:
        for offset in range(0, len(pending), cfg['inference_workers']):
            batch = pending[offset:offset + cfg['inference_workers']]
            reservation = sum(len(canonical(p)) for _, p in batch) / 1e6 * cfg['input_usd_per_million_tokens']
            if spent + reservation > cfg['max_estimated_inference_usd']:
                failures.append({'error': 'Configured budget reached.'})
                break
            futures = [pool.submit(work, item) for item in batch]
            for item, future in zip(batch, futures):
                try:
                    record = future.result()
                    records.append(record)
                    spent += record['response']['usage']['input_tokens'] / 1e6 * cfg['input_usd_per_million_tokens']
                except Exception as exc:
                    failures.append({'accession': item[0]['accession'], 'error': str(exc)})
            if offset % 40 == 0 or failures:
                save()
                print(f'50 questions: {len(records)}/{len(events)} filings, estimated ${spent:.4f}', flush=True)
            if failures:
                break
    summary = save()
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError('Extraction stopped; inspect saved failures before resuming.')


def answer_features(answers, cfg):
    """Keep n-1 probabilities; preserve unknown/unclear information implicitly."""
    result = {}
    for q in cfg['question_order'][4:]:
        probs = choice_probabilities(answers[q])
        omitted = 'not_disclosed' if 'not_disclosed' in probs else 'unknown'
        if omitted not in probs:
            raise ValueError('Expected a defined reference category.')
        result.update({f'extra__{q}__{label}': p for label, p in probs.items() if label != omitted})
    return result


def nested_columns(frame, cfg):
    base = [c for c in frame if c.startswith(('m_', 'spy_feature_', 'meta_'))]
    old = [c for c in frame if c.startswith('jev_')]
    if len(old) != 8:
        raise ValueError('Expected eight original numeric features.')
    result = {}
    for count in cfg['question_counts']:
        wanted = set(cfg['question_order'][4:count])
        extra = [c for c in frame if c.startswith('extra__') and c.split('__')[1] in wanted]
        result[str(count)] = base + (old if count else []) + extra
    return result


def evaluate():
    cfg = prepare()
    if (OUT / 'summary.json').exists():
        raise ValueError('Exploratory results already exist. Do not silently rerun or retune.')
    inference = json.loads((STUDY / 'inference_summary.json').read_text())
    if not inference['complete']:
        raise ValueError('Finish all requested inference first; no outcome-dependent case selection.')
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from event_study import paired_block_interval, price_panels, scores, simulate

    old_cfg = json.loads((ROOT / 'config/large_study.json').read_text())
    frame = pd.read_csv(OLD / 'events_with_features_and_labels.csv', float_precision='round_trip',
                        parse_dates=['entry_date', 'exit_date', 'feature_date'])
    events = json.loads((SOURCE / 'events.json').read_text())
    extracted, repeated, raw_answers, latencies, argmax_mismatches = [], [], [], [], []
    for event in events:
        record = checked_record(event, cfg, STUDY / 'jev' / (event['accession'] + '.json'))
        answers = record['response']['answers']
        extracted.append({'accession': event['accession'], **answer_features(answers, cfg)})
        latencies.append(record['latency_seconds'])
        original = json.loads((SOURCE / 'jev' / (event['accession'] + '.json')).read_text())
        for q in cfg['question_order']:
            probabilities = choice_probabilities(answers[q])
            if probabilities[answers[q]['choice']] < max(probabilities.values()) - 1e-6:
                argmax_mismatches.append({'accession': event['accession'], 'question': q,
                    'raw_choice': answers[q]['choice'], 'probabilities': probabilities})
            raw_answers.append({'accession': event['accession'], 'question': q, 'choice': answers[q]['choice'],
                                **{'p_' + k: v for k, v in probabilities.items()}})
            if q in cfg['question_order'][:4]:
                previous = choice_probabilities(original['response']['answers'][q])
                repeated.append({'accession': event['accession'], 'question': q,
                    'choice_agrees': answers[q]['choice'] == original['response']['answers'][q]['choice'],
                    'max_probability_change': max(abs(probabilities[k] - previous[k]) for k in previous)})
    frame = frame.merge(pd.DataFrame(extracted), on='accession', how='left', validate='one_to_one', sort=False)
    columns = nested_columns(frame, cfg)
    inputs = columns['50']
    if not np.isfinite(frame[inputs].to_numpy()).all():
        raise ValueError('Missing or invalid features.')
    if not (frame.feature_date < frame.entry_date).all():
        raise ValueError('Features use future price observations.')
    train, val, test = [frame.loc[frame.split == name] for name in ['train', 'validation', 'test']]
    sessions, panels = price_panels(pd.read_csv(SOURCE / 'bars.csv'))
    test_sessions = sessions[(sessions >= old_cfg['validation_end']) & (sessions < old_cfg['test_end'])]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'manifest.json', json.loads((STUDY / 'manifest.json').read_text()))
    write_json(OUT / 'feature_columns.json', columns)
    write_json(OUT / 'software.json', {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__})
    (OUT / 'evaluation_code.py').write_bytes(Path(__file__).read_bytes())
    pd.DataFrame(raw_answers).to_csv(OUT / 'answers.csv', index=False)
    pd.DataFrame(repeated).to_csv(OUT / 'repeated_four.csv', index=False)
    write_json(OUT / 'api_argmax_mismatches.json', argmax_mismatches)
    frame.to_csv(OUT / 'events_with_features_and_labels.csv', index=False)
    # Variability audit is training-only and never selects or drops features.
    train[inputs].agg(['mean', 'std', 'min', 'max']).T.to_csv(OUT / 'training_feature_distributions.csv')
    predictions = test[['accession', 'symbol', 'entry_date', 'exit_date', 'target', 'excess_return']].copy()
    validation = val[['accession', 'target']].copy()
    metrics, portfolios, curves, probabilities = {}, {}, {}, {}
    old_predictions = pd.read_csv(OLD / 'test_predictions.csv').set_index('accession')
    for learner in ['boosting', 'logistic']:
        for count, features in columns.items():
            name = learner + '__q' + count
            if learner == 'boosting':
                model = HistGradientBoostingClassifier(**{k: v for k, v in old_cfg['learner'].items() if k != 'name'})
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=old_cfg['logistic_C'], max_iter=3000, random_state=old_cfg['seed']))
            model.fit(train[features], train.target)
            vp = model.predict_proba(val[features])[:, 1]
            tp = model.predict_proba(test[features])[:, 1]
            if count in ['0', '4']:
                old_name = learner + ('__market_and_metadata' if count == '0' else '__market_metadata_jev')
                expected = old_predictions.loc[test.accession, old_name].to_numpy()
                if not np.allclose(tp, expected, rtol=1e-8, atol=1e-10):
                    raise ValueError('Original baseline predictions did not reproduce.')
            metrics[name] = {'question_count': int(count), 'numeric_features': len(features),
                'train': scores(train.target, model.predict_proba(train[features])[:, 1]),
                'validation': scores(val.target, vp), 'exploratory_test': scores(test.target, tp),
                'test_by_year': {str(y): scores(g.target, tp[test.entry_date.dt.year.to_numpy() == y]) for y, g in test.groupby(test.entry_date.dt.year)}}
            probabilities[name] = tp
            predictions[name] = tp
            validation[name] = vp
            stats, curve, trades = simulate(test, tp, float(np.quantile(vp, .8)), test_sessions, panels, old_cfg['portfolio'])
            portfolios[name] = stats
            curves[name] = curve
            curve.to_csv(OUT / (name + '_equity.csv'), index=False)
            pd.DataFrame(trades).to_csv(OUT / (name + '_trades.csv'), index=False)
            print(f'{name}: log loss {metrics[name]["exploratory_test"]["log_loss"]:.5f}, net proxy return {stats["net_return"]:.2%}', flush=True)
    comparisons = {learner + '__50_minus_' + base: paired_block_interval(test,
        probabilities[learner + '__q' + base], probabilities[learner + '__q50'], test_sessions, old_cfg)
        for learner in ['boosting', 'logistic'] for base in ['0', '4']}
    repeated_frame = pd.DataFrame(repeated)
    summary = {'created_at': now(), 'status': cfg['status'], 'inference': inference,
        'split_counts': frame.groupby('split').size().to_dict(), 'metrics': metrics,
        'portfolios': portfolios, 'paired_comparisons': comparisons,
        'train_prior': scores(test.target, np.full(len(test), train.target.mean())),
        'repeated_four_choice_agreement': float(repeated_frame.choice_agrees.mean()),
        'repeated_four_mean_max_probability_change': float(repeated_frame.max_probability_change.mean()),
        'median_latency_seconds': float(np.median(latencies)),
        'api_label_argmax_mismatches': len(argmax_mismatches),
        'limits': ['Previously inspected 2024–2025 data; exploratory comparison, not fresh confirmation.',
                  'Question count ordering is fixed but not random; this compares particular feature bundles.',
                  'Only original financing-selected 8-K primary text, no prior filings or exhibits added.',
                  'New questions have no comprehensive independent accuracy or calibration audit.',
                  'Current-universe survivorship bias, retrospective adjusted prices, unknown model training cutoff.',
                  'All original learner parameters, horizon and portfolio assumptions kept fixed.',
                  'Intervals are descriptive, not adjusted for multiple comparisons or adaptive research.',
                  'Simulated daily-open proxies with costs, not executable trading results.']}
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    validation.to_csv(OUT / 'validation_predictions.csv', index=False)
    write_json(OUT / 'summary.json', summary)
    render(summary, curves)
    print('Saved exploratory comparison:', OUT, flush=True)


def render(summary, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, label in [('train', 'Training'), ('validation', '2023 validation'), ('exploratory_test', '2024–25 inspected test')]:
        counts = [0, 4, 10, 25, 50]
        axes[0].plot(counts, [summary['metrics'][f'boosting__q{n}'][split]['log_loss'] for n in counts], '-o', label=label)
    axes[0].axhline(summary['train_prior']['log_loss'], ls='--', color='gray', label='Constant baseline on test')
    axes[0].set(xlabel='Questions (fixed nested bundles)', ylabel='Log loss — lower is better', title='Does adding questions help?')
    for n in [0, 4, 10, 25, 50]:
        curve = curves[f'boosting__q{n}']
        axes[1].plot(curve.date, curve.equity, label=f'{n} questions')
    axes[1].set(title='Costed simulated portfolio', ylabel='Equity, starting at 1')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Exploratory historical reuse — gradient boosting, unchanged parameters')
    fig.tight_layout(); fig.savefig(OUT / 'comparison.png', dpi=150); plt.close(fig)
    lines = ['# Fifty-question exploratory experiment', '', summary['status'], '',
        'Original four cached probabilities are held fixed; 46 new questions are added. All 50 were asked per new API call. Repeated originals measure batch-context stability only.', '',
        '| Learner | Questions | Numeric inputs | Train log loss | Validation log loss | Inspected-test log loss | Net simulated return | Trades |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['metrics'].items():
        p = summary['portfolios'][name]
        lines.append(f'| {name.split("__")[0]} | {row["question_count"]} | {row["numeric_features"]} | {row["train"]["log_loss"]:.5f} | {row["validation"]["log_loss"]:.5f} | {row["exploratory_test"]["log_loss"]:.5f} | {p["net_return"]:.2%} | {p["trades"]} |')
    lines += ['', f'Constant training-prior baseline inspected-test log loss: {summary["train_prior"]["log_loss"]:.5f}.', '',
        'Paired differences below are 50 questions minus comparator; negative log-loss differences favor 50.', '']
    for name, row in summary['paired_comparisons'].items():
        lines.append(f'- {name}: {row["delta_log_loss"]:+.5f}; approximate 95% date-block interval {row["approximate_95pct_date_block_interval"]}.')
    lines += ['', f'Inference: {summary["inference"]["completed"]} documents × 50 questions; estimated ${summary["inference"]["estimated_total_usd"]:.4f} at configured published pricing. Successful requests only; failed calls may cost extra.',
        f'Median recorded request latency: {summary["median_latency_seconds"]:.3f}s. Repeated original-question choice agreement: {summary["repeated_four_choice_agreement"]:.2%}.',
        f'API selected-label/argmax inconsistencies: {summary["api_label_argmax_mismatches"]}; raw responses preserved, only validated probability distributions used by predictors.', '',
        'Training 2018–2022 (1,088 events), validation 2023 (138), previously inspected test 2024–2025 (321). Next-session entry and five-session holding period, original cost and risk assumptions unchanged.', '',
        '## Limits', ''] + ['- ' + s for s in summary['limits']] + ['', '![Comparison](comparison.png)']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'extract', 'evaluate'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'extract':
        load_env(); extract(args.limit)
    elif args.command == 'prepare':
        prepare(); print('Frozen 50-question protocol and source hashes.')
    else:
        evaluate()
