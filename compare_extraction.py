#!/usr/bin/env python3
"""Compare saved Jev answers and the frozen rules on exactly the same cases."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from collect import write_json
from jev import digest, validate_response

CLASSES = ('disclosed', 'not_disclosed', 'unclear')


def label_metrics(rows, field):
    matrix = {y: {p: 0 for p in CLASSES} for y in CLASSES}
    for r in rows:
        matrix[r['reference']][r[field]] += 1
    per_class = {}
    for c in CLASSES:
        tp = matrix[c][c]
        support = sum(matrix[c].values())
        predicted = sum(matrix[y][c] for y in CLASSES)
        per_class[c] = {'support': support, 'predicted': predicted,
                        'precision': tp / predicted if predicted else None,
                        'recall': tp / support if support else None}
    correct = sum(r['reference'] == r[field] for r in rows)
    groups = {}
    for r in rows:
        groups.setdefault(r['case_id'], []).append(r['reference'] == r[field])
    return {'answers': len(rows), 'correct': correct,
            'accuracy': correct / len(rows) if rows else None,
            'documents_all_correct': sum(all(v) for v in groups.values()),
            'confusion_matrix_reference_rows': matrix, 'per_class': per_class}


def compare(run: Path):
    config = json.loads((run / 'config.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    reference = json.loads((run / 'reference.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    baseline = json.loads((run / 'baseline_predictions.json').read_text())
    if not summary['complete']:
        raise ValueError('Refusing to compare an incomplete run.')
    if digest(reference) != manifest['reference_sha256'] or digest(config['questions']) != reference['questions_sha256']:
        raise ValueError('Reference integrity mismatch.')
    if hashlib.sha256((run / 'text_baseline.py').read_bytes()).hexdigest() != config['baseline_sha256']:
        raise ValueError('Frozen baseline hash mismatch.')
    rows = []
    for case in manifest['cases']:
        cid = case['case_id']
        r = json.loads((run / 'responses' / f'{cid}.json').read_text())
        request = json.loads((run / 'requests' / f'{cid}.json').read_text())
        if r['request_sha256'] != digest(request) or r['response_sha256'] != digest(r['response']):
            raise ValueError('Saved request/response hash mismatch.')
        if hashlib.sha256(request['state'].encode()).hexdigest() != case['text_sha256'] or reference['cases'][cid]['text_sha256'] != case['text_sha256']:
            raise ValueError('Source text does not match reference.')
        validate_response(r['response'], request)
        for q in config['questions']:
            rows.append({'case_id': cid, 'symbol': case['symbol'], 'question': q,
                         'reference': reference['cases'][cid]['labels'][q]['label'],
                         'jev': r['response']['answers'][q]['choice'],
                         'rules': baseline[cid][q]['choice'], 'constant': 'not_disclosed',
                         'source_path': case['source_path']})
    def metrics(subset):
        return {field: label_metrics(subset, field) for field in ('jev', 'rules', 'constant')}
    paired = Counter(('both_correct' if row['jev'] == row['reference'] and row['rules'] == row['reference']
                      else 'jev_only_correct' if row['jev'] == row['reference']
                      else 'rules_only_correct' if row['rules'] == row['reference']
                      else 'neither_correct') for row in rows)
    result = {'documents': len(manifest['cases']), 'unique_symbols': len({c['symbol'] for c in manifest['cases']}),
              'metrics': metrics(rows), 'paired_counts': dict(paired),
              'per_question': {q: metrics([r for r in rows if r['question'] == q]) for q in config['questions']},
              'disagreements': [r for r in rows if r['jev'] != r['reference'] or r['rules'] != r['reference']],
              'interpretation': 'Descriptive assistant-reference agreement only. Questions and repeated issuers are dependent. Rules emit labels, not calibrated probabilities. Targeted challenge sets must be reported separately from hash-selected samples.'}
    write_json(run / 'comparison.json', result)
    write_json(run / 'comparison_rows.json', rows)
    lines = ['# Extraction comparison', '', result['interpretation'], '',
             '| Model | Correct / answers | Accuracy | Entire documents correct |', '|---|---:|---:|---:|']
    for name, m in result['metrics'].items():
        lines.append(f"| {name} | {m['correct']} / {m['answers']} | {m['accuracy']:.1%} | {m['documents_all_correct']} / {result['documents']} |")
    lines.extend(['', '## Disagreements', '', '| Case | Question | Reference | Jev | Rules |', '|---|---|---|---|---|'])
    for r in result['disagreements']:
        lines.append(f"| {r['symbol']} {r['case_id']} | {r['question']} | {r['reference']} | {r['jev']} | {r['rules']} |")
    (run / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'run': str(run), 'metrics': {name: {k: v for k, v in m.items() if k not in ('confusion_matrix_reference_rows', 'per_class')} for name, m in result['metrics'].items()}, 'paired_counts': dict(paired)}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    compare(parser.parse_args().run)
