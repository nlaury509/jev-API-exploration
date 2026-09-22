"""Descriptive training-only feature audit. No return labels or API calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def main():
    root = Path(__file__).resolve().parent
    source = root / 'results/large_study_v1/events_with_features_and_labels.csv'
    data = pd.read_csv(source, usecols=lambda c: c == 'split' or c.startswith(('jev_', 'meta_item_')))
    data = data.loc[data['split'] == 'train'].copy()
    flags = [c for c in data if c.startswith('meta_item_')]
    features = [c for c in data if c.startswith('jev_')]
    out = root / 'results/feature_overlap_audit'
    out.mkdir(exist_ok=True)
    # Fixed bins: <=.1, (.1,.5], (.5,.9], >.9. Descriptive MI in bits,
    # with shuffled references for finite-sample bias, not significance tests.
    rng = np.random.default_rng(42)
    permutations = [rng.permutation(len(data)) for _ in range(200)]
    joint = pd.factorize(pd.MultiIndex.from_frame(data[flags]))[0]
    distributions, pairs, conditional = [], [], []
    for feature in features:
        x = data[feature].to_numpy()
        binned = np.digitize(x, [.1, .5, .9], right=True)
        entropy = mutual_info_score(binned, binned) / np.log(2)
        distributions.append({'feature': feature, 'n': len(x), 'mean': float(x.mean()),
            'std': float(x.std()), 'fraction_le_0_1': float((x <= .1).mean()),
            'fraction_ge_0_9': float((x >= .9).mean()),
            'binned_entropy_bits': float(entropy),
            **{f'q{q}': float(np.percentile(x, q)) for q in [0, 10, 25, 50, 75, 90, 100]}})
        for flag in flags + ['ALL_ITEM_FLAGS_JOINT']:
            y = joint if flag == 'ALL_ITEM_FLAGS_JOINT' else data[flag].to_numpy()
            mi = mutual_info_score(binned, y) / np.log(2)
            shuffled = [mutual_info_score(binned[p], y) / np.log(2) for p in permutations]
            corr = float(np.corrcoef(x, y)[0, 1]) if flag in flags and x.std() > 0 and y.std() > 0 else None
            pairs.append({'feature': feature, 'flag': flag, 'pearson_r': corr,
                'mi_bits': float(mi), 'shuffled_mean_mi_bits': float(np.mean(shuffled)),
                'shuffled_95th_mi_bits': float(np.quantile(shuffled, .95)),
                'fraction_binned_entropy_explained': float(mi / entropy) if entropy else None})
            if flag in flags:
                for value in [0, 1]:
                    subset = x[y == value]
                    conditional.append({'feature': feature, 'flag': flag, 'value': value,
                        'n': len(subset), 'mean_probability': float(subset.mean()) if len(subset) else None,
                        'fraction_ge_0_9': float((subset >= .9).mean()) if len(subset) else None})
    pd.DataFrame(distributions).to_csv(out / 'distributions.csv', index=False)
    pd.DataFrame(pairs).to_csv(out / 'associations.csv', index=False)
    pd.DataFrame(conditional).to_csv(out / 'conditional_distributions.csv', index=False)
    metadata = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'split': 'train', 'n': len(data), 'seed': 42, 'shuffle_references': 200,
        'notes': ['No outcomes read, no models refit, no inference calls.',
                  'MI uses four fixed probability bins and is descriptive, not a return-predictiveness measure.',
                  'Shuffling ignores issuer/date dependence; references are not calibrated hypothesis tests.',
                  'Pairwise association cannot establish redundancy; joint binned MI is also not a full conditional signal test.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Training filings:', len(data))
    for row in distributions:
        if not row['feature'].endswith('_disclosed'):
            continue
        matches = [p for p in pairs if p['feature'] == row['feature'] and p['pearson_r'] is not None]
        strongest = max(matches, key=lambda p: abs(p['pearson_r']))
        combined = next(p for p in pairs if p['feature'] == row['feature'] and p['flag'] == 'ALL_ITEM_FLAGS_JOINT')
        print(json.dumps({'distribution': row, 'strongest_correlation': strongest, 'joint_flags': combined}))


if __name__ == '__main__':
    main()
