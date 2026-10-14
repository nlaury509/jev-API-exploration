#!/usr/bin/env python3
"""Chronological, matched-sample predictive test of SEC text features."""
import argparse
import hashlib
import json
import math
import os
import re
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/jev-matplotlib')
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from collect import ROOT, NY, now, write_json
from jev import choice_probabilities, digest, payload_for, validate_response
from large_study import STUDY
from text_baseline import classify


def price_panels(bars):
    bars = bars.copy()
    bars['session'] = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    if bars.duplicated(['symbol', 'session']).any():
        raise ValueError('Multiple daily bars per symbol/session.')
    sessions = pd.DatetimeIndex(sorted(bars.loc[bars.symbol == 'SPY', 'session']))
    if sessions.empty:
        raise ValueError('SPY session reference is missing.')
    panels = {s: g.set_index('session').reindex(sessions) for s, g in bars.groupby('symbol')}
    return sessions, panels


def market_features(panel):
    """Features indexed by the LAST completed daily bar, never centered."""
    close = panel['close']
    r = close.pct_change(fill_method=None)
    f = pd.DataFrame(index=panel.index)
    for window in (1, 5, 20, 60):
        f[f'return_{window}'] = close / close.shift(window) - 1
    for window in (5, 20):
        f[f'volatility_{window}'] = r.rolling(window, min_periods=window).std()
    for window in (20, 60):
        f[f'sma_distance_{window}'] = close / close.rolling(window, min_periods=window).mean() - 1
    f['relative_volume_20'] = panel.volume / panel.volume.rolling(20, min_periods=20).mean()
    f['range'] = (panel.high - panel.low) / close
    f['open_gap'] = panel.open / close.shift(1) - 1
    f['close_vs_open'] = close / panel.open - 1
    return f


def event_timing(accepted, sessions, horizon, publication_lag_minutes=None,
                 processing_lag_minutes=0):
    accepted = pd.Timestamp(accepted)
    if accepted.tzinfo is None:
        raise ValueError('Filing acceptance timestamp must include a timezone.')
    if publication_lag_minutes is not None and publication_lag_minutes < 0:
        raise ValueError('Publication lag must be nonnegative.')
    if processing_lag_minutes < 0:
        raise ValueError('Processing lag must be nonnegative.')
    if publication_lag_minutes is None:
        if processing_lag_minutes:
            raise ValueError('Processing lag requires a publication lag.')
        local_day = accepted.tz_convert('America/New_York').tz_localize(None).normalize()
        entry = int(sessions.searchsorted(local_day, side='right'))
    else:
        signal_ready = accepted + pd.Timedelta(
            minutes=publication_lag_minutes + processing_lag_minutes)
        opens = sessions.tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)
        entry = int(opens.searchsorted(signal_ready, side='right'))
    exit_index = entry + horizon
    if entry < 61 or exit_index >= len(sessions):
        return None
    return entry, exit_index


def model_for(learner, cfg):
    if learner == 'boosting':
        params = {k: v for k, v in cfg['learner'].items() if k != 'name'}
        return HistGradientBoostingClassifier(**params)
    if learner == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=cfg['logistic_C'], max_iter=3000, random_state=cfg['seed']))
    raise ValueError(f'Unknown learner: {learner}')


def split_for(entry, exit_date, cfg):
    # Purge outcomes crossing a split boundary, including validation/test.
    for name, start, end in [('train', cfg['start'], cfg['train_end']),
                             ('validation', cfg['train_end'], cfg['validation_end']),
                             ('test', cfg['validation_end'], cfg['test_end'])]:
        if pd.Timestamp(start) <= entry < pd.Timestamp(end):
            return name if exit_date < pd.Timestamp(end) else None
    return None


def build_dataset(cfg, study=STUDY, publication_lag_minutes=15,
                 processing_lag_minutes=1):
    sessions, panels = price_panels(pd.read_csv(study / 'bars.csv'))
    features = {s: market_features(p) for s, p in panels.items()}
    events = json.loads((study / 'events.json').read_text())
    extraction = json.loads((ROOT / 'config/extraction.json').read_text())
    questions = extraction['questions']
    if digest(questions) != cfg['questions_sha256']:
        raise ValueError('Question configuration differs from the study specification.')
    rows, dropped = [], []
    for e in events:
        cid, symbol = e['accession'], e['symbol']
        reason = None
        if symbol not in panels:
            reason = 'no price series'
        timing = event_timing(e['accepted_at'], sessions, cfg['horizon_sessions'],
                              publication_lag_minutes, processing_lag_minutes)
        if timing is None:
            reason = 'insufficient calendar history or forward sessions'
        if reason:
            dropped.append({'accession': cid, 'reason': reason}); continue
        i, j = timing
        split = split_for(sessions[i], sessions[j], cfg)
        if split is None:
            dropped.append({'accession': cid, 'reason': 'label crosses split boundary or entry outside study'}); continue
        p = panels[symbol]
        # Require every return observation and holding-period open; do not carry
        # stale prices through missing sessions, halts, or IPO history.
        if p['close'].iloc[i-61:i].isna().any() or p['open'].iloc[i:j+1].isna().any():
            dropped.append({'accession': cid, 'reason': 'missing lookback or holding-period bars'}); continue
        text = (study / e['text_path']).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != e['text_sha256']:
            raise ValueError('Input text changed.')
        response_path = study / 'jev' / f'{cid}.json'
        if not response_path.exists():
            dropped.append({'accession': cid, 'reason': 'missing Jev inference; excluded from ALL comparisons'}); continue
        result = json.loads(response_path.read_text())
        if result['text_sha256'] != e['text_sha256'] or digest(result['response']) != result['response_sha256']:
            raise ValueError('Jev source/response integrity mismatch.')
        payload = payload_for(text, extraction)
        if result['request_sha256'] != digest(payload):
            raise ValueError('Jev request differs from the fixed question specification.')
        validate_response(result['response'], payload)
        asset = features[symbol].iloc[i-1]
        spy = features['SPY'].iloc[i-1]
        if not np.isfinite(asset.to_numpy()).all() or not np.isfinite(spy.to_numpy()).all():
            dropped.append({'accession': cid, 'reason': 'nonfinite market features'}); continue
        f = {'m_' + k: float(v) for k, v in asset.items()}
        f.update({'spy_feature_' + k: float(v) for k, v in spy.items()})
        f['m_excess_return_20'] = float(asset.return_20 - spy.return_20)
        items = set(re.findall(r'\d\.\d{2}', e['items']))
        for item in ['1.01','1.02','2.02','2.03','2.04','3.02','5.02','7.01','8.01']:
            f['meta_item_' + item] = int(item in items)
        f['meta_log_characters'] = math.log1p(len(text))
        accepted = pd.Timestamp(e['accepted_at']).tz_convert('America/New_York')
        entry_time = sessions[i].tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)
        signal_ready = accepted + pd.Timedelta(
            minutes=publication_lag_minutes + processing_lag_minutes)
        f['meta_hours_since_acceptance'] = (entry_time - accepted).total_seconds() / 3600
        for sector in sorted(set(cfg['sectors'].values())):
            f['meta_sector_' + sector] = int(cfg['sectors'][symbol] == sector)
        # Issuer controls shared by every learner/feature set.
        for issuer in cfg['symbols']:
            f['meta_issuer_' + issuer] = int(symbol == issuer)
        rules = classify(text)
        for q in questions:
            f['rule_' + q] = int(rules[q]['choice'] == 'disclosed')
            probabilities = choice_probabilities(result['response']['answers'][q])
            for label in ('disclosed', 'unclear'):
                f['jev_' + q + '_' + label] = probabilities[label]
        asset_return = float(p.open.iloc[j] / p.open.iloc[i] - 1)
        spy_return = float(panels['SPY'].open.iloc[j] / panels['SPY'].open.iloc[i] - 1)
        rows.append({'accession': cid, 'symbol': symbol, 'accepted_at': e['accepted_at'],
                     'signal_ready_at': signal_ready.isoformat(),
                     'publication_lag_minutes': publication_lag_minutes,
                     'processing_lag_minutes': processing_lag_minutes,
                     'entry_time': entry_time.isoformat(),
                     'entry_date': sessions[i], 'exit_date': sessions[j], 'feature_date': sessions[i-1],
                     'split': split, 'asset_return': asset_return, 'spy_return': spy_return,
                     'excess_return': asset_return-spy_return, 'target': int(asset_return > spy_return), **f})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No usable matched events.')
    frame = frame.sort_values(['entry_date', 'symbol', 'accepted_at', 'accession'])
    duplicate = frame.duplicated(['symbol', 'entry_date'], keep='last')
    dropped.extend({'accession': cid, 'reason': 'same issuer/entry date; kept latest available filing'} for cid in frame.loc[duplicate, 'accession'])
    return frame.loc[~duplicate].reset_index(drop=True), dropped, sessions, panels


def scores(y, p):
    return {'n': len(y), 'positive_fraction': float(np.mean(y)), 'log_loss': float(log_loss(y, p, labels=[0,1])),
            'brier': float(brier_score_loss(y,p)), 'accuracy': float(accuracy_score(y,p>=.5)),
            'auc': float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}


def feature_columns(frame):
    base=[c for c in frame if c.startswith(('m_','spy_feature_','meta_'))]
    sets={'market_and_metadata':base,'market_metadata_rules':base+[c for c in frame if c.startswith('rule_')],
          'market_metadata_jev':base+[c for c in frame if c.startswith('jev_')]}
    forbidden={'target','asset_return','spy_return','excess_return','entry_date','exit_date','split'}
    if any(forbidden.intersection(columns) for columns in sets.values()):
        raise ValueError('Outcome or split metadata entered model features.')
    return sets


def paired_block_interval(frame, base, extended, sessions, cfg):
    y = frame.target.to_numpy()
    def loss(p):
        p=np.clip(p,1e-15,1-1e-15)
        return -(y*np.log(p)+(1-y)*np.log(1-p))
    delta = loss(extended)-loss(base)
    daily = pd.DataFrame({'date': frame.entry_date.to_numpy(), 'delta':delta}).groupby('date').delta.agg(['sum','count']).reindex(sessions, fill_value=0)
    sums, counts = daily['sum'].to_numpy(), daily['count'].to_numpy()
    rng = np.random.default_rng(cfg['seed'])
    length = cfg['bootstrap_block_sessions']
    n = len(sessions)
    estimates = []
    for _ in range(cfg['bootstrap_replicates']):
        starts = rng.integers(0,n,size=math.ceil(n/length))
        indices = ((starts[:,None]+np.arange(length))%n).ravel()[:n]
        if counts[indices].sum():
            estimates.append(sums[indices].sum()/counts[indices].sum())
    return {'delta_log_loss':float(delta.mean()), 'approximate_95pct_date_block_interval':np.quantile(estimates,[.025,.975]).tolist(),
            'block_sessions':length,'replicates':len(estimates), 'negative_favors_added_features':True}


def simulate(frame, probabilities, threshold, sessions, panels, risk):
    """Fixed-size asset-long/SPY-short diagnostic; adjusted-open price proxies."""
    signals = frame.assign(probability=probabilities)
    grouped = {pd.Timestamp(k):g for k,g in signals.groupby('entry_date')}
    equity, peak, fees_total, borrow_total = 1., 1., 0., 0.
    active, trades, curve = {}, [], []
    cost = risk['one_way_cost_bps']/10000
    borrow_rate = risk['annual_short_borrow_bps']/10000/252
    turnover = 0.
    for date in sessions:
        spy_open = float(panels['SPY'].loc[date,'open'])
        for symbol, pos in list(active.items()):
            asset_open = float(panels[symbol].loc[date,'open'])
            gain = pos['asset_shares']*(asset_open-pos['last_asset']) + pos['spy_shares']*(spy_open-pos['last_spy'])
            borrow = abs(pos['spy_shares']*spy_open)*borrow_rate
            equity += gain-borrow
            pos['pnl'] += gain-borrow
            borrow_total += borrow
            pos['last_asset'], pos['last_spy'] = asset_open, spy_open
            if date >= pos['exit_date']:
                notional = abs(pos['asset_shares']*asset_open)+abs(pos['spy_shares']*spy_open)
                fee = cost*notional
                turnover += notional/max(equity,1e-12)
                fees_total += fee; equity -= fee; pos['pnl'] -= fee
                trades.append({k:v for k,v in pos.items() if k not in ('last_asset','last_spy')})
                del active[symbol]
        peak = max(peak,equity)
        if equity > 0 and equity/peak-1 > -risk['max_drawdown_new_entries']:
            today = grouped.get(date)
            if today is not None:
                for _,row in today.sort_values(['probability','accession'],ascending=[False,True]).iterrows():
                    symbol = row.symbol
                    if row.probability < threshold or symbol in active:
                        continue
                    gross = sum(abs(p['asset_shares']*float(panels[s].loc[date,'open']))+abs(p['spy_shares']*spy_open) for s,p in active.items())
                    a = equity*risk['asset_weight']; b = equity*risk['benchmark_hedge_weight']
                    fee = cost*(a+b)
                    if gross+a+b > (equity-fee)*risk['max_gross_exposure']:
                        continue
                    turnover += (a+b)/equity
                    equity -= fee; fees_total += fee
                    asset_open = float(panels[symbol].loc[date,'open'])
                    active[symbol] = {'symbol':symbol,'accession':row.accession,'entry_date':date,'exit_date':row.exit_date,
                         'probability':row.probability,'asset_shares':a/asset_open,'spy_shares':-b/spy_open,
                         'last_asset':asset_open,'last_spy':spy_open,'pnl':-fee}
        peak = max(peak,equity)
        curve.append({'date':date,'equity':equity,'drawdown':equity/peak-1,'active_positions':len(active),
                      'gross_same_positions_equity':equity+fees_total+borrow_total})
    if active:
        raise ValueError('Portfolio ended with unresolved positions.')
    curve = pd.DataFrame(curve)
    daily = curve.equity.pct_change().fillna(curve.equity.iloc[0]-1)
    stats = {'threshold_from_validation':float(threshold), 'trades':len(trades), 'net_return':float(equity-1),
             'gross_return_same_positions':float(equity+fees_total+borrow_total-1),
             'max_drawdown':float(curve.drawdown.min()), 'transaction_costs_initial_equity':fees_total,
             'borrow_costs_initial_equity':borrow_total,'turnover':turnover,
             'annualized_daily_sharpe_zero_cash_rate':float(daily.mean()/daily.std()*np.sqrt(252)) if daily.std()>0 else None,
             'trade_win_rate':float(np.mean([t['pnl']>0 for t in trades])) if trades else None,
             'caveat':'Adjusted daily-open proxies, estimated costs/borrow, no locate or capacity model; not executable performance.'}
    return stats, curve, trades


def evaluate_lag_scenario(cfg, frame, sessions):
    split_counts = frame.groupby('split').size().to_dict()
    for split, minimum in [('train', 150), ('validation', 40), ('test', 80)]:
        if split_counts.get(split, 0) < minimum:
            raise ValueError(f'Insufficient {split} events for lag scenario: {split_counts}')
    train = frame.loc[frame.split == 'train']
    val = frame.loc[frame.split == 'validation']
    test = frame.loc[frame.split == 'test']
    sets = feature_columns(frame)
    test_sessions = sessions[(sessions >= pd.Timestamp(cfg['validation_end'])) &
                             (sessions < pd.Timestamp(cfg['test_end']))]
    probabilities, metrics = {}, {}
    for learner in ('boosting', 'logistic'):
        for name, columns in sets.items():
            key = learner + '__' + name
            model = model_for(learner, cfg)
            model.fit(train[columns], train.target)
            vp = model.predict_proba(val[columns])[:, 1]
            tp = model.predict_proba(test[columns])[:, 1]
            probabilities[key] = tp
            metrics[key] = {'validation': scores(val.target, vp), 'test': scores(test.target, tp)}
    paired = paired_block_interval(
        test, probabilities['boosting__market_and_metadata'],
        probabilities['boosting__market_metadata_jev'], test_sessions, cfg)
    return {'split_counts': {str(k): int(v) for k, v in split_counts.items()},
            'metrics': metrics, 'primary_comparison': paired}


def evaluate_lag_sensitivity(cfg, processing_lag_minutes=1, study=STUDY, output=None):
    if processing_lag_minutes < 0:
        raise ValueError('Processing lag must be nonnegative.')
    inference = json.loads((study / 'inference_summary.json').read_text())
    if not inference['complete']:
        raise ValueError('Finish extraction before evaluating any outcomes.')
    if json.loads((study / 'config.json').read_text()) != cfg:
        raise ValueError('Configuration differs from frozen collection configuration.')
    out = output or ROOT / 'results' / 'large_study_v1_lag_sensitivity'
    if out.exists():
        raise ValueError(f'Lag-sensitivity results already exist: {out}')
    scenarios = [
        ('next_session_after_acceptance_local_date', None, 0),
        *((f'publication_{lag}m_processing_{processing_lag_minutes}m', lag,
           processing_lag_minutes) for lag in (0, 15, 30, 60)),
    ]
    results, flat_metrics = [], []
    for name, publication_lag, processing_lag in scenarios:
        frame, dropped, sessions, _ = build_dataset(
            cfg, study, publication_lag, processing_lag)
        result = evaluate_lag_scenario(cfg, frame, sessions)
        result |= {'name': name, 'publication_lag_minutes': publication_lag,
                   'processing_lag_minutes': processing_lag,
                   'usable_events': len(frame),
                   'excluded_counts': dict(Counter(r['reason'] for r in dropped))}
        results.append(result)
        for model, values in result['metrics'].items():
            flat_metrics.append({'scenario': name, 'model': model,
                                 'validation_log_loss': values['validation']['log_loss'],
                                 'test_log_loss': values['test']['log_loss'],
                                 'test_brier': values['test']['brier'],
                                 'test_auc': values['test']['auc'],
                                 'test_accuracy': values['test']['accuracy']})
        print(name, 'usable events', len(frame),
              'primary test log-loss difference',
              round(result['primary_comparison']['delta_log_loss'], 5), flush=True)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', cfg)
    write_json(out / 'lag_sensitivity.json', {
        'created_at': now(),
        'processing_lag_minutes_for_publication_scenarios': processing_lag_minutes,
        'entry_rule': 'Enter at the first daily session open strictly after signal_ready_at.',
        'signal_ready_rule': 'SEC acceptance time + publication lag + processing lag.',
        'scenarios': results,
        'limitations': [
            'Publication lags are assumptions because SEC acceptance is not an exact public-availability timestamp.',
            'The processing lag is illustrative; replace it with measured end-to-end inference latency.',
            'Historical inference responses do not prove that the same signal would have been produced or available in real time.',
            'Daily open prices are proxies, not executable fills.',
        ],
    })
    pd.DataFrame(flat_metrics).to_csv(out / 'scenario_metrics.csv', index=False)
    print('Saved lag sensitivity:', out, flush=True)


def evaluate(cfg, publication_lag_minutes=15, processing_lag_minutes=1):
    if publication_lag_minutes < 0 or processing_lag_minutes < 0:
        raise ValueError('Publication and processing lags must be nonnegative.')
    inference = json.loads((STUDY/'inference_summary.json').read_text())
    if not inference['complete']:
        raise ValueError('Finish extraction before evaluating any outcomes.')
    if json.loads((STUDY/'config.json').read_text()) != cfg:
        raise ValueError('Configuration differs from frozen collection configuration.')
    out = ROOT/'results'/f'large_study_v1_publication_{publication_lag_minutes}m_processing_{processing_lag_minutes}m'
    if (out/'summary.json').exists():
        raise ValueError('This test was already evaluated. Inspect the saved result; do not silently retune/retest.')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg)
    timing_assumptions = {
        'publication_lag_minutes': publication_lag_minutes,
        'processing_lag_minutes': processing_lag_minutes,
        'entry_rule': 'First daily session open strictly after signal_ready_at.',
        'signal_ready_rule': 'SEC acceptance time + publication lag + processing lag.',
    }
    write_json(out/'timing_assumptions.json', timing_assumptions)
    (out/'evaluation_code.py').write_bytes(Path(__file__).read_bytes())
    import sklearn
    write_json(out/'software.json',{'evaluated_at':now(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,
        'supporting_code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
          for name in ['collect.py','large_study.py','jev.py','text_baseline.py']}})
    frame,dropped,sessions,panels = build_dataset(
        cfg, publication_lag_minutes=publication_lag_minutes,
        processing_lag_minutes=processing_lag_minutes)
    frame.to_csv(out/'events_with_features_and_labels.csv',index=False)
    write_json(out/'excluded_events.json',dropped)
    split_counts = frame.groupby('split').size().to_dict()
    for split,minimum in [('train',150),('validation',40),('test',80)]:
        if split_counts.get(split,0)<minimum:
            raise ValueError(f'Insufficient {split} events for this experiment: {split_counts}')
    print('Matched event counts:',split_counts,flush=True)
    train=frame.loc[frame.split=='train']; val=frame.loc[frame.split=='validation']; test=frame.loc[frame.split=='test']
    sets=feature_columns(frame)
    write_json(out/'feature_columns.json',sets)
    test_sessions = sessions[(sessions>=pd.Timestamp(cfg['validation_end'])) & (sessions<pd.Timestamp(cfg['test_end']))]
    probabilities,metrics,portfolios,curves={}, {}, {}, {}
    predictions=test[['accession','symbol','entry_date','exit_date','target','asset_return','spy_return','excess_return']].copy()
    validation_predictions=val[['accession','target','entry_date','excess_return']].copy()
    for learner in ('boosting','logistic'):
        for name,columns in sets.items():
            key=learner+'__'+name
            model=model_for(learner,cfg)
            model.fit(train[columns],train.target)
            vp=model.predict_proba(val[columns])[:,1]
            tp=model.predict_proba(test[columns])[:,1]
            probabilities[key]=tp; predictions[key]=tp; validation_predictions[key]=vp
            metrics[key]={'validation':scores(val.target,vp),'test':scores(test.target,tp),
                'test_by_year':{str(year):scores(g.target,tp[test.entry_date.dt.year.to_numpy()==year]) for year,g in test.groupby(test.entry_date.dt.year)}}
            threshold=float(np.quantile(vp,.8))
            stats,curve,trades=simulate(test,tp,threshold,test_sessions,panels,cfg['portfolio'])
            portfolios[key]=stats; curves[key]=curve
            curve.to_csv(out/f'{key}_equity.csv',index=False)
            pd.DataFrame(trades).to_csv(out/f'{key}_trades.csv',index=False)
            print(key,'test log loss',round(metrics[key]['test']['log_loss'],5),'net proxy return',round(stats['net_return'],5),flush=True)
    constant=np.full(len(test),float(train.target.mean()))
    metrics['train_class_prior']={'test':scores(test.target,constant)}
    metrics['uniform']={'test':scores(test.target,np.full(len(test),.5))}
    paired={}
    for learner in ('boosting','logistic'):
        for extra in ('rules','jev'):
            paired[learner+'_'+extra]=paired_block_interval(test, probabilities[learner+'__market_and_metadata'],
                probabilities[learner+'__market_metadata_'+extra],test_sessions,cfg)
    predictions.to_csv(out/'test_predictions.csv',index=False)
    validation_predictions.to_csv(out/'validation_predictions.csv',index=False)
    confidence=[]
    bins=[0,.4,.45,.5,.55,.6,.7,1.000001]
    for name,p in probabilities.items():
        for low,high in zip(bins[:-1],bins[1:]):
            take=(p>=low)&(p<high)
            if take.any():
                confidence.append({'model':name,'lower':low,'upper':min(high,1),'n':int(take.sum()),
                    'mean_probability':float(p[take].mean()),'positive_frequency':float(test.target.to_numpy()[take].mean()),
                    'mean_excess_return_bps':float(test.excess_return.to_numpy()[take].mean()*10000)})
    write_json(out/'confidence_analysis.json',confidence)
    summary={'created_at':now(),'collection':json.loads((STUDY/'collection_summary.json').read_text()),'inference':inference,
        'timing_assumptions':timing_assumptions,
        'split_counts':split_counts,'unique_symbols':int(frame.symbol.nunique()),'test_symbols':int(test.symbol.nunique()),
        'excluded_counts':dict(Counter(r['reason'] for r in dropped)), 'metrics':metrics,'paired_comparisons':paired,'portfolios':portfolios,
        'usable_events_by_year':{str(k):int(v) for k,v in frame.groupby(frame.entry_date.dt.year).size().items()},
        'usable_events_by_symbol':{str(k):int(v) for k,v in frame.groupby('symbol').size().items()},
        'jev_feature_variation':{c:{'min':float(frame[c].min()),'max':float(frame[c].max()),'std':float(frame[c].std())} for c in frame if c.startswith('jev_')},
        'primary_result':paired['boosting_jev'],'limitations':cfg['limitations']+[
            'Publication lag is an assumption because SEC acceptance is not an exact public-availability timestamp.',
            'Processing lag is an assumption; replace it with measured end-to-end inference latency.',
            'Historical inference responses do not prove that the same signal would have been produced or available in real time.',
            'Four text features do not test the value of all possible language-model inputs.',
            'Issuer/date dependence is approximated by 20-session blocks; intervals are exploratory, not proof.',
            'No tuning after viewing test results; secondary learner/rules/year analyses are descriptive.',
            'Portfolio equity is a costed hedged diagnostic with approximate adjusted daily opens, no execution validation.']}
    write_json(out/'summary.json',summary)
    plots(out,curves,metrics,confidence)
    report(out,summary)
    print('Completed predictive study:',out,flush=True)


def plots(out,curves,metrics,confidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    colors={'market_and_metadata':'#64748b','market_metadata_rules':'#d97706','market_metadata_jev':'#2563eb'}
    for name,color in colors.items():
        key='boosting__'+name; c=curves[key]
        label=name.replace('market_metadata_','+ ').replace('market_and_metadata','Market + metadata')
        axes[0,0].plot(c.date,c.equity,label=label,color=color)
        axes[0,1].plot(c.date,100*c.drawdown,label=label,color=color)
        points=[r for r in confidence if r['model']==key]
        axes[1,0].plot([r['mean_probability'] for r in points],[r['positive_frequency'] for r in points],'-o',label=label,color=color)
        axes[1,1].plot([r['mean_probability'] for r in points],[r['mean_excess_return_bps'] for r in points],'-o',label=label,color=color)
    axes[1,0].plot([0,1],[0,1],'k--',alpha=.4)
    axes[1,1].axhline(0,color='black',lw=.7)
    for ax,title in zip(axes.ravel(),['Costed hedged diagnostic equity','Drawdown (%)','Test probability calibration','Confidence vs five-session excess return (bps)']):
        ax.set_title(title); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[1,0].set_xlabel('Mean predicted probability'); axes[1,0].set_ylabel('Observed positive frequency')
    axes[1,1].set_xlabel('Mean predicted probability')
    fig.suptitle('Exploratory historical test — adjusted daily-open proxies, not executable performance')
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)


def report(out,s):
    p=s['primary_result']; lo,hi=p['approximate_95pct_date_block_interval']
    conclusion=('Primary comparison favors the Jev features in this historical sample; the approximate interval is below zero.' if hi<0 else
                'Primary comparison does not establish an improvement from Jev features; the interval includes zero.' if lo<=0 else
                'Primary comparison favors the baseline: adding Jev features increases log loss in this historical sample.')
    lines=['# Large historical predictive study','',conclusion,'',
       f"Matched events: {s['split_counts']}; {s['unique_symbols']} issuers.",'',
       f"Timing: publication lag {s['timing_assumptions']['publication_lag_minutes']} minutes plus processing lag {s['timing_assumptions']['processing_lag_minutes']} minutes; enter at the first daily open strictly after the signal is ready.",
       'Target: positive asset-minus-SPY adjusted return from entry open through five trading sessions later.',
       'Training: 2018–2022; validation: 2023; test: 2024–2025. Labels crossing split boundaries were purged.','',
       '| Model | Test log loss ↓ | Brier ↓ | AUC ↑ | Accuracy |','|---|---:|---:|---:|---:|']
    for key,value in s['metrics'].items():
        m=value['test']; lines.append(f"| {key} | {m['log_loss']:.5f} | {m['brier']:.5f} | {m['auc']:.3f} | {m['accuracy']:.1%} |")
    lines += ['',f"Primary paired log-loss difference (Jev − baseline): **{p['delta_log_loss']:+.5f}**, approximate 95% interval **[{lo:+.5f}, {hi:+.5f}]** using 20-session date blocks.",'',
        '## Costed portfolio diagnostic','',
        'Each selected event holds 5% equity long the stock and 5% short SPY for five sessions, with no overlapping positions in the same issuer and a 100% gross entry limit. Entry requires probability at or above that model’s 2023 validation 80th percentile. Costs: 5 bps per leg per side plus 100 bps/year short borrowing. A 20% drawdown stops new entries while the drawdown persists.','',
        '| Model | Trades | Net return | Max drawdown |','|---|---:|---:|---:|']
    for key,m in s['portfolios'].items():
        lines.append(f"| {key} | {m['trades']} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} |")
    lines += ['','These are adjusted daily-open proxies, not fillable quotes; no capacity or locate model is included.','',
              '## Limitations','']+['- '+v for v in s['limitations']]
    lines += ['', '## Saved evidence','',
      'config.json and evaluation_code.py preserve the specification and code; feature_columns.json defines model inputs; test_predictions.csv preserves every prediction; excluded_events.json records every dropped case. The raw sources and inference responses remain in data/large_study_v1.','',
      f"Estimated total inference cost: ${s['inference']['estimated_total_usd']:.4f}; {s['inference']['completed']} documents. This is usage-based estimation, not a billing receipt.", '', '![Model comparisons](comparison.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication-lag-minutes', type=int, default=15,
                        help='Assumed SEC-publication delay for the main evaluation (default: 15).')
    parser.add_argument('--processing-lag-minutes', type=int, default=1,
                        help='Assumed signal-processing delay (default: 1; replace with measured latency).')
    parser.add_argument('--lag-sensitivity', action='store_true',
                        help='Evaluate the next-session baseline and 0/15/30/60-minute publication delays.')
    args = parser.parse_args()
    config = json.loads((ROOT/'config/large_study.json').read_text())
    if args.lag_sensitivity:
        evaluate_lag_sensitivity(config, args.processing_lag_minutes)
    else:
        evaluate(config, args.publication_lag_minutes, args.processing_lag_minutes)
