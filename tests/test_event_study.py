import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from event_study import build_dataset, event_timing, feature_columns, market_features, paired_block_interval, simulate, split_for
from jev import ROOT, digest, payload_for


class EventStudyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'config/large_study.json').read_text())

    def test_next_session_is_strictly_after_acceptance_local_day(self):
        sessions=pd.bdate_range('2024-01-01',periods=150)
        # Friday after close, including UTC Saturday, maps to Monday.
        stamp='2024-04-06T00:15:00Z'
        i,j=event_timing(stamp,sessions,5)
        self.assertEqual(sessions[i],pd.Timestamp('2024-04-08'))
        self.assertEqual(j-i,5)
        # Before-open filings also wait until the next full session by design.
        i,_=event_timing('2024-04-08T10:00:00Z',sessions,5)
        self.assertEqual(sessions[i],pd.Timestamp('2024-04-09'))

    def test_purge_outcomes_across_all_split_boundaries(self):
        self.assertIsNone(split_for(pd.Timestamp('2022-12-29'),pd.Timestamp('2023-01-05'),self.cfg))
        self.assertIsNone(split_for(pd.Timestamp('2023-12-28'),pd.Timestamp('2024-01-04'),self.cfg))
        self.assertEqual(split_for(pd.Timestamp('2024-01-05'),pd.Timestamp('2024-01-12'),self.cfg),'test')

    def test_future_prices_cannot_change_market_features(self):
        n=100
        panel=pd.DataFrame({'close':np.arange(n)+100.,'open':np.arange(n)+99.,
                            'high':np.arange(n)+102.,'low':np.arange(n)+98.,'volume':np.arange(n)+1000.})
        first=market_features(panel).iloc[70].copy()
        panel.loc[71:,['close','open','high','low','volume']]*=10
        pd.testing.assert_series_equal(first,market_features(panel).iloc[70])

    def test_end_to_end_rows_lag_prices_and_future_changes_only_labels(self):
        sessions=pd.bdate_range('2020-01-01',periods=100)
        records=[]
        for symbol in ['SPY','AAPL']:
            for i,date in enumerate(sessions):
                o=100+i*(1. if symbol=='AAPL' else .5)
                records.append({'symbol':symbol,'timestamp':date.tz_localize('America/New_York').isoformat(),
                                'open':o,'high':o+2,'low':o-2,'close':o+1,'volume':1000+i})
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp);(study/'jev').mkdir()
            bars=pd.DataFrame(records);bars.to_csv(study/'bars.csv',index=False)
            (study/'primary.txt').write_text('A generic public filing.')
            h=hashlib.sha256((study/'primary.txt').read_bytes()).hexdigest()
            acceptance=sessions[70].tz_localize('America/New_York')+pd.Timedelta(hours=17)
            event={'accession':'fixture','symbol':'AAPL','accepted_at':acceptance.isoformat(),'items':'1.01',
                   'text_path':'primary.txt','text_sha256':h}
            (study/'events.json').write_text(json.dumps([event]))
            extraction=json.loads((ROOT/'config/extraction.json').read_text())
            response={'model':extraction['model'],'usage':{'input_tokens':100,'output_tokens':20},
                      'answers':{q:{'type':'choice','choice':'not_disclosed','confidence':.7,
                      'probabilities':{'disclosed':.1,'unclear':.2,'not_disclosed':.7}} for q in extraction['questions']}}
            payload=payload_for((study/'primary.txt').read_text(),extraction)
            (study/'jev/fixture.json').write_text(json.dumps({'response':response,'text_sha256':h,
                 'response_sha256':digest(response),'request_sha256':digest(payload)}))
            before,_,_,_=build_dataset(self.cfg,study)
            self.assertEqual(before.iloc[0].feature_date,sessions[70])
            self.assertEqual(before.iloc[0].entry_date,sessions[71])
            # Huge change in the eventual exit price must not affect any input.
            bars.loc[(bars.symbol=='AAPL')&(bars.index%100==76),'open']*=1.5
            bars.loc[(bars.symbol=='SPY')&(bars.index%100==76),'open']*=1.25
            bars.to_csv(study/'bars.csv',index=False)
            after,_,_,_=build_dataset(self.cfg,study)
            sets=feature_columns(before)
            features=sorted(set(c for columns in sets.values() for c in columns))
            self.assertTrue({'asset_return','spy_return','excess_return','target','entry_date','exit_date'}.isdisjoint(features))
            pd.testing.assert_frame_equal(before[features],after[features])
            self.assertNotEqual(before.iloc[0].excess_return,after.iloc[0].excess_return)
            self.assertNotEqual(before.iloc[0].spy_return,after.iloc[0].spy_return)

    def test_hedged_flat_prices_lose_exact_costs_and_disallow_overlap(self):
        dates=pd.bdate_range('2024-01-02',periods=7)
        panels={s:pd.DataFrame({'open':100.},index=dates) for s in ['SPY','AAPL']}
        frame=pd.DataFrame([{'accession':'one','symbol':'AAPL','entry_date':dates[0],'exit_date':dates[5]},
                            {'accession':'two','symbol':'AAPL','entry_date':dates[1],'exit_date':dates[6]}])
        risk=self.cfg['portfolio']|{'annual_short_borrow_bps':0}
        stats,curve,trades=simulate(frame,np.array([.9,.9]),.8,dates,panels,risk)
        self.assertEqual(len(trades),1)
        self.assertAlmostEqual(stats['net_return'],-.0001)
        self.assertAlmostEqual(stats['gross_return_same_positions'],0)
        self.assertTrue((curve.equity<=1).all())

    def test_identical_predictions_have_zero_paired_difference(self):
        dates=pd.bdate_range('2024-01-02',periods=30)
        frame=pd.DataFrame({'entry_date':dates,'target':np.arange(30)%2})
        p=np.linspace(.3,.7,30)
        result=paired_block_interval(frame,p,p,dates,self.cfg|{'bootstrap_replicates':20})
        self.assertEqual(result['delta_log_loss'],0)
        self.assertEqual(result['approximate_95pct_date_block_interval'],[0,0])


if __name__=='__main__':
    unittest.main()
