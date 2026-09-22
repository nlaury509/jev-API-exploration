import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jev
from compare_extraction import label_metrics


class JevTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((jev.ROOT / 'config/extraction.json').read_text())
        self.payload = jev.payload_for('A public filing.', self.config)
        self.response = {'model': self.config['model'], 'usage': {'input_tokens': 100, 'output_tokens': 20},
                         'answers': {q: {'type': 'choice', 'choice': 'not_disclosed', 'confidence': .8,
                              'probabilities': {'disclosed': .1, 'not_disclosed': .8, 'unclear': .1}}
                              for q in self.config['questions']}}

    def test_invalid_responses_rejected(self):
        self.assertIs(jev.validate_response(self.response, self.payload), self.response)
        bad = []
        for field, value in [('model', 'other-model'), ('usage', None), ('answers', {})]:
            r = copy.deepcopy(self.response)
            r[field] = value
            bad.append(r)
        for field, value in [('probabilities', None), ('choice', 'disclosed'), ('confidence', float('nan'))]:
            r = copy.deepcopy(self.response)
            r['answers']['debt_refinancing'][field] = value
            bad.append(r)
        for value in [float('nan'), -1, True, .9]:
            r = copy.deepcopy(self.response)
            r['answers']['debt_refinancing']['probabilities']['disclosed'] = value
            bad.append(r)
        for r in bad:
            with self.subTest(response=r), self.assertRaises(ValueError):
                jev.validate_response(r, self.payload)

    def test_payload_has_no_outcomes_and_never_truncates(self):
        self.assertEqual(set(self.payload), {'model', 'state', 'questions'})
        self.assertEqual(self.payload['state'], 'A public filing.')
        with self.assertRaises(ValueError):
            jev.payload_for('x' * (self.config['max_state_characters'] + 1), self.config)

    def test_observed_api_rounding_is_preserved_and_normalized_only_for_use(self):
        answer = self.response['answers']['debt_refinancing']
        answer['probabilities'] = {'disclosed': .81, 'not_disclosed': .17, 'unclear': .01}
        answer['choice'] = 'disclosed'
        original = copy.deepcopy(self.response)
        jev.validate_response(self.response, self.payload)
        self.assertEqual(self.response, original)
        normalized = jev.choice_probabilities(answer)
        self.assertAlmostEqual(sum(normalized.values()), 1.)
        self.assertAlmostEqual(normalized['disclosed'], .81 / .99)
        answer['probabilities']['disclosed'] = .79
        with self.assertRaises(ValueError):
            jev.validate_response(self.response, self.payload)

    def test_cache_avoids_paid_replay_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jev, 'ROOT', Path(tmp)), patch.object(jev, 'required', return_value='test-key'):
            client = jev.JevClient()
            with patch.object(client, '_request', return_value=(self.response, .2)) as request:
                first = client.predict(self.payload)
                second = client.predict(self.payload)
                self.assertFalse(first['cache_hit'])
                self.assertTrue(second['cache_hit'])
                request.assert_called_once()
                cache = next((Path(tmp) / 'data/jev_cache').glob('*.json'))
                record = json.loads(cache.read_text())
                record['response']['usage']['input_tokens'] = 999
                cache.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    client.predict(self.payload)

    def test_evaluation_scores_reference_not_model_confidence(self):
        labels = {q: {'label': 'disclosed'} for q in self.config['questions']}
        record = {'case_id': 'test', 'text_sha256': 'abc', 'response': self.response}
        gold = {'test': {'text_sha256': 'abc', 'labels': labels}}
        score = jev.evaluate([record], gold, list(self.config['questions']))
        self.assertEqual(score['accuracy'], 0)
        self.assertEqual(len(score['disagreements']), 4)
        self.assertAlmostEqual(score['brier_sum_over_classes'], 1.46)
        gold['test']['text_sha256'] = 'changed-source'
        with self.assertRaises(ValueError):
            jev.evaluate([record], gold, list(self.config['questions']))

    def test_previous_cases_excluded_before_limit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jev, 'ROOT', Path(tmp).resolve()):
            dataset = Path(tmp) / 'dataset'
            dataset.mkdir()
            (dataset / 'text.txt').write_text('A public filing.')
            events = [{'accession': cid, 'symbol': 'TEST', 'documents': [{'type': '8-K', 'text_path': 'text.txt'}],
                       'source': {'url': 'https://example.invalid/filing'}} for cid in ['old', 'fresh']]
            (dataset / 'events.json').write_text(json.dumps(events))
            config = self.config | {'limit': 1, 'exclude_accessions': ['old']}
            cases, _ = jev.prepare(dataset, config)
            self.assertEqual([c['case_id'] for c in cases], ['fresh'])
            with self.assertRaisesRegex(ValueError, 'prewritten reference'):
                jev.run_audit(dataset, config | {'require_reference': True}, None, True)

    def test_class_metrics_expose_missed_positives_and_missing_classes(self):
        rows = [{'case_id': 'one', 'reference': 'disclosed', 'model': 'not_disclosed'},
                {'case_id': 'one', 'reference': 'not_disclosed', 'model': 'not_disclosed'},
                {'case_id': 'two', 'reference': 'not_disclosed', 'model': 'not_disclosed'}]
        m = label_metrics(rows, 'model')
        self.assertAlmostEqual(m['accuracy'], 2 / 3)
        self.assertEqual(m['per_class']['disclosed']['recall'], 0)
        self.assertIsNone(m['per_class']['disclosed']['precision'])
        self.assertIsNone(m['per_class']['unclear']['recall'])
        self.assertEqual(m['documents_all_correct'], 1)


if __name__ == '__main__':
    unittest.main()
