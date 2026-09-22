import unittest

import pandas as pd

from feature_study import answer_features, configuration, nested_columns
from jev import payload_for, validate_response


class FiftyQuestionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()
        self.answers = {}
        for q, question in self.cfg['questions'].items():
            labels = list(question['criteria'])
            self.answers[q] = {'type': 'choice', 'choice': labels[-1], 'confidence': 1.,
                               'probabilities': {label: float(label == labels[-1]) for label in labels}}

    def test_complete_fifty_question_contract_and_missing_answer_rejected(self):
        payload = payload_for('Evidence only.', self.cfg)
        response = {'model': self.cfg['model'], 'answers': self.answers,
                    'usage': {'input_tokens': 100, 'output_tokens': 100}}
        validate_response(response, payload)
        del response['answers'][self.cfg['question_order'][-1]]
        with self.assertRaises(ValueError):
            validate_response(response, payload)

    def test_repeated_original_answers_do_not_change_added_features(self):
        before = answer_features(self.answers, self.cfg)
        for q in self.cfg['question_order'][:4]:
            self.answers[q]['probabilities'] = {'disclosed': 1., 'not_disclosed': 0., 'unclear': 0.}
        self.assertEqual(before, answer_features(self.answers, self.cfg))

    def test_probability_only_mode_preserves_but_does_not_trust_selected_label(self):
        q = self.cfg['question_order'][0]
        self.answers[q] = {'type': 'choice', 'choice': 'unclear', 'confidence': .1,
                          'probabilities': {'disclosed': .48, 'unclear': .47, 'not_disclosed': .05}}
        response = {'model': self.cfg['model'], 'answers': self.answers,
                    'usage': {'input_tokens': 100, 'output_tokens': 100}}
        payload = payload_for('Evidence only.', self.cfg)
        with self.assertRaises(ValueError):
            validate_response(response, payload)
        validate_response(response, payload, require_argmax=False)
        self.assertEqual(response['answers'][q]['choice'], 'unclear')
        self.answers[q]['choice'] = 'invalid_label'
        with self.assertRaises(ValueError):
            validate_response(response, payload, require_argmax=False)

    def test_nested_bundles_exclude_labels_and_preserve_all_original_inputs(self):
        row = {'m_return': .01, 'meta_item_1.01': 1, 'spy_feature_return': .02,
               'target': 1, 'asset_return': .1, 'spy_return': .05, 'excess_return': .05,
               'split': 'train', 'entry_date': '2020-01-01'}
        row.update({f'jev_{q}_{label}': .1 for q in self.cfg['question_order'][:4]
                    for label in ['disclosed', 'unclear']})
        row.update(answer_features(self.answers, self.cfg))
        sets = nested_columns(pd.DataFrame([row]), self.cfg)
        for count in [0, 4, 10, 25, 50]:
            columns = sets[str(count)]
            self.assertTrue(set(columns).isdisjoint({'target','asset_return','spy_return','excess_return','split','entry_date'}))
            extra_questions = {c.split('__')[1] for c in columns if c.startswith('extra__')}
            self.assertEqual(extra_questions, set(self.cfg['question_order'][4:count]))
        for a, b in zip([0, 4, 10, 25], [4, 10, 25, 50]):
            self.assertTrue(set(sets[str(a)]) < set(sets[str(b)]))

    def test_unknown_is_not_converted_to_favorable_category(self):
        features = answer_features(self.answers, self.cfg)
        # All probability mass is on unknown for this multi-level choice.
        prefix = 'extra__borrowing_terms_change__'
        self.assertTrue(all(value == 0 for name, value in features.items() if name.startswith(prefix)))
        # Fact-question ambiguity remains explicit; it is not a negative answer.
        self.assertEqual(features['extra__secured_borrowing__unclear'], 1.)
        self.assertEqual(features['extra__secured_borrowing__disclosed'], 0.)


if __name__ == '__main__':
    unittest.main()
