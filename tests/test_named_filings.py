import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect


class NamedFilingTests(unittest.TestCase):
    def test_raw_source_and_acceptance_time_preserved_without_prices(self):
        raw = b'''<SEC-HEADER>
<ACCEPTANCE-DATETIME>20240906163000
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<FILENAME>example.htm
<TEXT><html><body>A public financing disclosure.</body></html></TEXT>
</DOCUMENT>'''
        cfg = {'selection': 'test fixture', 'filings': [
            {'accession': '0000000001-24-000001', 'cik': 1, 'symbol': 'TEST'}]}
        meta = {'url': 'https://www.sec.gov/Archives/example', 'sha256': hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp, patch.object(collect, 'ROOT', Path(tmp)), patch.object(collect, 'Client') as client:
            client.return_value.get.return_value = (raw, meta)
            with contextlib.redirect_stdout(io.StringIO()):
                run = collect.collect_named_filings(cfg)
            client.assert_called_once_with('sec')
            self.assertEqual((run / 'filings/0000000001-24-000001/submission.txt').read_bytes(), raw)
            event = json.loads((run / 'events.json').read_text())[0]
            self.assertEqual(event['accepted_at'], '2024-09-06T20:30:00+00:00')
            self.assertEqual(event['source']['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual((run / event['documents'][0]['text_path']).read_text(), 'A public financing disclosure.')
            self.assertFalse((run / 'bars.csv').exists())

    def test_invalid_or_duplicate_accessions_rejected_before_network(self):
        row = {'accession': '../bad', 'cik': 1, 'symbol': 'TEST'}
        with patch.object(collect, 'Client') as client:
            with self.assertRaises(ValueError):
                collect.collect_named_filings({'selection': 'test', 'filings': [row]})
            row['accession'] = '0000000001-24-000001'
            with self.assertRaises(ValueError):
                collect.collect_named_filings({'selection': 'test', 'filings': [row, row]})
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
