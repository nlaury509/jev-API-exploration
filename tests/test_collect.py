import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import collect


class CollectionTests(unittest.TestCase):
    def test_acceptance_offsets_and_dst(self):
        self.assertEqual(collect.accepted_at("2025-01-06T16:01:00").hour, 21)
        self.assertEqual(collect.accepted_at("2025-07-07T16:01:00").hour, 20)
        self.assertEqual(collect.accepted_at("2025-07-07T20:01:00Z"),
                         datetime(2025, 7, 7, 20, 1, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            collect.accepted_at("")

    def test_selection_excludes_amendments_other_items_and_end_date(self):
        base = {"form": "8-K", "filingDate": "2025-01-06", "items": "1.01,9.01",
                "acceptanceDateTime": "2025-01-06T20:00:00Z"}
        rows = [base | {"accessionNumber": "A"}, base | {"accessionNumber": "B", "form": "8-K/A"},
                base | {"accessionNumber": "C", "items": "2.02,9.01"},
                base | {"accessionNumber": "D", "filingDate": "2026-01-01"}]
        cfg = {"start": "2025-01-01", "end": "2026-01-01", "items": ["1.01"], "seed": 42, "max_filings": 10}
        self.assertEqual([r["accessionNumber"] for r in collect.select_filings(rows, cfg)], ["A"])

    def test_sampling_is_independent_of_source_order_and_deduplicates(self):
        rows = [{"form": "8-K", "filingDate": "2025-01-06", "items": "1.01",
                 "acceptanceDateTime": "2025-01-06T20:00:00Z", "accessionNumber": str(i)} for i in range(20)]
        cfg = {"start": "2025-01-01", "end": "2026-01-01", "items": ["1.01"], "seed": 42, "max_filings": 5}
        self.assertEqual(collect.select_filings(rows + rows, cfg), collect.select_filings(list(reversed(rows)), cfg))
        self.assertEqual(len(collect.select_filings(rows, cfg)), 5)

    def test_original_submission_includes_agreement_but_not_graphics(self):
        raw = b'''<DOCUMENT><TYPE>8-K\n<FILENAME>main.htm\n<TEXT><html><style>ignore me</style>
        <p>Repayment extended.</p><script>also ignore</script><ix:hidden><div>hidden facts</div></ix:hidden></html></TEXT></DOCUMENT>
        <DOCUMENT><TYPE>EX-10.1\n<FILENAME>agreement.htm\n<TEXT><p>New maturity: 2030.</p></TEXT></DOCUMENT>
        <DOCUMENT><TYPE>GRAPHIC\n<FILENAME>logo.jpg\n<TEXT>binary</TEXT></DOCUMENT>'''
        docs = collect.parse_submission(raw)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["text"], "Repayment extended.")
        self.assertEqual(docs[1]["text"], "New maturity: 2030.")

    def test_pdf_exhibit_is_explicitly_unsupported(self):
        raw = b'''<DOCUMENT><TYPE>8-K\n<FILENAME>main.htm\n<TEXT><p>Main.</p></TEXT></DOCUMENT>
        <DOCUMENT><TYPE>EX-10.1\n<FILENAME>agreement.pdf\n<TEXT>binary</TEXT></DOCUMENT>'''
        self.assertFalse(collect.parse_submission(raw)[1]["text_supported"])
        with self.assertRaises(ValueError):
            collect.parse_submission(b"<html>Access denied</html>")

    def test_market_bars_sort_and_reject_conflicts_or_invalid_prices(self):
        bar = {"t": "2025-01-06T14:30:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}
        payload = {"bars": {"AAPL": [bar, bar]}}
        self.assertEqual(len(collect.normalize_bars([payload], ["AAPL"])), 1)
        with self.assertRaises(ValueError):
            collect.normalize_bars([payload, {"bars": {"AAPL": [bar | {"v": 101}]}}], ["AAPL"])
        for bad in (bar | {"h": 8}, bar | {"c": float("nan")}, bar | {"t": "2025-01-06T14:30:00"}):
            with self.assertRaises(ValueError):
                collect.normalize_bars([{"bars": {"AAPL": [bad]}}], ["AAPL"])

    def test_pagination_reaches_second_symbol(self):
        bar = {"t": "2025-01-06T14:30:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}
        class FakeClient:
            def __init__(self):
                self.urls = []
            def json(self, url, **kwargs):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return {"bars": {"AAPL": [bar]}, "next_page_token": "next"}, {}
                return {"bars": {"MSFT": [bar]}, "next_page_token": None}, {}
        fake = FakeClient()
        bars, _ = collect.get_bars(fake, ["AAPL", "MSFT"], "2025-01-06", "2025-01-07")
        self.assertEqual({b["symbol"] for b in bars}, {"AAPL", "MSFT"})
        self.assertIn("page_token=next", fake.urls[1])
        self.assertIn("feed=sip", fake.urls[0])
        self.assertIn("adjustment=raw", fake.urls[0])

    def test_credentials_are_not_sent_to_other_hosts_or_redirects(self):
        with patch.dict("os.environ", {"APCA_API_KEY_ID": "test-key", "APCA_API_SECRET_KEY": "test-secret"}):
            client = collect.Client("alpaca")
            for url in ("https://paper-api.alpaca.markets/v2/orders", "https://example.org/", "http://data.alpaca.markets/"):
                with self.assertRaises(ValueError):
                    client.get(url)
        handler = collect.SameOriginRedirects()
        with self.assertRaises(ValueError):
            handler.redirect_request(Request("https://data.alpaca.markets/x"), None, 302, "", {}, "https://example.org/")

    def test_environment_file_is_data_not_shell(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {}, clear=True):
            path = Path(folder) / ".env"
            path.write_text("TEST_VALUE=$(do-not-execute)\n")
            collect.load_env(path)
            import os
            self.assertEqual(os.environ["TEST_VALUE"], "$(do-not-execute)")


if __name__ == "__main__":
    unittest.main()
