#!/usr/bin/env python3
"""Read-only SEC/Alpaca research collector. Python 3.11+, no dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import threading
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
UTC = timezone.utc
NY = ZoneInfo("America/New_York")
MAX_BYTES = 32 * 1024 * 1024


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_env(path: Path = ROOT / ".env") -> None:
    """Read simple KEY=value entries without executing shell expressions."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("Invalid .env entry; use KEY=value.")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name}; set it in the local .env file.")
    return value


class SameOriginRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if new.scheme != "https" or new.netloc != old.netloc:
            raise ValueError("Refusing to forward credentials across origins.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Client:
    _sec_lock = threading.Lock()
    _sec_last_request = 0.0
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "sec":
            identity = required("SEC_USER_AGENT")
            if "@" not in identity:
                raise ValueError("SEC_USER_AGENT needs an identifying contact email.")
            self.headers = {"User-Agent": identity, "Accept-Encoding": "identity"}
            self.hosts = {"www.sec.gov", "data.sec.gov"}
        else:
            self.headers = {
                "APCA-API-KEY-ID": required("APCA_API_KEY_ID"),
                "APCA-API-SECRET-KEY": required("APCA_API_SECRET_KEY"),
            }
            self.hosts = {"data.alpaca.markets"}
        self.opener = build_opener(SameOriginRedirects())
        self.last_request = 0.0

    def get(self, url: str, *, cache: bool = True) -> tuple[bytes, dict]:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc not in self.hosts:
            raise ValueError("Only approved read-only data hosts are allowed.")
        key = hashlib.sha256(url.encode()).hexdigest()
        folder = ROOT / "data" / "cache" / self.provider
        body_path, meta_path = folder / f"{key}.bin", folder / f"{key}.json"
        if cache and body_path.exists() and meta_path.exists():
            body, meta = body_path.read_bytes(), json.loads(meta_path.read_text())
            if hashlib.sha256(body).hexdigest() != meta["sha256"]:
                raise ValueError("Cached response hash mismatch; inspect the cache.")
            return body, meta
        for attempt in range(4):
            if self.provider == 'sec':
                with Client._sec_lock:
                    time.sleep(max(0, 0.4 - (time.monotonic() - Client._sec_last_request)))
                    Client._sec_last_request = time.monotonic()
            else:
                time.sleep(max(0, 0.4 - (time.monotonic() - self.last_request)))
                self.last_request = time.monotonic()
            try:
                req = Request(url, headers=self.headers, method="GET")
                with self.opener.open(req, timeout=45) as response:
                    body = response.read(MAX_BYTES + 1)
                    if len(body) > MAX_BYTES:
                        raise ValueError("Response exceeds the 32 MiB download limit.")
                meta = {"url": url, "retrieved_at": now(),
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "cache_path": str(body_path.relative_to(ROOT))}
                if cache:
                    folder.mkdir(parents=True, exist_ok=True)
                    tmp = body_path.with_suffix(".part")
                    tmp.write_bytes(body)
                    tmp.replace(body_path)
                    write_json(meta_path, meta)
                return body, meta
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                # Do not print response bodies or authentication headers.
                raise RuntimeError(f"{self.provider}: HTTP {status} from {parts.netloc}.") from None
            except URLError:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{self.provider}: network connection failed to {parts.netloc}.") from None
        raise RuntimeError("Request failed.")

    def json(self, url: str, *, cache: bool = True):
        body, meta = self.get(url, cache=cache)
        return json.loads(body), meta


def accepted_at(value: str) -> datetime:
    if not value:
        raise ValueError("Filing lacks an acceptance timestamp.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        # Legacy EDGAR timestamp strings without an offset are Eastern time.
        parsed = parsed.replace(tzinfo=NY)
    return parsed.astimezone(UTC)


def filing_rows(columns: dict) -> list[dict]:
    n = len(columns.get("accessionNumber", []))
    for key in ("form", "filingDate", "acceptanceDateTime", "items"):
        if len(columns.get(key, [])) != n:
            raise ValueError(f"SEC submissions have a missing/misaligned {key} column.")
    return [{k: v[i] for k, v in columns.items() if isinstance(v, list) and len(v) == n}
            for i in range(n)]


def select_filings(rows: list[dict], cfg: dict) -> list[dict]:
    selected = {}
    wanted = set(cfg["items"])
    for row in rows:
        if row["form"] != "8-K" or not cfg["start"] <= row["filingDate"] < cfg["end"]:
            continue
        if not wanted.intersection(re.findall(r"\d\.\d{2}", row["items"])):
            continue
        accepted_at(row["acceptanceDateTime"])
        selected[row["accessionNumber"]] = row
    # Sample across all issuers before sorting; no outcome-based selection.
    ranked = sorted(selected.values(), key=lambda row: hashlib.sha256(
        f'{cfg["seed"]}:{row["accessionNumber"]}'.encode()).hexdigest())
    return sorted(ranked[:cfg["max_filings"]], key=lambda r: (r["acceptanceDateTime"], r["accessionNumber"]))


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if self.hidden or tag in {"script", "style", "ix:hidden"}:
            if tag not in {"br", "hr", "img", "meta", "link", "input"}:
                self.hidden.append(tag)
            return
        if tag in {"p", "div", "br", "tr", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if self.hidden:
            if tag in self.hidden:
                self.hidden = self.hidden[:len(self.hidden) - 1 - self.hidden[::-1].index(tag)]
        elif tag in {"p", "div", "tr", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def document_text(body: str) -> str:
    parser = TextExtractor()
    parser.feed(body)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip()
             for line in "".join(parser.parts).replace("\xa0", " ").splitlines()]
    return "\n".join(line for line in lines if line)


def parse_submission(raw: bytes) -> list[dict]:
    source = raw.decode("utf-8", errors="replace")
    documents = []
    for block in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", source, re.S | re.I):
        type_match = re.search(r"<TYPE>\s*([^\r\n<]+)", block, re.I)
        name_match = re.search(r"<FILENAME>\s*([^\r\n<]+)", block, re.I)
        content_match = re.search(r"<TEXT>(.*?)</TEXT>", block, re.S | re.I)
        if not type_match or not content_match:
            continue
        kind = type_match.group(1).strip().upper()
        if kind != "8-K" and not re.match(r"EX-(10|99)(?:\.|$)", kind):
            continue
        filename = name_match.group(1).strip() if name_match else "unnamed"
        supported = filename.lower().endswith((".htm", ".html", ".txt"))
        documents.append({"type": kind, "original_filename": filename,
                          "text_supported": supported,
                          "text": document_text(content_match.group(1)) if supported else ""})
    if not any(d["type"] == "8-K" and d["text"] for d in documents):
        raise ValueError("No readable primary 8-K in complete submission.")
    return documents


def discover(sec: Client, cfg: dict) -> tuple[list[dict], list[dict]]:
    ticker_data, provenance = sec.json("https://www.sec.gov/files/company_tickers.json")
    mapping = {v["ticker"]: v for v in ticker_data.values()}
    all_rows, sources = [], [provenance]
    for symbol in cfg["symbols"]:
        if symbol not in mapping:
            raise ValueError(f"No SEC CIK mapping for {symbol}.")
        cik = int(mapping[symbol]["cik_str"])
        obj, meta = sec.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        sources.append(meta)
        rows = filing_rows(obj["filings"]["recent"])
        for chunk in obj["filings"].get("files", []):
            if chunk["filingTo"] < cfg["start"] or chunk["filingFrom"] >= cfg["end"]:
                continue
            name = chunk["name"]
            if not re.fullmatch(r"CIK\d+-submissions-\d+\.json", name):
                raise ValueError("Unexpected SEC history filename.")
            old, meta = sec.json(f"https://data.sec.gov/submissions/{name}")
            sources.append(meta)
            rows.extend(filing_rows(old))
        for row in rows:
            row.update(symbol=symbol, cik=cik, company=mapping[symbol]["title"])
        all_rows.extend(rows)
        print(f"SEC metadata: {symbol}", flush=True)
    return select_filings(all_rows, cfg), sources


def normalize_bars(payloads: list[dict], symbols: list[str]) -> list[dict]:
    seen = {}
    for payload in payloads:
        for symbol, bars in (payload.get("bars") or {}).items():
            if symbol not in symbols:
                raise ValueError("Unexpected symbol in Alpaca response.")
            for bar in bars:
                ts = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    raise ValueError("Market bar timestamp has no timezone.")
                values = [float(bar[k]) for k in ("o", "h", "l", "c", "v")]
                o, h, low, c, v = values
                if not all(math.isfinite(x) for x in values) or min(o, h, low, c) <= 0 or v < 0:
                    raise ValueError("Invalid price or volume in market bar.")
                if not low <= min(o, c) <= max(o, c) <= h:
                    raise ValueError("Inconsistent OHLC range.")
                row = dict(zip(("open", "high", "low", "close", "volume"), values))
                row.update(symbol=symbol, timestamp=ts.astimezone(UTC).isoformat())
                key = symbol, row["timestamp"]
                if key in seen and row != seen[key]:
                    raise ValueError("Conflicting duplicate market bar.")
                seen[key] = row
    return [seen[key] for key in sorted(seen)]


def get_bars(alpaca: Client, symbols: list[str], start: str, end: str,
             timeframe: str = "1Day", *, cache: bool = True, adjustment: str = 'raw'):
    if adjustment not in {'raw', 'all', 'split'}:
        raise ValueError('Unsupported bar adjustment.')
    params = {"symbols": ",".join(symbols), "start": start, "end": end,
              "timeframe": timeframe, "feed": "sip", "adjustment": adjustment,
              "asof": "-", "sort": "asc", "limit": 10000}
    payloads, sources, tokens = [], [], set()
    while True:
        obj, meta = alpaca.json("https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params), cache=cache)
        if "bars" not in obj:
            raise ValueError("Alpaca response is missing bars.")
        payloads.append(obj)
        sources.append(meta)
        token = obj.get("next_page_token")
        if not token:
            break
        if token in tokens:
            raise ValueError("Alpaca pagination token repeated.")
        tokens.add(token)
        params["page_token"] = token
    return normalize_bars(payloads, symbols), sources


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(cfg: dict) -> Path:
    sec, alpaca = Client("sec"), Client("alpaca")
    run = ROOT / "data" / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    write_json(run / "config.json", cfg)
    events, sources = discover(sec, cfg)
    if not events:
        raise ValueError("No matching filings; review the universe and dates.")
    write_json(run / "selected_filings.json", events)
    completed, failures = [], []
    for i, row in enumerate(events, 1):
        accession = row["accessionNumber"]
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("Invalid SEC accession identifier.")
        url = f'https://www.sec.gov/Archives/edgar/data/{row["cik"]}/{accession.replace("-", "")}/{accession}.txt'
        try:
            body, meta = sec.get(url)
            docs = parse_submission(body)
            folder = run / "filings" / accession
            folder.mkdir(parents=True)
            (folder / "submission.txt").write_bytes(body)
            doc_records = []
            for n, doc in enumerate(docs):
                path = folder / f"document_{n:02d}.txt"
                if doc["text_supported"]:
                    path.write_text(doc["text"] + "\n")
                doc_records.append({k: v for k, v in doc.items() if k != "text"} | {
                    "text_path": str(path.relative_to(run)) if doc["text_supported"] else None,
                    "characters": len(doc["text"])})
            event = {"symbol": row["symbol"], "cik": row["cik"], "company": row["company"],
                     "accession": accession, "items": row["items"],
                     "accepted_at": accepted_at(row["acceptanceDateTime"]).isoformat(),
                     "filing_date": row["filingDate"], "source": meta,
                     "raw_path": str((folder / "submission.txt").relative_to(run)),
                     "documents": doc_records}
            write_json(folder / "metadata.json", event)
            completed.append(event)
            print(f"Filing {i}/{len(events)}: {row['symbol']} {accession}", flush=True)
        except (RuntimeError, ValueError) as exc:
            failures.append({"symbol": row["symbol"], "accession": accession, "error": str(exc)})
            print(f"Filing {i}/{len(events)}: failed ({row['symbol']})", flush=True)
    write_json(run / "events.json", completed)
    write_json(run / "failures.json", failures)
    write_json(run / "discovery_sources.json", sources)
    if not completed:
        raise RuntimeError(f"No filings downloaded. Details: {run / 'failures.json'}")
    first = min(accepted_at(e["accepted_at"]) for e in completed)
    last = max(accepted_at(e["accepted_at"]) for e in completed)
    start = (first - timedelta(days=cfg["price_lookback_days"])).date().isoformat()
    end_time = min(last + timedelta(days=cfg["price_forward_days"]), datetime.now(UTC) - timedelta(minutes=20))
    symbols = sorted({e["symbol"] for e in completed} | {cfg["benchmark"]})
    print(f"Downloading {cfg['timeframe']} historical SIP bars for {len(symbols)} symbols...", flush=True)
    bars, price_sources = get_bars(alpaca, symbols, start, end_time.isoformat(), cfg["timeframe"])
    write_csv(run / "bars.csv", bars, ["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    write_json(run / "price_sources.json", price_sources)
    coverage = {}
    for symbol in symbols:
        times = [bar["timestamp"] for bar in bars if bar["symbol"] == symbol]
        coverage[symbol] = {"bars": len(times), "first": min(times) if times else None,
                            "last": max(times) if times else None}
    summary = {"created_at": now(), "selected_filings": len(events), "downloaded_filings": len(completed),
               "failed_filings": len(failures), "bar_count": len(bars), "coverage": coverage,
               "feed": "sip", "adjustment": "raw", "symbol_mapping": "disabled (asof=-)",
               "unsupported_exhibits": sum(not d["text_supported"] for e in completed for d in e["documents"]),
               "limitations": ["Convenience universe, not a point-in-time stock universe.",
                    "Acceptance time is not necessarily first public announcement or exact dissemination time.",
                    "Raw prices require corporate-action handling before computing returns.",
                    "Bar timestamps indicate interval starts; close/volume are only known after the interval ends.",
                    "Coverage counts are not an exchange-calendar missing-bar audit.",
                    "Historical data retrieved now may contain vendor corrections.",
                    "No labels, model predictions, profitability claims, or trading execution."]}
    write_json(run / "summary.json", summary)
    report = ["# Data collection report", "", f"Filings: {len(completed)}/{len(events)}; bars: {len(bars)}", "",
              "| Symbol | Bars | First timestamp | Last timestamp |", "|---|---:|---|---|"]
    report.extend(f"| {s} | {v['bars']} | {v['first']} | {v['last']} |" for s, v in coverage.items())
    report.extend(["", "## Research limits", ""] + [f"- {s}" for s in summary["limitations"]])
    (run / "REPORT.md").write_text("\n".join(report) + "\n")
    if failures or any(not v["bars"] for v in coverage.values()):
        raise RuntimeError(f"Collection incomplete; inspect {run / 'summary.json'}")
    print(f"Saved {len(completed)} filings and {len(bars)} bars to {run}", flush=True)
    return run


def collect_named_filings(cfg: dict) -> Path:
    """Download a small explicit SEC challenge set without requesting prices."""
    rows = cfg['filings']
    if not 1 <= len(rows) <= 100:
        raise ValueError('Named filing collection must contain 1 to 100 filings.')
    if len({r['accession'] for r in rows}) != len(rows):
        raise ValueError('Duplicate accession in named collection.')
    for row in rows:
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', row['accession']) or type(row['cik']) is not int or row['cik'] <= 0:
            raise ValueError('Invalid accession or CIK.')
    run = ROOT / 'data/runs' / datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(run / 'config.json', cfg)
    sec, completed, failures = Client('sec'), [], []
    for row in rows:
        accession = row['accession']
        try:
            url = f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{accession.replace('-', '')}/{accession}.txt"
            raw, source = sec.get(url)
            folder = run / 'filings' / accession
            folder.mkdir(parents=True)
            (folder / 'submission.txt').write_bytes(raw)
            header = raw.decode('utf-8', errors='replace').split('<DOCUMENT>', 1)[0]
            stamp = re.search(r'<ACCEPTANCE-DATETIME>(\d{14})', header)
            if not stamp:
                raise ValueError('Missing SEC acceptance timestamp.')
            accepted = datetime.strptime(stamp[1], '%Y%m%d%H%M%S').replace(tzinfo=NY).astimezone(UTC).isoformat()
            documents = parse_submission(raw)
            for index, doc in enumerate(documents):
                text = doc.pop('text')
                if doc['text_supported']:
                    path = folder / f'document_{index:02d}.txt'
                    path.write_text(text)
                    doc['text_path'] = str(path.relative_to(run))
            event = row | {'accepted_at': accepted, 'source': source, 'documents': documents,
                           'selection': cfg['selection']}
            write_json(folder / 'metadata.json', event)
            completed.append(event)
            print(f"SEC named filing: {row['symbol']} {accession}", flush=True)
        except (RuntimeError, ValueError) as exc:
            failures.append({'accession': accession, 'error': str(exc)})
    write_json(run / 'events.json', completed)
    write_json(run / 'failures.json', failures)
    write_json(run / 'summary.json', {'requested': len(rows), 'downloaded': len(completed),
                                    'selection': cfg['selection'], 'prices_requested': False})
    if failures:
        raise RuntimeError(f'Named filing collection incomplete: {run}')
    print(f'Saved named filing collection: {run}')
    return run


def validate_config(cfg: dict) -> dict:
    if date.fromisoformat(cfg["start"]) >= date.fromisoformat(cfg["end"]):
        raise ValueError("start must precede end (end is exclusive for filing dates).")
    if not cfg["symbols"] or not 1 <= cfg["max_filings"] <= 1000:
        raise ValueError("Provide symbols and max_filings between 1 and 1000.")
    if len(set(cfg["symbols"])) != len(cfg["symbols"]):
        raise ValueError("Duplicate symbols in config.")
    for symbol in cfg["symbols"] + [cfg["benchmark"]]:
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", symbol):
            raise ValueError("Invalid configured symbol.")
    if cfg["timeframe"] not in {"1Day", "5Min"}:
        raise ValueError("This pilot supports 1Day or 5Min bars.")
    if min(cfg["price_lookback_days"], cfg["price_forward_days"]) < 1:
        raise ValueError("Price windows must be positive.")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Read-only live API checks; prints no credentials.")
    check.add_argument("--provider", choices=["alpaca", "sec", "all"], default="all")
    pilot = sub.add_parser("pilot", help="Collect original filings and historical prices.")
    pilot.add_argument("--config", type=Path, default=ROOT / "config" / "pilot.json")
    pilot.add_argument("--limit", type=int)
    filings = sub.add_parser('filings', help='Collect explicit historical SEC accessions; no market prices.')
    filings.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        load_env()
        if args.command == "check":
            results = {}
            if args.provider in ("alpaca", "all"):
                bars, _ = get_bars(Client("alpaca"), ["AAPL"], "2025-01-06T00:00:00Z",
                                   "2025-01-08T00:00:00Z", cache=False)
                if not bars:
                    raise ValueError("Alpaca authenticated but returned no sample bars.")
                results["alpaca"] = {"status": "ok", "sample_bars": len(bars), "feed": "sip"}
                print(f"Alpaca OK: {len(bars)} historical SIP bars; credentials not displayed.", flush=True)
            if args.provider in ("sec", "all"):
                obj, _ = Client("sec").json("https://data.sec.gov/submissions/CIK0000320193.json", cache=False)
                results["sec"] = {"status": "ok", "company": obj["name"],
                                  "recent_filings": len(filing_rows(obj["filings"]["recent"]))}
                print(f"SEC OK: {obj['name']} submission history.", flush=True)
            write_json(ROOT / "data" / "checks" / f"{args.provider}.json", {"checked_at": now(), **results})
        elif args.command == 'filings':
            collect_named_filings(json.loads(args.config.read_text()))
        else:
            cfg = json.loads(args.config.read_text())
            if args.limit is not None:
                cfg["max_filings"] = args.limit
            collect(validate_config(cfg))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
