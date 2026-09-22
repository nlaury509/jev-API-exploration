#!/usr/bin/env python3
"""Bounded historical event research. No brokerage or order endpoints."""
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect import ROOT, Client, accepted_at, discover, get_bars, load_env, now, parse_submission, write_csv, write_json
from jev import JevClient, canonical, digest, payload_for

STUDY = ROOT / 'data/large_study_v1'


def collect_large(cfg):
    STUDY.mkdir(parents=True, exist_ok=True)
    saved = STUDY / 'config.json'
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise ValueError('Study configuration changed; use a new study directory.')
    write_json(saved, cfg)
    sec, candidates, discovery_errors = Client('sec'), [], []
    for i, symbol in enumerate(cfg['symbols'], 1):
        try:
            rows, _ = discover(sec, cfg | {'symbols': [symbol], 'max_filings': 10000})
            candidates.extend(rows)
        except (ValueError, RuntimeError) as exc:
            discovery_errors.append({'symbol': symbol, 'error': str(exc)})
        if i % 10 == 0:
            print(f'Discovery {i}/{len(cfg["symbols"])}: {len(candidates)} candidate filings', flush=True)
    write_json(STUDY / 'discovery_errors.json', discovery_errors)
    write_json(STUDY / 'candidates.json', candidates)
    # Equal maximum per calendar year prevents later high-filing years from
    # consuming the whole budget. Hash selection never uses prices or outcomes.
    selected = []
    for year in range(int(cfg['start'][:4]), int(cfg['end'][:4])):
        rows = [r for r in candidates if r['filingDate'].startswith(str(year))]
        rows.sort(key=lambda r: digest([cfg['seed'], r['accessionNumber']]))
        selected.extend(rows[:cfg['max_filings_per_year']])
    selected.sort(key=lambda r: (r['acceptanceDateTime'], r['accessionNumber']))
    write_json(STUDY / 'selected_filings.json', selected)
    print(f'Selected {len(selected)} filings; downloading original sources.', flush=True)
    events, failures, skipped = [], [], []
    def download(row):
        accession = row['accessionNumber']
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
            raise ValueError('Invalid accession.')
        meta_path = STUDY / 'filings' / accession / 'metadata.json'
        try:
            if meta_path.exists():
                event = json.loads(meta_path.read_text())
                path = STUDY / event['text_path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != event['text_sha256']:
                    raise ValueError('Previously saved source hash mismatch.')
                return 'event', event
            url = f'https://www.sec.gov/Archives/edgar/data/{row["cik"]}/{accession.replace("-", "")}/{accession}.txt'
            raw, source = Client('sec').get(url)
            primary = next(d for d in parse_submission(raw) if d['type'] == '8-K')
            text = primary['text']
            if len(text) > cfg['max_state_characters']:
                return 'skipped', {'accession': accession, 'reason': 'primary text exceeds fixed context limit', 'characters': len(text)}
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            path = meta_path.parent / 'primary.txt'
            path.write_text(text)
            event = {'accession': accession, 'symbol': row['symbol'], 'cik': row['cik'],
                     'accepted_at': accepted_at(row['acceptanceDateTime']).isoformat(), 'filing_date': row['filingDate'],
                     'items': row['items'], 'source': source, 'text_path': str(path.relative_to(STUDY)),
                     'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'characters': len(text)}
            write_json(meta_path, event)
            return 'event', event
        except (ValueError, RuntimeError) as exc:
            return 'failure', {'accession': accession, 'error': str(exc)}
    # Network I/O overlaps, but Client globally spaces SEC requests by 0.4s.
    # Each task writes a different accession/cache key.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (kind, result) in enumerate(pool.map(download, selected), 1):
            {'event': events, 'skipped': skipped, 'failure': failures}[kind].append(result)
            if i % 50 == 0:
                write_json(STUDY / 'events.json', events)
                print(f'Sources {i}/{len(selected)}: {len(events)} eligible, {len(skipped)} oversized, {len(failures)} failed', flush=True)
    write_json(STUDY / 'events.json', events)
    write_json(STUDY / 'download_errors.json', failures)
    write_json(STUDY / 'skipped_documents.json', skipped)
    symbols = sorted({e['symbol'] for e in events} | {'SPY'})
    alpaca, bars, price_errors, provenance = Client('alpaca'), [], [], []
    for i in range(0, len(symbols), 10):
        group = symbols[i:i+10]
        try:
            rows, sources = get_bars(alpaca, group, cfg['price_start'], cfg['price_end'], adjustment='all')
            bars.extend(rows)
            provenance.extend(sources)
        except (ValueError, RuntimeError) as exc:
            price_errors.append({'symbols': group, 'error': str(exc)})
        print(f'Prices {min(i+10,len(symbols))}/{len(symbols)}: {len(bars)} daily bars', flush=True)
    write_csv(STUDY / 'bars.csv', bars, ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
    write_json(STUDY / 'price_sources.json', provenance)
    write_json(STUDY / 'price_errors.json', price_errors)
    write_json(STUDY / 'collection_summary.json', {'created_at': now(), 'candidate_filings': len(candidates),
        'selected_filings': len(selected), 'eligible_filings': len(events), 'symbols_with_filings': len(symbols)-1,
        'bars': len(bars), 'oversized': len(skipped), 'download_failures': len(failures),
        'discovery_failures': len(discovery_errors), 'price_failures': len(price_errors)})
    print('Collection complete:', STUDY, flush=True)


def extract_large(cfg):
    previous_summary = STUDY / 'inference_summary.json'
    if previous_summary.exists():
        previous = json.loads(previous_summary.read_text())
        write_json(STUDY / 'inference_attempts' / f'{digest(previous)}.json', previous)
    events = json.loads((STUDY / 'events.json').read_text())
    extraction = json.loads((ROOT / 'config/extraction.json').read_text())
    if digest(extraction['questions']) != cfg['questions_sha256']:
        raise ValueError('Extraction questions changed after preregistration.')
    out = STUDY / 'jev'
    out.mkdir(exist_ok=True)
    total_spent, records, pending = 0.0, [], []
    for event in events:
        path = out / f'{event["accession"]}.json'
        text = (STUDY / event['text_path']).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != event['text_sha256']:
            raise ValueError('Source text integrity mismatch.')
        payload = payload_for(text, extraction)
        if path.exists():
            record = json.loads(path.read_text())
            if record['request_sha256'] != digest(payload) or record['response_sha256'] != digest(record['response']):
                raise ValueError('Saved inference integrity mismatch.')
            records.append(record)
            total_spent += record['response']['usage']['input_tokens'] / 1e6 * cfg['input_usd_per_million_tokens']
        else:
            pending.append((event, payload))
    failures = []
    started = time.perf_counter()
    def work(item):
        event, payload = item
        result = JevClient().predict(payload)
        record = result | {'accession': event['accession'], 'symbol': event['symbol'], 'text_sha256': event['text_sha256']}
        write_json(out / f'{event["accession"]}.json', record)
        return record
    with ThreadPoolExecutor(max_workers=cfg['inference_workers']) as pool:
        for offset in range(0, len(pending), cfg['inference_workers']):
            batch = pending[offset:offset+cfg['inference_workers']]
            # UTF-8 bytes are a conservative token reservation for these ASCII-
            # dominated texts; actual returned usage determines accounting.
            reservation = sum(len(canonical(p)) for _, p in batch) / 1e6 * cfg['input_usd_per_million_tokens']
            if total_spent + reservation > cfg['max_estimated_inference_usd']:
                failures.append({'error': 'Configured inference budget reached; remaining sources not evaluated.'})
                break
            futures = [pool.submit(work, item) for item in batch]
            for item, future in zip(batch, futures):
                try:
                    r = future.result()
                    records.append(r)
                    total_spent += r['response']['usage']['input_tokens'] / 1e6 * cfg['input_usd_per_million_tokens']
                except Exception as exc:
                    failures.append({'accession': item[0]['accession'], 'error': str(exc)})
            if failures:
                break
            if offset % 100 == 0:
                print(f'Jev {len(records)}/{len(events)}, estimated total ${total_spent:.4f}', flush=True)
    write_json(STUDY / 'inference_summary.json', {'created_at': now(), 'completed': len(records), 'requested': len(events),
        'estimated_total_usd': total_spent, 'elapsed_seconds_this_run': time.perf_counter()-started,
        'total_input_tokens': sum(r['response']['usage']['input_tokens'] for r in records),
        'workers': cfg['inference_workers'], 'new_requests_this_run': len(records) - (len(events)-len(pending)),
        'failures': failures, 'complete': len(records)==len(events)})
    print(f'Jev finished: {len(records)}/{len(events)}, estimated ${total_spent:.4f}', flush=True)
    if failures:
        raise RuntimeError('Extraction incomplete; inspect inference_summary.json before retrying.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'extract', 'extract50'])
    parser.add_argument('--limit', type=int, help='Optional number of new requests for extract50 smoke checks.')
    args = parser.parse_args()
    load_env()
    if args.command == 'extract50':
        from feature_study import extract
        extract(args.limit)
    else:
        config = json.loads((ROOT / 'config/large_study.json').read_text())
        {'collect': collect_large, 'extract': extract_large}[args.command](config)
