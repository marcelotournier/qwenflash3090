#!/usr/bin/env python3
"""Measure server-reported timings and client wall time; never infer tokens from text size."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, url, env_file):
        self.url = url.rstrip('/')
        self.key = next((line.split('=', 1)[1].strip() for line in Path(env_file).read_text().splitlines()
                         if line.startswith('API_KEY=')), '')
        if not self.key:
            raise ValueError('API_KEY missing from ' + str(env_file))

    def request(self, path, body=None):
        req = urllib.request.Request(self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=3600) as response:
            return json.load(response)

    def complete(self, model, prompt, tokens):
        started = time.perf_counter()
        response = self.request('/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': tokens, 'temperature': 0, 'reasoning_effort': 'none', 'stream': False})
        elapsed = time.perf_counter() - started
        if 'error' in response:
            raise RuntimeError(response['error'])
        text = response['choices'][0]['message'].get('content') or ''
        if not text.strip() or response.get('usage', {}).get('completion_tokens', 0) < 1:
            raise RuntimeError('Empty answer or missing token accounting: ' + str(response))
        return {'wall_s': elapsed, 'usage': response['usage'], 'timings': response.get('timings'),
                'text': text, 'finish_reason': response['choices'][0].get('finish_reason')}


def qualify(client, slots=4):
    client.request('/health')
    model = client.request('/v1/models')['data'][0]['id']
    answer = client.complete(model, 'What is 2 + 2? Answer with only the digit.', 16)
    if answer['text'].strip() != '4':
        raise RuntimeError('Arithmetic smoke test failed: ' + answer['text'])
    status = client.request('/v1/status')
    if status.get('concurrency', {}).get('serving', 0) < slots:
        raise RuntimeError('Engine did not provision requested slots: ' + json.dumps(status.get('concurrency')))
    if status.get('context', {}).get('max_positions', 0) < 262144:
        raise RuntimeError('Engine reports less than 262144 context')
    print(f'API smoke passed; engine reports >= {slots} slots and 262144 context.')
    print('This does not validate full-length prompts or performance targets.')
    return status


def benchmark(client, concurrency, prompt, tokens):
    model = client.request('/v1/models')['data'][0]['id']
    barrier = threading.Barrier(concurrency)
    def one(index):
        # Different leading content prevents cross-request prefix cache reuse.
        unique = f'Run {uuid.uuid4().hex}, client {index}.\n' + prompt
        barrier.wait(timeout=30)
        return client.complete(model, unique, tokens)
    before = client.request('/v1/status')
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(concurrency)))
    elapsed = time.perf_counter() - started
    return {'at': datetime.now(timezone.utc).isoformat(), 'model': model,
            'concurrency': concurrency, 'requested_output_tokens': tokens,
            'wall_s': elapsed, 'status_before': before, 'status_after': client.request('/v1/status'),
            'aggregate_end_to_end_tok_s': sum(r['usage']['completion_tokens'] for r in results) / elapsed,
            'requests': results,
            'notes': 'Aggregate wall rate includes prefill and queue time; use per-request engine timings for decode/prefill. '
                     'Reported context is configuration, not proof of successful full-context processing.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--env-file', type=Path, default=ROOT / '.env')
    p.add_argument('--concurrency', type=int, choices=range(1, 9), default=4)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--prompt-file', type=Path, help='Exact prompt to send, including long-context qualification inputs')
    p.add_argument('--output', type=Path, default=ROOT / 'results/benchmark.json')
    p.add_argument('--qualify', action='store_true')
    a = p.parse_args()
    if a.tokens < 1:
        p.error('--tokens must be positive')
    client = Client(a.url, a.env_file)
    if a.qualify:
        qualify(client, a.concurrency)
    prompt = a.prompt_file.read_text() if a.prompt_file else (
        'Write a detailed explanation of how NVMe, pinned host RAM and VRAM expert caches '
        'can cooperate when running a sparse mixture-of-experts language model. Use at least 500 words.')
    result = benchmark(client, a.concurrency, prompt, a.tokens)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
