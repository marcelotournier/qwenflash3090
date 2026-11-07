#!/usr/bin/env python3
"""Repeat the same workload for Q2/Q3/Q4; publish compact reports, keep raw data local."""
import argparse
import json
from pathlib import Path
import statistics
from benchmark import Client, benchmark, qualify, ROOT

PASSES = ['COBALT-FALCON', 'VIOLET-OTTER', 'SILVER-IBIS']
SHORT = ('Explain in detail how NVMe, pinned RAM and GPU expert caches cooperate during '
         'sparse mixture-of-experts inference. Write at least 500 words.')
UNIT = 'We measure memory bandwidth, CPU expert work and PCIe transfers.\n'


def prompts():
    def prefill(n):
        return ('Read these notes and explain the inference pipeline in detail.\n' + UNIT * n +
                '\nExplain tradeoffs in at least 500 words.')
    long = ('Read all records below. Remember the passcodes for records amber, birch and cedar.\n' +
            UNIT * 2007 + 'Record amber has passcode COBALT-FALCON.\n' + UNIT * 8028 +
            'Record birch has passcode VIOLET-OTTER.\n' + UNIT * 8028 +
            'Record cedar has passcode SILVER-IBIS.\n' + UNIT * 2007 +
            '\nReturn only the three passcodes for amber, birch and cedar, in that order.')
    return {'solo': SHORT, 'four': SHORT, 'prefill4k': prefill(300), 'prefill16k': prefill(1200),
            'context256k': long}


def summarize(directory, quant):
    report = {'quant': quant, 'image': json.loads((ROOT / 'config/engines.json').read_text())['strata']['image'],
              'context_requested': 262144, 'slots_requested': 4, 'measurements': {},
              'limits': ['Synthetic repetitive prompts, not a general quality evaluation.',
                         'End-to-end rate includes prefill/queue time; native timings can exclude admission waits.',
                         'Long retrieval answers are short and not a sustained full-context decode benchmark.']}
    for mode in ['solo', 'four', 'prefill4k', 'prefill16k']:
        paths = sorted(directory.glob(mode + '-repeat-*.json'))
        if not paths:
            continue
        rows = [json.loads(path.read_text()) for path in paths]
        requests = [req for row in rows for req in row['requests']]
        def stats(values):
            return {'median': round(statistics.median(values), 2),
                    'min': round(min(values), 2), 'max': round(max(values), 2)}
        report['measurements'][mode] = {
            'rounds': len(rows), 'requests': len(requests),
            'engine': rows[0]['status_before']['engine'], 'model': rows[0]['model'],
            'actual_slots': rows[0]['status_before']['concurrency']['serving'],
            'prompt_tokens': stats([req['usage']['prompt_tokens'] for req in requests]),
            'prefill_tok_s': stats([req['timings']['prompt_per_second'] for req in requests]),
            'per_request_decode_tok_s': stats([req['timings']['predicted_per_second'] for req in requests]),
            'aggregate_end_to_end_tok_s': stats([row['aggregate_end_to_end_tok_s'] for row in rows])}
    for mode in ['context256k', 'context256k-four']:
        path = directory / (mode + '.json')
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        report['measurements'][mode] = {'concurrency': row['concurrency'], 'wall_s': round(row['wall_s'], 2),
            'requests': [{'usage': req['usage'], 'timings': req['timings'], 'answer': req['text'],
                          'retrieval_pass': all(code in req['text'] for code in PASSES)} for req in row['requests']]}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quant', choices=['q2', 'q3', 'q4'], required=True)
    p.add_argument('--directory', type=Path, help='Raw result directory; defaults to results/<quant>')
    p.add_argument('--run', action='store_true', help='Run the suite; otherwise summarize existing files only')
    a = p.parse_args()
    directory = a.directory or ROOT / 'results' / a.quant
    directory.mkdir(parents=True, exist_ok=True)
    error = None
    try:
        if a.run:
            client = Client('http://127.0.0.1:8000', ROOT / '.env')
            qualify(client, 4)
            data = prompts()
            for mode in ['solo', 'four', 'prefill4k', 'prefill16k']:
                for i in range(3):
                    row = benchmark(client, 4 if mode == 'four' else 1, data[mode], 256)
                    (directory / f'{mode}-repeat-{i}.json').write_text(json.dumps(row, indent=2))
                    print(mode, i, round(row['aggregate_end_to_end_tok_s'], 2), flush=True)
            for mode, concurrency in [('context256k', 1), ('context256k-four', 4)]:
                row = benchmark(client, concurrency, data['context256k'], 64)
                (directory / (mode + '.json')).write_text(json.dumps(row, indent=2))
                if not all(all(code in req['text'] for code in PASSES) for req in row['requests']):
                    raise RuntimeError(mode + ': retrieval failed')
                print(mode, 'retrieval passed', flush=True)
    except Exception as exc:
        error = repr(exc)
    report = summarize(directory, a.quant)
    if error:
        report['error'] = error
    out = ROOT / 'reports' / (a.quant + '.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(out)
    if error:
        raise SystemExit(error)


if __name__ == '__main__':
    main()
