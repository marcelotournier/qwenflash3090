#!/usr/bin/env python3
"""Plain Docker lifecycle; never stops unrelated services or starts on a busy GPU."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, capture=False):
    return subprocess.run(args, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def gpu_busy():
    output = run('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                 '--format=csv,noheader', capture=True)
    return output.strip()


def ensure_key():
    path = ROOT / '.env'
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write('API_KEY=' + secrets.token_hex(24) + '\n')
    return path


def start_command(engine, cfg, port, gpu, memlock=None):
    data = ROOT / 'data' / engine
    data.mkdir(parents=True, exist_ok=True)
    cmd = ['docker', 'run', '-d', '--name', cfg['name'], '--restart', 'no',
           '--device', gpu, '--security-opt', 'label=disable', '--shm-size', '2g',
           '--stop-timeout', '120', '-p', f'127.0.0.1:{port}:8080',
           '--env-file', str(ensure_key())]
    # Rootless Docker cannot raise the daemon's memlock hard limit to infinity.
    # Inherit it unless explicitly configured for this host.
    if memlock:
        cmd += ['--ulimit', f'memlock={memlock}:{memlock}']
    if engine == 'strata':
        cmd += ['-v', f'{data}:/data']
        for key, value in cfg['environment'].items():
            cmd += ['-e', f'{key}={value}']
    else:
        if not (ROOT / 'models/maya/pack/mtp/dense.txt').exists():
            raise SystemExit('Maya pack missing. Run make maya-download maya-pack first.')
        key = next(line.split('=', 1)[1] for line in (ROOT / '.env').read_text().splitlines()
                   if line.startswith('API_KEY='))
        cmd += ['-e', 'STRATA_API_KEY=' + key,
                '-v', f'{ROOT / "models/maya"}:/models:ro',
                '-v', f'{ROOT / "config/maya.json"}:/config/maya.json:ro']
    return cmd + [cfg['image']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'stop', 'status', 'logs', 'enable-restart'])
    p.add_argument('--engine', choices=['strata', 'maya'], default='strata')
    p.add_argument('--quant', choices=['q2', 'q3', 'q3-xxs', 'q4'], default='q2',
                   help='Strata: ISTA IQ2_XS, Unsloth UD-Q3_K_XL, UD-IQ3_XXS or UD-IQ4_XS')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--gpu', default='nvidia.com/gpu=all', help='Docker CDI GPU device')
    p.add_argument('--memlock', help='optional memlock limit in bytes (daemon must permit it)')
    a = p.parse_args()
    cfg = json.loads((ROOT / 'config/engines.json').read_text())[a.engine]
    if a.engine == 'strata' and a.quant != 'q2':
        cfg['name'] += '-' + a.quant
        cfg['environment']['MODEL'] = {'q3': 'UD-Q3_K_XL', 'q3-xxs': 'UD-IQ3_XXS',
                                       'q4': 'UD-IQ4_XS'}[a.quant]
        cfg['environment']['FAMILY'] = 'unsloth'
        if a.quant in ('q3', 'q3-xxs'):
            cfg['image'] = cfg['unsloth_image']
        cfg['environment']['PRESET'] = a.quant
        # At 256K x4, keeping all KV in VRAM leaves no prefill space.
        # Stream KV and cap expert RAM so its four host copies still fit 64 GB.
        budget = 32 if a.quant in ('q3', 'q3-xxs') else 28
        cfg['environment']['STRATA_SET'] = f'kv-resident=32768 resident-experts=on resident-budget-gib={budget}'
        if a.quant in ('q3', 'q3-xxs'):
            cfg['environment']['STRATA_SET'] += ' conversation-cache-mib=0'
    name = cfg['name']
    if a.action == 'start':
        busy = gpu_busy()
        if busy:
            raise SystemExit('GPU compute processes are active; stop them explicitly first:\n' + busy)
        # Refuse existing containers; do not remove or silently restart old configuration.
        names = run('docker', 'ps', '-a', '--format', '{{.Names}}', capture=True).splitlines()
        if name in names:
            raise SystemExit(f'{name} exists. Inspect it, then docker rm {name} before recreating.')
        run(*start_command(a.engine, cfg, a.port, a.gpu, a.memlock))
    elif a.action == 'stop':
        run('docker', 'stop', '-t', '120', name)
    elif a.action == 'status':
        run('docker', 'ps', '-a', '--filter', 'name=^/' + name + '$')
        run('nvidia-smi')
    elif a.action == 'logs':
        run('docker', 'logs', '-f', '--tail', '100', name)
    else:
        # Health and API qualification must pass before enabling boot-time restart.
        sys.path.insert(0, str(ROOT / 'scripts'))
        from benchmark import Client, qualify, benchmark
        client = Client(f'http://127.0.0.1:{a.port}', ROOT / '.env')
        slots = 4 if a.engine == 'strata' else 1
        qualify(client, slots)
        result = benchmark(client, slots, 'Explain in detail how a GPU works.', 64)
        output = ROOT / 'results' / f'{a.engine}-{a.quant}-qualification.json'
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
        run('docker', 'update', '--restart', 'unless-stopped', name)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
