#!/usr/bin/env python3
"""Download the upstream-pinned base Qwen model, or index it with Maya's own tools."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/maya'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, size, digest):
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['download', 'pack', 'verify'])
    a = p.parse_args()
    manifest = json.loads((ROOT / 'config/maya-model.lock.json').read_text())
    MODEL.mkdir(parents=True, exist_ok=True)
    if a.action == 'download':
        missing = sum(size - min(size, (MODEL / (name + '.part')).stat().st_size)
                      if (MODEL / (name + '.part')).exists() else size
                      for name, (size, _) in manifest['files'].items() if not (MODEL / name).exists())
        if shutil.disk_usage(MODEL).free < missing + 8 * 1024**3:
            raise SystemExit('Insufficient space: need remaining download plus 8 GiB for preparation.')
    for name, (size, digest) in manifest['files'].items():
        path = MODEL / name
        if a.action == 'download' and not path.exists():
            source = manifest.get('sources', {}).get(name, manifest)
            folder = source.get('folder', '')
            rel = '/'.join(filter(None, [folder, name]))
            url = f'https://huggingface.co/{source["repo"]}/resolve/{source["revision"]}/{quote(rel)}'
            part = MODEL / (name + '.part')
            subprocess.run(['curl', '-fL', '--retry', '5', '-C', '-', '-o', str(part), url], check=True)
            if not verify(part, size, digest):
                raise SystemExit(f'Size or SHA256 mismatch: {part}; not promoted to model file.')
            part.replace(path)
        if not verify(path, size, digest):
            raise SystemExit(f'Missing or corrupt model file: {path}')
        print('Verified', name, flush=True)
    if a.action == 'pack':
        image = json.loads((ROOT / 'config/engines.json').read_text())['maya']['image']
        first = next(iter(manifest['files']))
        commands = [
            ['tools/iq_pack.py', '--gguf', '/models/' + first, '--out', '/models/pack', '--compat-bf16'],
            ['tools/mtp_from_gguf.py', '--gguf', '/models/' + manifest['mtp'], '--out', '/models/pack/mtp']]
        for cmd in commands:
            subprocess.run(['docker', 'run', '--rm', '--security-opt', 'label=disable',
                            '-v', f'{MODEL}:/models', '--entrypoint', 'python3', image] + cmd, check=True)


if __name__ == '__main__':
    main()
