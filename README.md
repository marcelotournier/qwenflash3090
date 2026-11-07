# Qwen Flash on one RTX 3090

Reproducible Qwen3.8-Flash-Next deployment for a 24 GB RTX 3090 and 64 GB RAM. Research and current blockers: [AGENTS.md](AGENTS.md).

**Status:** Q2, Q3 and Q4 all passed the same 256K/four-slot suite, including five near-full-context retrieval requests each. Results: [`Q2`](reports/q2.json), [`Q3`](reports/q3.json), [`Q4`](reports/q4.json). **Q2 is restored as the running default** on `127.0.0.1:8000`, with `restart=unless-stopped`; Q3/Q4 are explicitly stopped.

| Quant | Solo decode | Four-client aggregate¹ | ~4K prefill | ~16K prefill |
|---|---:|---:|---:|---:|
| IQ2_XS | 123.6 | 134.4 | 1836.8 | 2639.8 |
| IQ3_XXS | 78.8 | 93.2 | 1223.0 | 1910.8 |
| UD-IQ4_XS | 16.8 | 29.8 | 406.6 | 791.3 |

Tokens/s, medians of three rounds. ¹End-to-end, includes short-prompt reads and admission. All run with 262144 context and four provisioned slots.

Q2 medians (3 rounds): **123.6 tok/s solo decode**, **134.4 tok/s aggregate four-client end-to-end**, **1836.8 / 2639.8 tok/s prefill** at ~4K / ~16K. Four-client per-request decode median: **41.1 tok/s**, not 100. All five near-256K retrieval requests returned the three passcodes; this synthetic test is not a general quality benchmark.

## Engines

| Engine | Model | Context requested | Concurrent decode |
|---|---|---:|---|
| Strata 0.2.9 / engine 0.1.55 (primary) | ISTA-DASLab IQ2_XS, original Qwen | 262144 | 4 requested; verify actual provisioned slots |
| Project Maya (baseline) | Unsloth UD-IQ2_XS, original Qwen | 262144 | 1; its Qwen path lacks GLM's scheduler |

Primary image is pinned by digest in `config/engines.json`. Maya source, ggml and CUDA bases are pinned in `Dockerfile.maya`; model revisions, sizes and SHA256 are pinned in `config/maya-model.lock.json`. Different quantizations mean this is a deployment comparison, not a controlled engine-only comparison.

Goals: ~100 tok/s decode and ~1000 tok/s prefill. Distinguish **per-request decode** from **aggregate throughput**. Four requests are separate HTTP calls, not OpenAI `n:4`.

## Run

Requires Python 3, GNU Make, Docker, NVIDIA Container Toolkit/CDI and an NVMe with about 120 GB free for Strata preparation. Model license: Qwen Community License 1.0; review the upstream model terms.

```sh
make test
# Inspect and explicitly stop any other model servers first:
docker ps
nvidia-smi
# On this host the pre-existing server is glm53-flash-maya.
make start
make logs                       # first start downloads ~74 GB including the drafter
# Once loaded:
make qualify                    # API + arithmetic, actual slots/context, four calls;
                                # then sets restart=unless-stopped
make bench                      # records engine timings and wall time in results/
```

`make start` refuses a busy compute GPU or an existing same-named container. It never stops or deletes unrelated containers. For a stopped existing deployment, inspect it then `docker start qwenflash-strata` (check GPU availability first). To apply configuration changes, `make stop`, `docker rm qwenflash-strata`, then `make start`; data remains in `data/strata`.

Host port: `127.0.0.1:8000`; OpenAI base URL: `http://127.0.0.1:8000/v1`. A random local API key is created in `.env` (0600, ignored by git). Use its `API_KEY` value; discover the model name with authenticated `GET /v1/models`. Health: `GET /health`.

Rootless Docker uses `--device nvidia.com/gpu=all` here. It cannot raise the daemon's memlock hard limit to infinity; the script inherits the daemon limit. `--memlock BYTES` is available if the host permits it. Only this machine's loopback port is published; SELinux label isolation is disabled for NVIDIA CDI compatibility. No privileged mode or host networking.

`make qualify` enables restart only after basic operation succeeds. It does **not** assert that performance goals or 256K prompts pass. A Docker healthcheck does not restart an unhealthy process; `unless-stopped` restarts exited containers. Rootless Docker must itself start at login/boot for this policy to take effect.

## Measurement and acceptance

```sh
python3 scripts/benchmark.py --concurrency 1 --output results/solo.json
python3 scripts/benchmark.py --concurrency 4 --output results/four.json
# For a representative long input (budget room for template/output):
python3 scripts/benchmark.py --prompt-file /path/to/prompt.txt --concurrency 1 --tokens 128
```

Results include actual token usage, per-request server `timings`, model, status snapshots and aggregate end-to-end rate (includes prefill/queue time). Leading request identifiers avoid shared-prefix reuse. A short default prompt measures decode, **not representative prefill speed**. Use a known 4K/16K-token input for prefill; repeat at least three times and report cold/warm state. Verify `usage.prompt_tokens` rather than assuming character counts equal tokens.

Qualification completed:
- Three rounds per quant: solo/four-client decode and ~4K/~16K prefill, with four actual slots.
- Per quant: one and four concurrently submitted **~261K-token** prompts; every answer retrieved all three passcodes. Four prompt reads are effectively serialized; these short retrieval answers do not measure sustained full-context generation.
- Q2 forced function call returned valid `add(x=3,y=5)` JSON; streaming SSE returned the correct answer and `[DONE]`.
- Q2 stopped/restarted successfully and re-qualified. Rootless Docker service is enabled and user lingering is on; no actual reboot test performed.

Still unqualified: cancellation under load, long-form answer quality, sustained four-client full-context decode, and actual reboot recovery. None of the quants reaches 100 tok/s **per client** at concurrency four. Q4 is constrained by expert/file-tier traffic and fails the original throughput targets on this host.

## Compare Q3 / Q4

```sh
make stop                         # stop Q2; check GPU before every switch
python3 scripts/server.py start --quant q3   # ISTA IQ3_XXS, same 256K / four slots
python3 scripts/suite.py --quant q3 --run    # once loaded; ~10 minutes of long tests
python3 scripts/server.py stop --quant q3
python3 scripts/server.py start --quant q4   # Unsloth UD-IQ4_XS, same settings
python3 scripts/suite.py --quant q4 --run
```

The profiles share `data/strata` and the MTP assets but have separate container/preset names. Q3/Q4 force streaming KV (`kv-resident=32768`) and a 28 GiB resident expert complement to fit the four full-context states; without streaming KV, Q3 failed at prefill-buffer allocation. Context and concurrency are not reduced. Q3's byte-identical n-gram shard can be hard-linked after verifying its SHA256 against the upstream manifest, saving 28.8 GB. Q4 uses a different n-gram shard. Allow another ~140 GB for Q3/Q4 plus preparation. `suite.py` repeats solo/four-client decode and 4K/16K prefill three times, then runs one and four near-256K retrieval requests. Compact reports are committed under `reports/`; raw files stay local under `results/<quant>`.

## Maya baseline

```sh
make maya-build
make maya-download              # ~80 GB additional; resumable, SHA256 checked
make maya-pack                  # CPU-only indexing + MTP preparation
make stop                       # never run both engines together
python3 scripts/server.py start --engine maya
python3 scripts/benchmark.py --concurrency 1 --output results/maya.json
```

Maya uses a 32 GB expert RAM cap, INT8 KV and NVMe-mapped n-grams. MTP weights must load even with drafting disabled (`--mtp-max-t 1`); benchmark before enabling proposals. No expert skipping or altered routing is enabled. The image is an optional baseline, not a claimed four-way solution.

## NVMe / RAM tuning

Keep `data/` and `models/` on NVMe. The ~29 GB n-gram table must not be copied into the already-constrained host RAM. Prefer IQ2_XS so experts fit in RAM + VRAM without constant SSD misses. Preserve upstream adaptive read/staging defaults until a measured baseline exists. A large thread count is not automatically faster.

For a measured bottleneck, compare identical prompts/restarts with direct vs mapped PLE reads and upstream stager settings (`STRATA_STAGER_THREADS`, `STRATA_STAGER_SSD`); record engine version, storage reads, timings and output quality. Avoid global cache drops or destructive disk benchmarks. Maya has different knobs (`MAYA_EXPERT_STAGERS`); do not reuse GLM-only settings for Qwen.

## Layout

- `scripts/server.py`: lifecycle, GPU preflight and qualified restart policy.
- `scripts/benchmark.py`: concurrent client and timing capture, stdlib only.
- `scripts/maya_model.py`: pinned, verified download and pack preparation.
- `config/`: pinned engines, Maya model manifest and baseline settings.
- `tests/`: offline safety/configuration/measurement tests; no model downloads.

Model files, keys and raw benchmark outputs are intentionally not committed. Summaries and blockers belong in `AGENTS.md`.
