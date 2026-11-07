# Qwen Flash on one RTX 3090

Reproducible Qwen3.8-Flash-Next deployment for a 24 GB RTX 3090 and 64 GB RAM. Research and current blockers: [AGENTS.md](AGENTS.md).

**Status:** implementation and offline tests complete; first Strata model download/qualification in progress. **No local throughput or full-context success claimed yet.**

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

Acceptance still required:
- Coherent text, tool calls, streaming, and cancellation.
- One ~256K prompt with retrieval checks, using reported prompt-token counts and room for output/template.
- Four concurrent long sessions and actual `concurrency.serving == 4`. Shared KV capacity may not support **four full 256K histories**; do not confuse the context setting with that stronger requirement.
- Repeated solo and four-client timings, then restart/reboot persistence.

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
