# Qwen Flash on one RTX 3090

Qwen3.8-Flash-Next on a 24 GB RTX 3090 / 64 GB RAM host (58.7 GiB reported). Research and experiment history: [AGENTS.md](AGENTS.md).

## Selected model

**[Unsloth UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/d96201194f103c906994f283f504209dc8d2a10c/UD-IQ3_XXS)**, Dynamic v3.1, is the largest tested Unsloth Q3 that reached roughly 70 tok/s on short/medium prompts. The larger **UD-Q3_K_XL** reached only 32–40 tok/s and was rejected. Its weights were removed after screening; reports retained. The earlier **ISTA Q3** and Unsloth Q4 weights were also deleted at user request.

Selected Q3 is running as **`qwenflash-strata-q3`**, healthy, `restart=unless-stopped`, at **`http://127.0.0.1:8000/v1`**. Q2 is preserved but stopped.

### Measured results

Three repeated rounds, 256 generated tokens, **262144 configured context / four actual slots**:

| Workload | Solo decode | Four-client aggregate¹ | Prefill |
|---|---:|---:|---:|
| Short prompt | **70.3** (68.8–76.7) | **76.6** | not representative |
| ~4K prompt | **72.6** | — | **896.5** |
| ~16K prompt | **73.8** | — | **1491.1** |

Tokens/s, medians. ¹Aggregate end-to-end, including prompt/admission time; four-client **per-request decode median is 23.25**, not 70.

Near-full-context retrieval also passed: one **261033-token** input in 158.01 s, then four **261032–261037-token** inputs in 639.76 s total. Every answer retrieved all three passcodes. **Full-context short-answer solo decode was 36.9 tok/s, not 70.** The 70s result applies to short/medium histories with 256K capacity allocated, not sustained generation at 256K or per client at concurrency four. Prompt reads are effectively serialized.

Evidence: [`reports/q3.json`](reports/q3.json). Repetitive synthetic prompts are sanity/performance checks, not a general quality evaluation.

### Placement and memory

- Strata **0.2.12 / engine 0.1.58**, image pinned by digest in `config/engines.json`.
- INT8 KV, 262144 tokens per slot, **20480 KV cells resident on GPU**; full KV host copies retained. This window does not reduce context.
- **34 GiB pinned expert complement**, ~13.04 GiB GPU expert cache, parked conversation cache off.
- `STRATA_RESIDENT_HEADROOM_GIB=2` instead of the upstream 4 GiB reserve. This avoids the budget being clamped near 32 GiB, allowing the ~33.80 GiB non-GPU expert set to fit RAM. No expert skipping, routing approximation, or lower-bit KV was enabled.
- **RAM is tight**: <1 GiB available after full-context tests, ~3.7 GiB swap used. Do not co-run model servers or RAM-heavy applications. Extra host usage can clamp residency and lower speed. No OOM during the suite.
- MTP drafting; DFlash2 is not available in this four-slot mode.

## Run

Requires Python 3, GNU Make, rootless Docker with NVIDIA CDI, and NVMe storage. Fresh Q3 setup needs roughly 100–120 GB free. Model terms: Qwen Community License 1.0; review upstream licenses.

```sh
make test
# Inspect and explicitly stop any other model servers first:
docker ps
nvidia-smi
make start                         # default: selected Unsloth Q3
make logs                          # first start downloads/prepares model
make qualify                       # smoke + actual slots/context + four calls,
                                   # then enables restart=unless-stopped
python3 scripts/suite.py --quant q3 --run
```

`make start` refuses a busy compute GPU or an existing same-named container. It never deletes/stops unrelated services. To restart the existing selected deployment, check GPU availability then `docker start qwenflash-strata-q3`. To change configuration: `make stop`, `docker rm qwenflash-strata-q3`, `make start`; persistent data remains in `data/strata`.

API key: private generated `.env` (0600, git-ignored). Use its `API_KEY`; obtain the model name from authenticated `GET /v1/models`. Health: `GET /health`. Host port is loopback-only. No host networking or privileged mode; SELinux label isolation disabled for CDI compatibility.

Rootless Docker cannot raise its daemon's memlock hard limit to infinity; the script inherits it (`--memlock BYTES` is optional). CUDA pinned registration succeeded on this host. Rootless Docker service is enabled and user lingering is on; no actual reboot test performed. Docker restart policy restarts exited containers, not merely unhealthy ones.

## Profiles and historical results

| Profile | Model / engine | State |
|---|---|---|
| `q3` (default) | **Unsloth UD-IQ3_XXS**, engine 0.1.58 | selected and running |
| `q2` | ISTA IQ2_XS, engine 0.1.55 | downloaded, stopped; solo 123.6 tok/s |
| `q3-xl` | Unsloth UD-Q3_K_XL, engine 0.1.58 | rejected for speed; weights removed |
| `q4` | Unsloth UD-IQ4_XS, engine 0.1.55 | tested, weights removed |

For Q2 fallback, stop Q3 and check the GPU first, then `docker start qwenflash-strata`. Lifecycle commands accept `--quant q2`, `--quant q3-xl`, or `--quant q4`; deleted models need a fresh download. Historical quant comparisons spanning engine versions are not controlled quant-only comparisons.

Reports: [Q2](reports/q2.json), [deleted ISTA Q3](reports/ista-iq3_xxs.json), [Q4](reports/q4.json), [Unsloth larger Q3 / 28 GiB RAM](reports/unsloth-ud-q3_k_xl-28g.json), [larger Q3 / 32 GiB](reports/unsloth-ud-q3_k_xl-32g.json), [selected Q3 before tuning](reports/unsloth-ud-iq3_xxs-32g.json). Shared n-gram hard links are retained where hashes match; metadata shards are quant-specific despite equal sizes. Upstream validates every file's SHA256.

## Measurement

```sh
python3 scripts/benchmark.py --concurrency 1 --output results/solo.json
python3 scripts/benchmark.py --concurrency 4 --output results/four.json
python3 scripts/benchmark.py --prompt-file /path/to/prompt.txt --concurrency 1 --tokens 256
python3 scripts/suite.py --quant q3 --run --quick   # omit long-context stage
```

Results record actual usage, per-request native timings, model, status and aggregate wall rate. Leading identifiers prevent shared-prefix reuse. A short prompt does not measure representative prefill speed. `suite.py` verifies the actual model and repeats short/four-client/4K/16K workloads three times, then tests one/four near-256K retrieval requests. Raw data stays under `results/`; compact reports and journal are committed.

Completed: actual four slots/context checks, retrieval, coherent short outputs; Q2 streaming SSE and forced tool-call JSON smoke; Q2 stop/start qualification. Still unqualified: cancellation under load, general agentic quality, sustained full-context four-client generation, actual reboot recovery.

## Optional Maya baseline

Maya's Qwen path lacks its GLM multi-request scheduler; it is a **single-sequence baseline**, not the four-way solution.

```sh
make maya-build
make maya-download                 # ~80 GB additional + preparation; capacity checked
make maya-pack
make stop
python3 scripts/server.py start --engine maya
python3 scripts/benchmark.py --concurrency 1 --output results/maya.json
```

Source, ggml and CUDA bases are pinned in `Dockerfile.maya`; model revisions/hashes in `config/maya-model.lock.json`. Image build and executable/pack tool help checks passed; Maya runtime is untested. Model download is resumable and SHA256-verified. Free enough space before attempting it.

## Layout

- `scripts/server.py`: GPU preflight, lifecycle, selected placement, qualified restart policy.
- `scripts/benchmark.py`, `scripts/suite.py`: stdlib client, timing capture and reproducible qualification.
- `scripts/maya_model.py`: optional verified model download/preparation.
- `config/`, `reports/`, `tests/`: pinned settings, compact results, offline safety tests.

Keep weights, secrets and raw logs out of git. NVMe tuning must be measured: preserve adaptive I/O defaults unless an A/B demonstrates a gain. More pinned RAM alone is not automatically faster; the rejected larger Q3 demonstrated that.
