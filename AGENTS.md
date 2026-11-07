# Goal
Create a Qwen3.8-Flash-Next server based on Project Maya for one RTX 3090 and 64 GB RAM. Target 262144-token context, four concurrent requests, 100 tok/s decode and ~1000 tok/s prefill. Evaluate Strata, NVMe streaming, and restart-unless-stopped Docker deployment. Keep this file as the live research journal. Repository: marcelotournier/qwenflash3090. Check running servers before every new launch.

# Journal — 2026-11-07

## Hardware and initial state
- RTX 3090 24576 MiB, driver 615.78.08; host reports 58 GiB RAM, 15 GiB swap, NVMe with 245 GiB free.
- Existing `glm53-flash-maya` container healthy on 127.0.0.1:8000, using 22.7 GiB VRAM and most host RAM. Do not run engines alongside it. Stop it explicitly before Qwen preparation; preserve its files/container.
- Rootless Docker works with CDI (`--device nvidia.com/gpu=all`) and `--security-opt label=disable`. Docker Compose is not installed, so use plain Docker scripts. GitHub authentication available.

## Sources and web research
- Maya: https://github.com/mw00/project-maya at `7bf15ac68f9bb5026bd5298849608c83be4ff6c2`.
- Maya Qwen settings: https://github.com/mw00/project-maya/blob/7bf15ac68f9bb5026bd5298849608c83be4ff6c2/docs/SETTINGS.md#qwen-38-flash-next-cyber-frost
- Maya base model manifest: `tools/qwen/download_unsloth.json` at that commit; UD-IQ2_XS, ~80 GB including 28.8 GB n-gram shard and 4.1 GB MTP. Experts/dense need ~47 GB combined RAM/VRAM. N-gram lookup stays on NVMe. Avoid CYBER-FROST (different fine-tune, 112.5 GB).
- Strata: https://github.com/Niko1221/Strata at `146df3c468739147f520ac3816de806f71232571`.
- https://github.com/Niko1221/Strata/blob/main/docs/BATCHING.md : native Qwen batching, `parallel:4`; engine reports actual slots and may silently reduce counts to fit VRAM. Verify `/v1/status`, not just configuration. Shared KV pool means configured 256K is not evidence of four simultaneous 256K sessions.
- https://github.com/Niko1221/Strata/blob/main/docs/DOCKER.md : official container, CONTEXT=256k, PARALLEL=4, KV=int8; persistent `/data`. Model IQ2_XS has ~35.5 GB expert shard plus ~29 GB n-gram shard; better fit than large Unsloth quants.
- Playwright CLI installed via npx. Chrome initially missing; downloaded Chromium into user cache and configured local executable. Google search returned 429 CAPTCHA. Bing search and GitHub source/issues browsing succeeded. No exact matching 3090/64 GB recipe with measured 100 tok/s was verified. Upstream performance claims are not local results.

## Implementation decisions / current blocks
- Maya's `--parallel` scheduler in `src/program/generate.cpp` is GLM-specific, not its Qwen path. Do not claim four-way Qwen decode by merely adding this flag. Retain Maya as a single-sequence baseline; evaluate Strata as the full-featured candidate.
- First candidate: official Strata 0.2.9 CUDA 13 image, ISTA-DASLab IQ2_XS, 256K context, four requested slots, INT8 KV, vision disabled, no approximate expert skipping. Record image digest after pull.
- Targets are acceptance goals, NOT measured throughput. Full-context and four-slot operation must be qualified independently.
- Keep model data, local secrets, large logs and third-party checkouts out of git.

## Implementation and trials
- Created public repo https://github.com/marcelotournier/qwenflash3090 and pushed the initial research commit.
- Added plain-Docker lifecycle with busy-GPU refusal, loopback binding, private generated key, explicit configuration, qualification-gated restart policy; stdlib concurrent benchmark with native timings; Maya source build, verified/resumable model downloader and pack commands; eleven passing offline tests (including busy-GPU refusal and a real local HTTP round trip) and CI.
- Strata image pulled and pinned: `ghcr.io/niko1221/strata@sha256:83e37db8625c67b61c579fd7c0f9068f2d831037a937f139c569502b54290998` (0.2.9, bundled engine 0.1.55).
- Explicitly stopped existing `glm53-flash-maya`, verified no remaining GPU compute processes, then launched `qwenflash-strata` on the same port 8000. Original container/data preserved; it remains stopped.
- First launch with unlimited memlock failed at OCI startup: rootless daemon cannot raise its hard limit. Retried with 8 MiB explicit memlock and NVIDIA CDI; container runs and downloads. Future script launches inherit daemon limit instead of asking for infinity. Check pinning diagnostics on engine load; CUDA allocation success is not proven by download success.
- Upstream warns 24 GB usually fits three requested slots, not four; it will sacrifice expert cache to try four. Qualification refuses silent fallback. INT8 KV kept for initial quality baseline.
- Model download is ~68 GB plus ~6 GB drafter; first launch still preparing. GPU remained idle while downloading. No throughput, four-slot success or long-context retrieval result yet.
- Optional Maya image build launched separately (CPU compilation only); log `/tmp/qwen-maya-build.log`. No Maya model download yet to avoid two large transfers/preparations competing.

## Additional tested-recipe evidence (Playwright)
- https://github.com/Niko1221/Strata/pull/5130 : single RTX 3090, i9-11900K, **94 GB RAM**, Samsung PM9A1, IQ3_S, 262144 context, INT8 KV, MTP on, streaming KV on, v0.2.10/source engine 0.1.56. Reported medians: 98.2 tok/s short decode, 86.6 at 4K, 87.4 at 16K; prefill 2062.6 at 4K and 2481.7 at 16K. 12/12 instruction/retrieval/tool sanity tests; concurrency stage NOT run. Useful evidence near target, but neither our RAM size nor four-way operation. Not a local measurement.
- GitHub search also surfaced #5327 (single-3090 long-context regression on 0.1.56 vs 0.1.55) and #4841 (3090 regression on 0.1.54/55). These are leads, not reproduced bugs; pin first, benchmark locally, then compare versions rather than blindly updating.

## Next acceptance gates
1. Finish download/pack and inspect engine startup/pinned-RAM diagnostics.
2. Run `make qualify`; require arithmetic, non-empty answers and actual four slots before enabling `unless-stopped`.
3. Benchmark repeated 4K/16K prefill and 256-token solo/four-client decode. Report both per-request and aggregate rates.
4. Verify ~256K retrieval and the stronger four-full-context requirement separately (shared KV may make the latter infeasible).
5. Only then tune NVMe/stager/MTP/KV settings or compare versions. Do not advertise target rates until measured.

## Q2 completed — local results
- Download/preparation completed. Strata engine 0.1.55 reports **four actual slots**, 262144 context, streaming KV with 32768 resident cells; each slot costs ~0.95 GiB VRAM. All experts load into a 33.02 GiB pinned arena; `cudaHostRegister PORTABLE ok` despite the rootless memlock setting. PCIe probe 13.4 GB/s; NVMe PLE direct probe **115390 rows/s**, so no NVMe override needed.
- `make qualify` passed arithmetic, context, actual slot count and four nonempty answers. Enabled and checked `restart=unless-stopped`; Docker health healthy. Explicitly stopping Q2 for quant comparisons preserves this policy and does not auto-restart it.
- Three repeated 256-output-token rounds: solo decode median **123.6 tok/s** (123.4–128.5); four-client per-request median **41.1 tok/s** (36.2–47.3); aggregate four-client end-to-end median **134.4 tok/s** (134.1–136.4), including short-prompt reads/admission.
- Repetitive ~3972-token prefill: median **1836.8 tok/s**, decode 125.6. ~15674-token prefill: median **2639.8 tok/s**, decode 125.9. INT8 KV, upstream MTP/default exact speed switches, no expert skipping. Solo goal met on these short/medium workloads; 100 tok/s PER CLIENT at concurrency four is NOT met.
- One **261032-token** prompt: 110.38 s prefill (**2364.9 tok/s**), 20 output tokens, 56.4 tok/s decode; all three synthetic retrieval passcodes correct.
- Four concurrently submitted near-full prompts: **261034–261037 input tokens each**, all three passcodes correct in every answer, 447.76 s total. Prompt work effectively serialized (~110 s each), so do not interpret this as parallel prefill or a sustained four-client decode benchmark. Short native decode figures varied 6.2–79 tok/s and exclude some admission/queue waits. Actual slots remained four.
- Memory is tight: ~58 GiB in use and <1 GiB available, ~665 MiB swap used after long tests. No container OOM or engine failure. Four full KV host states are allocated at load. Long-context retrieval is a repetitive synthetic sanity test, not a quality evaluation.
- Published compact evidence: `reports/q2.json`. Raw logs and JSON remain in `results/`. Maya image also built successfully (`sha256:e0f01724f464e19bd72ec98003d77c6bee894f7f35e4d1c9785163ca802ad8e9`); executable and pack/MTP tools' `--help` checks pass. Maya runtime baseline still not tested.

## User-requested Q3/Q4 comparison
- Stop Q2, verify empty GPU, then test Q3 **ISTA IQ3_XXS** and Q4 **Unsloth UD-IQ4_XS** at the identical 262144 context / requested four slots / INT8 / no vision, same pinned engine/image. Q4 is a mixed ~4-bit quant, not uniform Q4_K.
- Added `--quant q2|q3|q4` lifecycle profiles and a reproducible `scripts/suite.py` with the same repeated short/medium and near-full retrieval workloads; compact per-quant reports go into git.
- Q2 model SHA256 verified against upstream. Q3 n-gram shard is byte-identical (`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`) and hard-linked from Q2, saving 28.8 GB and another transfer. Q3 weight download started in `qwenflash-strata-q3`; Q2 remains stopped.
- Q4 files are being fetched/prepared in **no-start** container `qwenflash-q4-prepare`, using `STRATA_CONTROL_INSTALL=1` and upstream `setup.py --no-start`. It will not launch a competing model server. Plain setup without that environment tried the source-installer path; stopped it and switched to the bundled component path before any model download.
- Initial additional disk headroom 169 GiB, enough for Q3 weights (~47 GB), Q4 (~94 GB) and small native packs because Q3 n-gram/MTP are shared. Test engines sequentially; larger experts may require reduced RAM residency and NVMe reads.

## Q3 completed — 256K / four slots
- First attempt failed cleanly at load: setup did not stream KV at this RAM/quant combination, so four slot states used **14.61 GiB VRAM** and left no expert/prefill cache. Context and concurrency were NOT reduced.
- Fixed profiles Q3/Q4 to force `kv-resident=32768 resident-experts=on resident-budget-gib=28`. KV remains INT8; full context retained; only placement changes. This leaves enough host RAM for streaming KV without a ~40 GiB full Q3 expert arena. Q3 now has ~12.65 GiB expert VRAM; ~27.31 GiB complement plus 0.69 GiB lent-slot duplicates in the 28 GiB pinned tier. Some prefill-lent experts still read from files.
- Q4 downloader paused during Q3 benchmarks (no competing storage traffic), then resumed after Q3 stopped. Q3 qualified, healthy, `unless-stopped` enabled; explicitly stopped after tests.
- `reports/q3.json`: 3-round solo decode median **78.8 tok/s** (77.2–89.4); four-client per-request median **28.15 tok/s** (24.6–33.2); aggregate four-client end-to-end median **93.17 tok/s**.
- ~4K prefill median **1223.0 tok/s**; ~16K prefill median **1910.8 tok/s**. Same repetitive test fixtures and 256 generated tokens as Q2.
- Near-full single input **261034 tokens**: 142.10 s wall; prefill 1845.3 tok/s; short answer decode 42.7 tok/s; all passcodes correct.
- Four near-full requests: **573.25 s** wall, every request processed ~261K input tokens and returned every passcode. Same caution as Q2: serialized prompt reads and short answers, not sustained four-way full-context generation.
- Q3 misses the solo 100 tok/s target on this placement/hardware. It trades lower speed for a larger quant; no claim of measured quality improvement from these sanity tests.
