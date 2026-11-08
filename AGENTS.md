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

## Q4 completed — 256K / four slots
- Unsloth **UD-IQ4_XS** downloaded and SHA256-checked by the pinned upstream installer; prepared successfully in no-start mode. Native pack (~1.2 GiB dense conversion), shared MTP. Setup used the original pinned three-shard quant (not the newer four-shard v3.1 variant).
- Started Q4 only after Q3 stopped and GPU compute list was empty. Same engine 0.1.55, context 262144, four actual slots, INT8 KV, 32768 resident cells, 28 GiB pinned expert complement. Qualification passed; healthy, no OOM, `unless-stopped` enabled before tests.
- Q4 expert cache: **5271 slots / 11.87 GiB**. GPU-missing experts total 43.56 GiB; 28 GiB held in RAM, the rest read from files. None of the 1661 prefill-lent expert slots have duplicate RAM backing, so long prompts repeatedly fetch from SSD. Direct PLE probe 67989 rows/s is fine; the bottleneck is **expert tier capacity/reads**, not n-gram row latency.
- `reports/q4.json`: solo decode median **16.8 tok/s**; four-client per-request median **9.05 tok/s**; aggregate four-client end-to-end median **29.78 tok/s**. Three rounds of the same 256-output-token short workload.
- ~4K prefill median **406.6 tok/s**, decode 28.6; ~16K prefill median **791.3 tok/s**, decode 28.9. Cache/routing/workload effects are large, so do not extrapolate short-prompt decode to other workloads.
- Single near-full **261034-token** retrieval: **283.82 s** wall; all passcodes correct. Four near-full inputs **261031–261035 tokens**: **1169.57 s** wall, all passcodes correct in every answer. No reduced context/concurrency used.
- Q4 misses decode and prefill targets on this host. Potential future A/B: file I/O policy / RAM-vs-lent-slot placement / smaller prompt chunks, with identical prompts and output checks. No expert skipping or quality-changing shortcut was enabled. The final comparison uses the upstream adaptive I/O defaults, not an untested direct-read override.

## Final deployment / checks
- **Restored Q2 as running default** after explicitly stopping Q4 and checking GPU availability. Stop/start reload and arithmetic/four-request re-qualification passed; initial recheck at 55 s was too early, then succeeded after the load finished. Endpoint `http://127.0.0.1:8000/v1`; secret remains only in ignored `.env`.
- Q2 healthy with `restart=unless-stopped`. Q3/Q4 containers and model data retained, explicitly stopped so they do not compete or auto-start on daemon restart. Original GLM container also remains stopped and preserved.
- Rootless Docker user service is enabled, `Linger=yes`. Reboot itself was not performed. `unless-stopped` handles exited processes, not a merely unhealthy healthcheck.
- Additional Q2 API smoke: forced `add` function call produced valid arguments `{x:3,y:5}`; streaming SSE produced `4` and `[DONE]`. Local raw evidence `results/api-smoke.json`.
- Twelve offline tests pass, including profile settings, GPU conflict refusal and HTTP token accounting. GitHub repository contains compact Q2/Q3/Q4 reports, implementation, test suite, configs and this journal; model weights, secrets and large raw logs stay local.
- **Disk now only ~36 GiB free (97% used)** after retaining all three quants. Do not start the optional ~80 GB Maya download without freeing space or using another NVMe; its downloader checks capacity. Do not delete the shared Q2/Q3 n-gram inode while assuming they are independent copies.
- Remaining qualification: cancellation under load, general quality/agentic benchmarks, sustained full-context four-client generation, actual reboot recovery. The 100 tok/s goal is met by Q2 solo short/medium decode, NOT per client at four-way concurrency and NOT guaranteed at full context.

## User-requested Q4 cleanup
- Deleted Q4 GGUF directory `data/strata/Strata-data/models/unsloth-UD-IQ4_XS` and Q4-only derived pack `data/strata/Strata-data/packs/unsloth-ud-iq4_xs`.
- Reclaimed ~89 GiB; disk now ~123 GiB free. Q2/Q3 weights and shared MTP assets preserved; Q2 remains healthy and serving. Q4 reports/configuration retained, but running Q4 again requires downloading/preparing its weights again.

## Replace ISTA Q3 with Unsloth — user request
- User requested deleting ISTA Q3 and selecting the **largest Unsloth Q3 that actually sustains roughly 70s tok/s solo decode**, keeping 256K context and four provisioned slots. This is a measured selection target, not a speed promise.
- Deleted `models/IQ3_XXS` and `packs/iq3_xxs` under `data/strata/Strata-data`, and removed its stopped container. Q2's shared n-gram hard link remains valid; historical benchmark moved to `reports/ista-iq3_xxs.json` and local raw results to `results/ista-iq3_xxs`.
- First candidate: **Unsloth UD-Q3_K_XL**, 89.5 GB download / 56.2 GB experts; next smaller Q3 is **UD-IQ3_XXS**, 83.6 GB download / 50.3 GB experts. Source: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF at `d96201194f103c906994f283f504209dc8d2a10c`, four-shard Dynamic v3.1 sets. Model sizes verified against HF tree API and upstream catalog.
- These sets require engine >=0.1.57, so pulled/pinned Strata **0.2.12 / engine 0.1.58** image `ghcr.io/niko1221/strata@sha256:c3a853420670f4c0f5a948c2d9e9904af4cedf52b11bc8cf167602dc5c578bcc`. Q2/Q4 historical baseline image remains pinned separately; comparisons across engine versions are not quant-only comparisons.
- Explicitly stopped Q2 and verified no GPU compute processes before starting largest candidate in `qwenflash-strata-q3`. Requested context 262144, slots 4, INT8 streaming KV, initial 28 GiB expert complement. Download/preparation underway. Q2 preserved as fallback.
- Updated `--quant q3` to Unsloth UD-Q3_K_XL and added `--quant q3-xxs` for UD-IQ3_XXS. Suite now verifies the serving model matches the report label and supports `--quick` screening before spending time on near-full-context tests.
- UD-Q3_K_XL screening at 28 GiB expert RAM: solo decode median **39.9 tok/s** (30.9–40.1), aggregate four-client end-to-end 46.73 tok/s. `reports/unsloth-ud-q3_k_xl-28g.json`. It fails the requested ~70s target.
- Tried 32 GiB expert RAM with parked conversation cache disabled (no model/quality changes): solo median **32.3 tok/s** (19.6–34.4), aggregate four-client 34.13. More pinned memory reduces page-cache room and was not a win; expert tier remains file-bound. `reports/q3.json`. Did not spend time on full-context retrieval for the rejected speed candidate.
- Stopped largest candidate and verified empty compute GPU before fetching/starting **UD-IQ3_XXS**, initially 32 GiB expert RAM / parked cache off, same 256K and four slots. Largest candidate weights retained for now; only a measured result can select the final model.
- Shared Unsloth v3.1 n-gram shard SHA256 `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36` verified and hard-linked into the smaller quant, saving 28.8 GB. Metadata shards have equal size but different headers/hash: attempted metadata reuse was rejected by upstream SHA256 validation before engine launch; removed that bad link and let the installer fetch its correct metadata. No weights used without validation.

## Unsloth Q3 selected — 2026-11-07 local / 2026-11-08 UTC
- Selected **UD-IQ3_XXS**, the smaller of Unsloth's two Q3 sizes at the pinned revision and the largest of those tested that reached the requested ~70s range on short/medium histories. Larger UD-Q3_K_XL failed speed screening at both 28 and 32 GiB expert RAM; its weights and derived pack were removed after screening, preserving reports. ISTA Q3 remains deleted. Shared n-gram inode is now retained by the selected Unsloth quant; Q2 untouched.
- Initial UD-IQ3_XXS with 32 GiB expert RAM / 32768 resident KV: solo median **60.1 tok/s** (49.6–66.6). Requesting 34 GiB alone did not help: upstream's 4 GiB reserve clamped it to **32.17 GiB**; that run's median was 56.2 tok/s. Archived both screens in `reports/unsloth-ud-iq3_xxs-*.json`.
- Working placement: **INT8 KV**, **20480 resident cells** (full context remains **262144**), **four actual slots**, **34 GiB expert complement**, parked conversation cache off, `STRATA_RESIDENT_HEADROOM_GIB=2`. No expert pruning/skipping, routing approximation or lower-bit KV. The smaller resident GPU window frees more expert VRAM: **6863 slots / 13.04 GiB**; the remaining experts need **33.80 GiB** and now fit the pinned tier. It removes the critical file-tier misses rather than changing model math.
- Full final suite (`reports/q3.json`): three solo rounds **68.8 / 70.3 / 76.7 tok/s**, median **70.3**. ~4K prompt: decode median **72.6**, prefill **896.5**. ~16K: decode **73.8**, prefill **1491.1**. Four-client end-to-end median **76.60 tok/s aggregate**, per-request native decode median **23.25**. Neither is 70 tok/s per client at concurrency four.
- Near-full context: **261033-token** solo input, **158.01 s** wall, all three retrieval passcodes correct. Four inputs **261032–261037 tokens**, **639.76 s** wall, all passcodes correct in every answer. No OOM, engine healthy. This is serialized prompt work plus short answers, not sustained full-context generation.
- **Important limit:** the short solo retrieval answer after ~261K input decoded at **36.9 tok/s**, NOT 70. The request's 70s goal is met for short/medium histories while 256K capacity and four slots are provisioned, not at full history length. No tested Unsloth Q3 demonstrated 70 tok/s full-context decode under these constraints.
- Memory tradeoff: <1 GiB MemAvailable after long tests, ~3.7 GiB swap used. The 2 GiB startup headroom is deliberate and tighter than upstream defaults. Other RAM-heavy applications can invalidate the speed/fit result; no additional model server may run alongside it.
- Made selected Unsloth Q3 the **default profile** (`make start/stop/status/qualify`, `--quant q3`); renamed larger screening profile to `--quant q3-xl`. Running container is **qwenflash-strata-q3** on port 8000. Q2 remains explicitly stopped, available as fallback.
- Recreated the selected container from the final canonical profile and re-qualified arithmetic, actual slots/context, four nonempty calls; healthy with **restart=unless-stopped**. Post-recreate solo checks: **67.4 / 72.1 / 77.1 tok/s**, median **72.1**, recorded in `reports/q3-restart.json`. This checks persistence across cold container recreation, not reboot.
- Rejected XL cleanup restored disk to ~75 GiB free. Twelve offline tests pass, including selected model/provider, pinned new image and placement flags. Final implementation, screens, selected report, restart check and journal committed/pushed.
