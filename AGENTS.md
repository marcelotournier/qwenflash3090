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
