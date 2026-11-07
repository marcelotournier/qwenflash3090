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
