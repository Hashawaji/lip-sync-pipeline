# FastLipSynth: viseme-library distillation for edge-deployable lip-sync

FastLipSynth turns an expensive audio-driven talking-head generator (e.g. Wav2Lip, VideoReTalking) into a per-actor **viseme library** so that runtime synthesis becomes lookup + per-pixel composite, with **no neural inference at playback**. The runtime is small enough to run on a 2 GB-RAM Android phone at real time.

This repository contains the offline library-construction pipeline and a Streamlit demo for end-to-end generation.

> Paper: *FastLipSynth: Generating Talking Head Videos on Edge.* Companion runtime (Android NDK + Kotlin) lives in a separate repo.

---

## What's in here

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Web UI: type text → choose an actor → get a lip-synced video |
| `run_cleopatra_baseline.py` | Headless one-shot: render a fixed 15 s clip on `actor_1` |
| `generate_cleopatra_demos.py` | Batch-render several scripted demos on `actor_1` |
| `fast_phoneme_extractor.py` | MFA wrapper that returns phoneme-level timestamps |
| `phoneme-json-transformation.py` | Quantises MFA output into triphone-indexed segments (the offline/runtime quantizer) |
| `create_triphone_visemes.py` | Slices the master video into triphone-keyed segments and encodes residuals |
| `create_video_from_triphones.py` | Runtime: composites residual segments against the I-frame anchor |
| `actor_manager.py` | Loads per-actor metadata + viseme library |
| `tts_engine.py` | Pluggable TTS backends (Edge-TTS, gTTS, Kokoro) |
| `video_generator.py` | End-to-end glue (TTS → align → segment → composite → mux) |
| `evaluation/` | FID, lip-LMD, SyncNet, latency benchmarks |
| `blink_module/` | Optional eye-blink overlay |
| `actors/` | Per-actor viseme libraries (not tracked — see *Actor data* below) |

---

## Quick start

### 1. Clone and create the conda environment

```bash
git clone https://github.com/Hashawaji/lip-sync-pipeline.git
cd lip-sync-pipeline

conda env create -f environment.yml
conda activate mfa-dev
```

The environment is built around the Montreal Forced Aligner (Kalpy backend); OpenCV, NumPy, Streamlit, and the chosen TTS backend are pulled in via `pip`.

### 2. Get an actor's viseme library

Actor libraries are several hundred MB each and are not in the git repo. You have two options:

- **Pre-built actor pack (recommended for a first run).** Download an actor's pack from the project's GitHub Releases page and extract it under `actors/`:

  ```bash
  mkdir -p actors
  # download actor_1.tar.gz from the Releases page, then:
  tar xzf actor_1.tar.gz -C actors/
  ```

- **Build a library from scratch.** Provide a curated master video + audio and run:

  ```bash
  bash generate_mfa_and_transformation.sh <master_audio> <master_text>
  python create_triphone_visemes.py --actor actor_1 --master_video <master.mp4>
  ```

  This is the full offline pipeline described in the paper.

### 3. Run the demo

**Web UI:**

```bash
streamlit run streamlit_app.py
```

Type text, pick an actor, hit *Generate*. Output video is in `outputs/`.

**Headless one-shot:**

```bash
python run_cleopatra_baseline.py
# → outputs/cleopatra_baseline_15s/output_video.mp4
```

---

## How it works (in one minute)

1. **Offline (one-time per actor).** An expensive generative model — e.g. Wav2Lip on a still portrait + a curated TTS-generated master script — produces a master video. The master is force-aligned with MFA, sliced into short segments, and each segment is indexed by its **tri-viseme** key `(left, centre, right)`. Segments are stored as residuals against a single anchor frame, so the library is roughly 85–95 % smaller than storing each frame as a full JPEG.

2. **Runtime.** Input text → TTS → MFA-align → segment-key sequence → for each key, decode the residual JPEG and composite it against the cached anchor: `frame = clip(I + (P − 128), 0, 255)`. No neural network runs at playback.

The two-stage split decouples runtime cost from the master generator's complexity. The on-device path runs in ~2 ms / frame on a mid-tier arm64 phone, at 1024×1536 resolution.

---

## Actor data

The `actors/` directory is git-ignored. Each actor needs:

```
actors/<actor_id>/
├── metadata.yaml           # display name, voice config, blink-asset path
├── portrait.png            # source identity frame
├── visemes_library/        # triphone-keyed residual segments
└── blink_assets/           # optional, for eye-blink overlay
```

`actors/actors_config.yaml` holds defaults (TTS backend, accent, etc.) that per-actor `metadata.yaml` files can override.

---

## Repository layout

- `environment.yml` — conda spec (`mfa-dev`)
- `requirements.txt` — pip-only fallback
- `Montreal-Forced-Aligner/`, `MFA/` — bundled aligner + pretrained acoustic models
- `evaluation/` — quantitative-eval scripts (FID, LipLMD, SyncNet, latency)
- `outputs/` — generated videos (git-ignored)
- `mfa_workspace_v3/`, `mfa_runtime/`, `mfa_cli_files/` — MFA scratch (git-ignored)

---

## Citation

If you build on this, please cite the FastLipSynth paper (in submission).

## License

Code is research-only at this stage; please contact the authors before redistribution.
