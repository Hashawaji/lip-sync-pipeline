# Lip-Sync Evaluation Framework Proposal

## Overview

This document outlines quantitative evaluation metrics for comparing your **triphone-based viseme concatenation** approach against state-of-the-art lip-sync methods. Your approach is unique: fast, edge-device friendly, and uses pre-recorded RealTalk master videos as the viseme library source.

---

## 1. Lip-Sync Quality Metrics (Standard in Literature)

### 1.1 SyncNet-Based Metrics

| Metric | Description | How to Compute |
|--------|-------------|----------------|
| **LSE-D** (Lip Sync Error - Distance) | Measures audio-visual embedding distance | Use pre-trained SyncNet to extract audio/video embeddings, compute Euclidean distance |
| **LSE-C** (Lip Sync Error - Confidence) | Confidence score from SyncNet | Higher = better sync. Standard benchmark metric used by Wav2Lip, PC-AVS |

**Why important:** LSE-C is the most widely cited metric in lip-sync papers. Essential for peer comparison.

### 1.2 Visual Quality Metrics

| Metric | Description | Target Region |
|--------|-------------|---------------|
| **SSIM** (Structural Similarity Index) | Perceptual similarity | Mouth region only |
| **PSNR** (Peak Signal-to-Noise Ratio) | Pixel-level reconstruction | Mouth region only |
| **LPIPS** (Learned Perceptual Image Patch Similarity) | Deep perceptual distance | Full face / mouth |
| **FID** (Fréchet Inception Distance) | Distribution-level quality | Mouth region crops |

### 1.3 Landmark-Based Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **LMD** (Landmark Distance) | Average Euclidean distance of mouth landmarks (48-68 in dlib) | Compare generated vs ground truth |
| **Mouth Aspect Ratio** | Height/width ratio of mouth opening | Temporal correlation with audio energy |
| **Lip Velocity** | Frame-to-frame movement of lip landmarks | Should correlate with phoneme transitions |

---

## 2. Edge-Device Performance Metrics (Your Unique Contribution)

Your approach targets **real-time performance on edge devices**. This is your key differentiator.

### 2.1 Latency Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Text-to-First-Frame Latency** | Time from text input to first video frame | Include TTS + MFA + video generation |
| **Per-Frame Generation Time** | Time to generate each frame | Average over video length |
| **End-to-End Latency** | Total pipeline time | For fixed-length text inputs |

### 2.2 Resource Metrics

| Metric | Description | Comparison Point |
|--------|-------------|------------------|
| **Peak Memory (RAM)** | Maximum memory during inference | Compare: Wav2Lip (~2GB), SadTalker (~4GB), Yours (~?MB) |
| **Model Size** | Total storage for models/assets | Your viseme library vs neural network weights |
| **CPU vs GPU** | Whether GPU is required | Your approach: CPU-only? |
| **FPS at Inference** | Frames generated per second | Target: 30+ FPS for real-time |

### 2.3 Scalability Metrics

| Metric | Description |
|--------|-------------|
| **Cold Start Time** | Time to load models/library |
| **Warm Inference Time** | After models loaded |
| **Battery Impact** | Power consumption on mobile (if applicable) |

---

## 3. Comparison with Other Methods

### 3.1 Methods to Compare Against

| Method | Type | Year | Key Characteristic |
|--------|------|------|-------------------|
| **Wav2Lip** | Audio-driven GAN | 2020 | Fast, widely used baseline |
| **SadTalker** | 3DMM + Flow | 2023 | Expressive, state-of-art quality |
| **DINet** | Deformable implicit | 2023 | Good generalization |
| **RealTalk** | Neural rendering | 2023 | Your master video source |
| **IP-LAP** | Landmark-based | 2023 | Efficient |
| **VideoReTalking** | GAN-based | 2022 | High quality |

### 3.2 Comparison Dimensions

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPARISON MATRIX                            │
├─────────────────────────────────────────────────────────────────┤
│                 Quality ────────────────► Speed                 │
│                    │                        │                   │
│   RealTalk ●       │                        │        ● Wav2Lip  │
│   SadTalker ●      │                        │                   │
│                    │      YOUR METHOD       │                   │
│                    │          ●?            │                   │
│                    │    (Target zone)       │                   │
│                    │                        │                   │
│                    ▼                        ▼                   │
│               High Quality            Real-time                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Master Video Comparison (Your Ground Truth)

Since you use RealTalk to generate master videos for viseme extraction, you have a natural ground truth.

### 4.1 Reconstruction Quality

| Experiment | Description |
|------------|-------------|
| **Same-Text Reconstruction** | Generate video for same text used to create viseme library. Compare output vs master. |
| **Novel-Text Generation** | Generate video for new text. Compare visual quality and naturalness. |
| **Triphone Coverage Analysis** | Report % of triphones covered by library vs fallback to diphones/monophones. |

### 4.2 Metrics for Master Comparison

| Metric | What It Measures |
|--------|------------------|
| **Frame-by-Frame SSIM** | How close each generated frame is to master |
| **Mouth Region MSE** | Pixel-level mouth accuracy |
| **Temporal Alignment Error** | Are phoneme boundaries correctly aligned? |
| **Viseme Selection Accuracy** | Did the system pick the correct triphone? |

---

## 5. User Study Metrics (Perceptual)

For paper completeness, consider:

| Study Type | Metric |
|------------|--------|
| **MOS (Mean Opinion Score)** | 1-5 rating of naturalness |
| **A/B Preference** | Side-by-side comparison with other methods |
| **Turing Test** | Can users distinguish from real video? |
| **Lip-Read Accuracy** | Can lip readers understand the speech? |

---

## 6. Proposed Evaluation Scripts

### Script 1: `syncnet_evaluator.py`
- Compute LSE-D and LSE-C using pre-trained SyncNet
- Input: video + audio
- Output: sync scores

### Script 2: `visual_quality_evaluator.py`
- Compute SSIM, PSNR, LPIPS on mouth region
- Input: generated video + reference video (master)
- Output: quality metrics

### Script 3: `landmark_evaluator.py`
- Extract dlib/mediapipe landmarks
- Compute LMD, mouth aspect ratio, lip velocity
- Input: video
- Output: landmark metrics + correlation with audio

### Script 4: `performance_benchmark.py`
- Measure latency, memory, FPS
- Input: test sentences of varying length
- Output: performance report

### Script 5: `comparison_runner.py`
- Run your method + baselines on same inputs
- Generate comparison table
- Input: test dataset, methods list
- Output: comprehensive comparison CSV + visualizations

---

## 7. Recommended Test Dataset

| Dataset | Description | Use Case |
|---------|-------------|----------|
| **LRS2** | Lip Reading Sentences | Standard benchmark |
| **LRS3** | Large-scale lip reading | More diverse |
| **GRID** | Constrained vocabulary | Controlled comparison |
| **VoxCeleb2** | Celebrity videos | Identity diversity |
| **Custom Test Set** | Your own sentences | Domain-specific evaluation |

---

## 8. Implementation Priority

### Phase 1: Core Metrics (Must Have)
1. ✅ LSE-C/LSE-D (SyncNet) - **IMPLEMENTED** in `syncnet/evaluator.py`
2. ✅ SSIM/PSNR mouth region - **IMPLEMENTED** in `visual_quality.py`
3. ⬜ Latency/FPS benchmark - Your key advantage

### Phase 2: Comparison (Should Have)
4. ⬜ LMD (Landmark distance)
5. ⬜ Wav2Lip baseline comparison
6. ⬜ Master video comparison framework

### Phase 3: Publication-Ready (Nice to Have)
7. ⬜ LPIPS/FID scores
8. ⬜ User study protocol
9. ⬜ Multi-method comparison table

---

## 9. Expected Results Format

```
┌────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION RESULTS SUMMARY                          │
├────────────────────────────────────────────────────────────────────────┤
│ Method          │ LSE-C ↑ │ SSIM ↑ │ LMD ↓ │ FPS ↑ │ Memory ↓ │ GPU   │
├─────────────────┼─────────┼────────┼───────┼───────┼──────────┼───────┤
│ RealTalk        │  9.2    │ 0.95   │ 1.2   │  0.5  │  8 GB    │  Yes  │
│ Wav2Lip         │  8.5    │ 0.82   │ 2.1   │ 25    │  2 GB    │  Yes  │
│ SadTalker       │  8.8    │ 0.88   │ 1.8   │  3    │  4 GB    │  Yes  │
│ Ours (Triphone) │  ?      │ ?      │ ?     │ 60+   │ 200 MB   │  No   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Questions for You

Before I implement, please clarify:

1. **Which metrics are highest priority?** (I suggest: LSE-C, SSIM, Latency)
2. **Do you have access to pre-trained SyncNet?** (Or should I include download script?)
3. **Which baseline methods should I include?** (Wav2Lip is easiest to set up)
4. **Do you want user study tooling?** (Web interface for MOS collection)
5. **What test sentences should I use?** (Standard datasets vs custom?)

---

## Next Steps

Once you confirm the approach, I will implement:
1. Modular evaluation framework with pluggable metrics
2. CLI interface for running evaluations
3. Comparison scripts for baselines
4. Visualization and report generation
