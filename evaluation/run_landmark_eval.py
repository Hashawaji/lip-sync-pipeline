#!/usr/bin/env python3
"""
Landmark-Based Evaluation Metrics
- LMD (Landmark Distance): Average Euclidean distance of mouth landmarks
- Mouth Aspect Ratio: Height/width ratio correlation with audio
- Lip Velocity: Frame-to-frame lip movement

Uses MediaPipe Face Mesh for landmark detection.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# MediaPipe for face landmarks (using Tasks API for v0.10+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available (pip install mediapipe)")

# For audio analysis
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not available for audio analysis")


# MediaPipe Face Mesh mouth landmark indices
# Inner lip: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
# Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
MOUTH_LANDMARKS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MOUTH_LANDMARKS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_LANDMARKS_ALL = MOUTH_LANDMARKS_INNER + MOUTH_LANDMARKS_OUTER

# Key landmarks for mouth aspect ratio
UPPER_LIP_TOP = 13      # Top of upper lip
LOWER_LIP_BOTTOM = 14   # Bottom of lower lip
LEFT_CORNER = 61        # Left mouth corner
RIGHT_CORNER = 291      # Right mouth corner


class LandmarkEvaluator:
    """Extract and compare facial landmarks between videos"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe required: pip install mediapipe")
        
        # Download face landmarker model if not exists
        import urllib.request
        model_path = Path(__file__).parent / "face_landmarker.task"
        if not model_path.exists():
            print("Downloading MediaPipe Face Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, str(model_path))
            print("Model downloaded")
        
        # Create Face Landmarker
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None, 
                      sample_every: int = 1) -> Tuple[List[np.ndarray], float]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % sample_every == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
            count += 1
        
        cap.release()
        return frames, fps
    
    def get_mouth_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract mouth landmarks from frame.
        Returns array of shape (40, 2) for mouth coordinates, or None if no face.
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect face landmarks
        result = self.face_landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        face_landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract mouth landmarks
        mouth_coords = []
        for idx in MOUTH_LANDMARKS_ALL:
            lm = face_landmarks[idx]
            mouth_coords.append([lm.x * w, lm.y * h])
        
        return np.array(mouth_coords)
    
    def get_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """
        Compute mouth aspect ratio (MAR) = vertical / horizontal opening.
        Higher MAR = more open mouth.
        """
        if landmarks is None:
            return 0.0
        
        # Find key points by index in our landmark array
        inner_start = 0  # Inner lip landmarks start
        outer_start = len(MOUTH_LANDMARKS_INNER)  # Outer lip landmarks start
        
        # Get indices in our array
        upper_idx = MOUTH_LANDMARKS_INNER.index(UPPER_LIP_TOP)
        lower_idx = MOUTH_LANDMARKS_INNER.index(LOWER_LIP_BOTTOM)
        left_idx = outer_start + MOUTH_LANDMARKS_OUTER.index(LEFT_CORNER)
        right_idx = outer_start + MOUTH_LANDMARKS_OUTER.index(RIGHT_CORNER)
        
        # Vertical distance (mouth opening)
        vertical = np.linalg.norm(landmarks[upper_idx] - landmarks[lower_idx])
        
        # Horizontal distance (mouth width)
        horizontal = np.linalg.norm(landmarks[left_idx] - landmarks[right_idx])
        
        if horizontal < 1e-6:
            return 0.0
        
        return vertical / horizontal
    
    def compute_lmd(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """
        Compute Landmark Distance between two sets of mouth landmarks.
        Normalizes by face/mouth size for scale invariance.
        """
        if landmarks1 is None or landmarks2 is None:
            return float('nan')
        
        # Normalize landmarks by centering and scaling
        def normalize(lm):
            center = np.mean(lm, axis=0)
            lm_centered = lm - center
            scale = np.max(np.abs(lm_centered))
            if scale < 1e-6:
                return lm_centered
            return lm_centered / scale
        
        lm1_norm = normalize(landmarks1)
        lm2_norm = normalize(landmarks2)
        
        # Compute average Euclidean distance
        distances = np.linalg.norm(lm1_norm - lm2_norm, axis=1)
        return float(np.mean(distances))
    
    def compute_lip_velocity(self, landmarks_sequence: List[np.ndarray]) -> List[float]:
        """
        Compute frame-to-frame lip movement velocity.
        Returns list of velocity values.
        """
        velocities = []
        
        for i in range(1, len(landmarks_sequence)):
            if landmarks_sequence[i] is None or landmarks_sequence[i-1] is None:
                velocities.append(0.0)
                continue
            
            # Compute displacement of all mouth landmarks
            displacement = landmarks_sequence[i] - landmarks_sequence[i-1]
            velocity = np.mean(np.linalg.norm(displacement, axis=1))
            velocities.append(velocity)
        
        return velocities


def extract_audio_energy(video_path: str, fps: float, n_frames: int) -> np.ndarray:
    """Extract audio energy per frame from video"""
    import subprocess
    import tempfile
    
    # Extract audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_audio = f.name
    
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            temp_audio
        ], capture_output=True, check=True)
        
        # Load audio
        y, sr = librosa.load(temp_audio, sr=16000)
        
        # Compute frame-level energy
        hop_length = int(sr / fps)
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Align to frame count
        if len(energy) > n_frames:
            energy = energy[:n_frames]
        elif len(energy) < n_frames:
            energy = np.pad(energy, (0, n_frames - len(energy)))
        
        return energy
        
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def main():
    # Paths
    comparison_dir = Path.home() / "Desktop" / "lip_sync_comparison"
    ground_truth = comparison_dir / "1_GROUND_TRUTH.mp4"
    generated = comparison_dir / "2_GENERATED_60fps.mp4"
    
    if not ground_truth.exists() or not generated.exists():
        print("ERROR: Videos not found")
        sys.exit(1)
    
    print("=" * 60)
    print("LANDMARK-BASED EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = LandmarkEvaluator()
    
    # Extract frames (sample every 6th = 10fps from 60fps)
    print("\nExtracting frames (sampling every 6th frame = 10fps)...")
    gen_frames, gen_fps = evaluator.extract_frames(str(generated), sample_every=6)
    gt_frames, gt_fps = evaluator.extract_frames(str(ground_truth), sample_every=6)
    
    # Align frame counts
    min_len = min(len(gen_frames), len(gt_frames))
    gen_frames = gen_frames[:min_len]
    gt_frames = gt_frames[:min_len]
    print(f"Frames to analyze: {min_len}")
    
    # Extract landmarks for all frames
    print("\nExtracting mouth landmarks...")
    gen_landmarks = []
    gt_landmarks = []
    gen_mar = []  # Mouth Aspect Ratio
    gt_mar = []
    
    for i, (gen_f, gt_f) in enumerate(zip(gen_frames, gt_frames)):
        gen_lm = evaluator.get_mouth_landmarks(gen_f)
        gt_lm = evaluator.get_mouth_landmarks(gt_f)
        
        gen_landmarks.append(gen_lm)
        gt_landmarks.append(gt_lm)
        
        gen_mar.append(evaluator.get_mouth_aspect_ratio(gen_lm))
        gt_mar.append(evaluator.get_mouth_aspect_ratio(gt_lm))
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{min_len} frames...")
    
    print(f"  Landmarks extracted for {sum(1 for lm in gen_landmarks if lm is not None)}/{min_len} generated frames")
    print(f"  Landmarks extracted for {sum(1 for lm in gt_landmarks if lm is not None)}/{min_len} ground truth frames")
    
    # ========== LMD (Landmark Distance) ==========
    print("\n--- Computing LMD (Landmark Distance) ---")
    lmd_scores = []
    for gen_lm, gt_lm in zip(gen_landmarks, gt_landmarks):
        lmd = evaluator.compute_lmd(gen_lm, gt_lm)
        if not np.isnan(lmd):
            lmd_scores.append(lmd)
    
    lmd_mean = float(np.mean(lmd_scores)) if lmd_scores else float('nan')
    lmd_std = float(np.std(lmd_scores)) if lmd_scores else float('nan')
    print(f"  LMD: {lmd_mean:.4f} ± {lmd_std:.4f}")
    
    # ========== Mouth Aspect Ratio Analysis ==========
    print("\n--- Analyzing Mouth Aspect Ratio ---")
    gen_mar = np.array(gen_mar)
    gt_mar = np.array(gt_mar)
    
    # Correlation between generated and ground truth MAR
    valid_mask = (gen_mar > 0) & (gt_mar > 0)
    if np.sum(valid_mask) > 10:
        mar_correlation = float(np.corrcoef(gen_mar[valid_mask], gt_mar[valid_mask])[0, 1])
    else:
        mar_correlation = float('nan')
    
    print(f"  Generated MAR: mean={np.mean(gen_mar[gen_mar > 0]):.4f}, std={np.std(gen_mar[gen_mar > 0]):.4f}")
    print(f"  Ground Truth MAR: mean={np.mean(gt_mar[gt_mar > 0]):.4f}, std={np.std(gt_mar[gt_mar > 0]):.4f}")
    print(f"  MAR Correlation (Gen vs GT): {mar_correlation:.4f}")
    
    # ========== Lip Velocity ==========
    print("\n--- Computing Lip Velocity ---")
    gen_velocity = evaluator.compute_lip_velocity(gen_landmarks)
    gt_velocity = evaluator.compute_lip_velocity(gt_landmarks)
    
    gen_velocity = np.array(gen_velocity)
    gt_velocity = np.array(gt_velocity)
    
    # Correlation between velocities
    valid_vel = (gen_velocity > 0) & (gt_velocity > 0)
    if np.sum(valid_vel) > 10:
        velocity_correlation = float(np.corrcoef(gen_velocity[valid_vel], gt_velocity[valid_vel])[0, 1])
    else:
        velocity_correlation = float('nan')
    
    print(f"  Generated Velocity: mean={np.mean(gen_velocity):.4f}, std={np.std(gen_velocity):.4f}")
    print(f"  Ground Truth Velocity: mean={np.mean(gt_velocity):.4f}, std={np.std(gt_velocity):.4f}")
    print(f"  Velocity Correlation: {velocity_correlation:.4f}")
    
    # ========== Audio-Visual Correlation (if audio available) ==========
    av_correlation = None
    if LIBROSA_AVAILABLE:
        print("\n--- Audio-Visual Correlation ---")
        try:
            audio_energy = extract_audio_energy(str(generated), gen_fps / 6, min_len)
            
            # Correlation between MAR and audio energy
            valid_av = (gen_mar > 0) & (audio_energy > 0)
            if np.sum(valid_av) > 10:
                av_correlation = float(np.corrcoef(gen_mar[valid_av], audio_energy[valid_av])[0, 1])
                print(f"  MAR vs Audio Energy Correlation: {av_correlation:.4f}")
        except Exception as e:
            print(f"  Audio analysis failed: {e}")
    
    # ========== Results Summary ==========
    print("\n" + "=" * 60)
    print("LANDMARK EVALUATION RESULTS")
    print("=" * 60)
    
    results = {
        "frames_analyzed": min_len,
        "sample_rate": "every 6th frame (10fps from 60fps)",
        "lmd": {
            "mean": lmd_mean,
            "std": lmd_std,
            "valid_frames": len(lmd_scores)
        },
        "mouth_aspect_ratio": {
            "generated_mean": float(np.mean(gen_mar[gen_mar > 0])),
            "generated_std": float(np.std(gen_mar[gen_mar > 0])),
            "ground_truth_mean": float(np.mean(gt_mar[gt_mar > 0])),
            "ground_truth_std": float(np.std(gt_mar[gt_mar > 0])),
            "correlation": mar_correlation
        },
        "lip_velocity": {
            "generated_mean": float(np.mean(gen_velocity)),
            "generated_std": float(np.std(gen_velocity)),
            "ground_truth_mean": float(np.mean(gt_velocity)),
            "ground_truth_std": float(np.std(gt_velocity)),
            "correlation": velocity_correlation
        }
    }
    
    if av_correlation is not None:
        results["audio_visual_correlation"] = av_correlation
    
    # Verdict
    # LMD: lower is better (0 = identical landmark positions)
    if lmd_mean < 0.05:
        lmd_quality = "Excellent"
    elif lmd_mean < 0.10:
        lmd_quality = "Very Good"
    elif lmd_mean < 0.15:
        lmd_quality = "Good"
    elif lmd_mean < 0.20:
        lmd_quality = "Fair"
    else:
        lmd_quality = "Poor"
    
    # MAR Correlation: higher is better (1 = perfect correlation)
    if mar_correlation > 0.9:
        mar_quality = "Excellent"
    elif mar_correlation > 0.7:
        mar_quality = "Very Good"
    elif mar_correlation > 0.5:
        mar_quality = "Good"
    elif mar_correlation > 0.3:
        mar_quality = "Fair"
    else:
        mar_quality = "Poor"
    
    results["verdict"] = {
        "lmd_rating": lmd_quality,
        "lmd_interpretation": f"LMD {lmd_mean:.4f} - {lmd_quality} (lower = better, 0 = identical)",
        "mar_correlation_rating": mar_quality,
        "mar_interpretation": f"MAR Correlation {mar_correlation:.4f} - {mar_quality} (higher = better)",
        "velocity_correlation": f"Velocity Correlation {velocity_correlation:.4f}"
    }
    
    print(f"\nLMD: {lmd_mean:.4f} - {lmd_quality}")
    print(f"MAR Correlation: {mar_correlation:.4f} - {mar_quality}")
    print(f"Velocity Correlation: {velocity_correlation:.4f}")
    print("=" * 60)
    
    # Save results
    output_file = comparison_dir / "landmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
