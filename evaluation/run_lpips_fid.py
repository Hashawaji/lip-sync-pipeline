#!/usr/bin/env python3
"""
LPIPS and FID Evaluation Script
- LPIPS: Learned Perceptual Image Patch Similarity (per-frame)
- FID: Fréchet Inception Distance (distribution-level)
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchvision import transforms
from scipy import linalg

# Import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available")

# For FID - use Inception v3
from torchvision.models import inception_v3, Inception_V3_Weights


def extract_frames(video_path, max_frames=None, sample_every=1):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
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
    return frames


def extract_mouth_region(frame, bounds=(0.3, 0.5, 0.7, 0.85)):
    """Extract mouth region using relative coordinates"""
    h, w = frame.shape[:2]
    x1 = int(w * bounds[0])
    y1 = int(h * bounds[1])
    x2 = int(w * bounds[2])
    y2 = int(h * bounds[3])
    return frame[y1:y2, x1:x2]


class LPIPSEvaluator:
    """Compute LPIPS perceptual distance"""
    
    def __init__(self, device='mps'):
        print("Loading LPIPS model (alex)...")
        self.device = device
        self.model = lpips.LPIPS(net='alex').to(device)
        self.model.eval()
        print("LPIPS model loaded")
    
    def to_tensor(self, img):
        """Convert numpy image to normalized tensor"""
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)
    
    def compute(self, img1, img2):
        """Compute LPIPS between two images"""
        # Resize to same size if needed
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        t1 = self.to_tensor(img1)
        t2 = self.to_tensor(img2)
        
        with torch.no_grad():
            dist = self.model(t1, t2)
        
        return float(dist.cpu().item())


class FIDEvaluator:
    """Compute FID using Inception v3 features"""
    
    def __init__(self, device='mps'):
        print("Loading Inception v3 for FID...")
        self.device = device
        
        # Load Inception v3
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # Remove classification layer
        self.model = self.model.to(device)
        self.model.eval()
        
        # Preprocessing for Inception
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        print("Inception v3 loaded")
    
    def get_features(self, images):
        """Extract Inception features from list of images"""
        features = []
        
        for img in images:
            # Transform image
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(tensor)
            
            features.append(feat.cpu().numpy().flatten())
        
        return np.array(features)
    
    def compute_statistics(self, features):
        """Compute mean and covariance of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Compute FID between two distributions"""
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Handle imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


def main():
    # Paths
    comparison_dir = Path.home() / "Desktop" / "lip_sync_comparison"
    ground_truth = comparison_dir / "1_GROUND_TRUTH.mp4"
    generated = comparison_dir / "2_GENERATED_60fps.mp4"
    
    if not ground_truth.exists() or not generated.exists():
        print("ERROR: Videos not found")
        sys.exit(1)
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("LPIPS & FID EVALUATION")
    print("=" * 60)
    
    # Extract frames (sample every 12th = 100 frames from 1200)
    print("\nExtracting frames (sampling every 12th frame)...")
    gen_frames = extract_frames(generated, sample_every=12)
    gt_frames = extract_frames(ground_truth, sample_every=12)
    
    # Align to same length
    min_len = min(len(gen_frames), len(gt_frames))
    gen_frames = gen_frames[:min_len]
    gt_frames = gt_frames[:min_len]
    print(f"Frames to compare: {min_len}")
    
    # Extract mouth regions
    print("Extracting mouth regions...")
    gen_mouths = [extract_mouth_region(f) for f in gen_frames]
    gt_mouths = [extract_mouth_region(f) for f in gt_frames]
    
    results = {
        "frames_compared": min_len,
        "sample_rate": "every 12th frame"
    }
    
    # ========== LPIPS ==========
    if LPIPS_AVAILABLE:
        print("\n--- Computing LPIPS ---")
        lpips_eval = LPIPSEvaluator(device)
        
        lpips_scores_full = []
        lpips_scores_mouth = []
        
        for i, (gen, gt, gen_m, gt_m) in enumerate(zip(gen_frames, gt_frames, gen_mouths, gt_mouths)):
            # Full frame
            score_full = lpips_eval.compute(gen, gt)
            lpips_scores_full.append(score_full)
            
            # Mouth region
            score_mouth = lpips_eval.compute(gen_m, gt_m)
            lpips_scores_mouth.append(score_mouth)
            
            if (i + 1) % 20 == 0:
                print(f"  LPIPS: {i + 1}/{min_len} frames...")
        
        results["lpips"] = {
            "full_frame": {
                "mean": float(np.mean(lpips_scores_full)),
                "std": float(np.std(lpips_scores_full)),
                "min": float(np.min(lpips_scores_full)),
                "max": float(np.max(lpips_scores_full))
            },
            "mouth_region": {
                "mean": float(np.mean(lpips_scores_mouth)),
                "std": float(np.std(lpips_scores_mouth)),
                "min": float(np.min(lpips_scores_mouth)),
                "max": float(np.max(lpips_scores_mouth))
            }
        }
        
        print(f"\n  LPIPS Full Frame:  {results['lpips']['full_frame']['mean']:.4f} ± {results['lpips']['full_frame']['std']:.4f}")
        print(f"  LPIPS Mouth Region: {results['lpips']['mouth_region']['mean']:.4f} ± {results['lpips']['mouth_region']['std']:.4f}")
    
    # ========== FID ==========
    print("\n--- Computing FID ---")
    fid_eval = FIDEvaluator(device)
    
    # Full frame FID
    print("  Extracting Inception features (full frame)...")
    gen_features_full = fid_eval.get_features(gen_frames)
    gt_features_full = fid_eval.get_features(gt_frames)
    
    mu_gen_full, sigma_gen_full = fid_eval.compute_statistics(gen_features_full)
    mu_gt_full, sigma_gt_full = fid_eval.compute_statistics(gt_features_full)
    fid_full = fid_eval.compute_fid(mu_gen_full, sigma_gen_full, mu_gt_full, sigma_gt_full)
    
    # Mouth region FID
    print("  Extracting Inception features (mouth region)...")
    gen_features_mouth = fid_eval.get_features(gen_mouths)
    gt_features_mouth = fid_eval.get_features(gt_mouths)
    
    mu_gen_mouth, sigma_gen_mouth = fid_eval.compute_statistics(gen_features_mouth)
    mu_gt_mouth, sigma_gt_mouth = fid_eval.compute_statistics(gt_features_mouth)
    fid_mouth = fid_eval.compute_fid(mu_gen_mouth, sigma_gen_mouth, mu_gt_mouth, sigma_gt_mouth)
    
    results["fid"] = {
        "full_frame": float(fid_full),
        "mouth_region": float(fid_mouth)
    }
    
    print(f"\n  FID Full Frame:  {fid_full:.2f}")
    print(f"  FID Mouth Region: {fid_mouth:.2f}")
    
    # ========== Interpretation ==========
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # LPIPS interpretation (lower = better, 0 = identical)
    if "lpips" in results:
        lpips_val = results["lpips"]["mouth_region"]["mean"]
        if lpips_val < 0.05:
            lpips_quality = "Excellent (nearly identical)"
        elif lpips_val < 0.10:
            lpips_quality = "Very Good"
        elif lpips_val < 0.20:
            lpips_quality = "Good"
        elif lpips_val < 0.30:
            lpips_quality = "Fair"
        else:
            lpips_quality = "Poor"
        
        results["verdict_lpips"] = {
            "rating": lpips_quality,
            "interpretation": f"LPIPS {lpips_val:.4f} - {lpips_quality}",
            "note": "LPIPS measures perceptual distance (0 = identical, lower = better)"
        }
        print(f"\nLPIPS Mouth: {lpips_val:.4f} - {lpips_quality}")
    
    # FID interpretation (lower = better, 0 = identical distributions)
    fid_val = results["fid"]["mouth_region"]
    if fid_val < 10:
        fid_quality = "Excellent (very similar distributions)"
    elif fid_val < 30:
        fid_quality = "Very Good"
    elif fid_val < 50:
        fid_quality = "Good"
    elif fid_val < 100:
        fid_quality = "Fair"
    else:
        fid_quality = "Poor"
    
    results["verdict_fid"] = {
        "rating": fid_quality,
        "interpretation": f"FID {fid_val:.2f} - {fid_quality}",
        "note": "FID measures distribution distance (0 = identical, lower = better)"
    }
    print(f"FID Mouth:   {fid_val:.2f} - {fid_quality}")
    
    print("=" * 60)
    
    # Save results
    output_file = comparison_dir / "lpips_fid_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
