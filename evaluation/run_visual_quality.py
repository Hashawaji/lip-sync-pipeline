#!/usr/bin/env python3
"""
Visual Quality Comparison Script
Compares generated lip-sync video to ground truth using:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- LPIPS (Learned Perceptual Image Patch Similarity) - if available
"""

import os
import sys
import json
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
os.chdir(script_dir.parent)

from visual_quality import VisualQualityEvaluator

# Try to import LPIPS
LPIPS_AVAILABLE = False
try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
    print("LPIPS available")
except ImportError:
    print("LPIPS not available (pip install lpips)")


class EnhancedVisualQualityEvaluator(VisualQualityEvaluator):
    """Extended evaluator with LPIPS support"""
    
    def __init__(self, mouth_region_only: bool = True, use_lpips: bool = True):
        super().__init__(mouth_region_only)
        self.lpips_model = None
        
        if LPIPS_AVAILABLE and use_lpips:
            print("Loading LPIPS model (alex)...")
            self.lpips_model = lpips.LPIPS(net='alex')
            # Use MPS if available
            if torch.backends.mps.is_available():
                self.lpips_model = self.lpips_model.to('mps')
            print("LPIPS model loaded")
    
    def compute_lpips(self, img1, img2):
        """Compute LPIPS perceptual distance"""
        import cv2
        import numpy as np
        
        if self.lpips_model is None:
            return None
        
        # Resize to same size if needed
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # Convert to tensor BCHW, normalize to [-1, 1]
        def to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = img * 2 - 1  # Normalize to [-1, 1]
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img).unsqueeze(0)  # Add batch dim
            if torch.backends.mps.is_available():
                img = img.to('mps')
            return img
        
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        
        with torch.no_grad():
            dist = self.lpips_model(t1, t2)
        
        return float(dist.cpu().item())
    
    def compare(self, generated_video, ground_truth_video, max_frames=None, sample_every=1):
        """Extended compare with LPIPS and frame sampling"""
        import cv2
        import numpy as np
        
        print(f"Loading generated video: {generated_video}")
        gen_frames, gen_fps = self.extract_frames(generated_video, max_frames)
        
        print(f"Loading ground truth video: {ground_truth_video}")
        gt_frames, gt_fps = self.extract_frames(ground_truth_video, max_frames)
        
        # Align frames
        gen_frames, gt_frames = self.align_frames(gen_frames, gt_frames)
        
        # Sample frames if requested
        if sample_every > 1:
            gen_frames = gen_frames[::sample_every]
            gt_frames = gt_frames[::sample_every]
        
        n_frames = len(gen_frames)
        print(f"Comparing {n_frames} frames (sampled every {sample_every})...")
        
        # Compute metrics
        ssim_scores = []
        psnr_scores = []
        mse_scores = []
        lpips_scores = []
        
        ssim_mouth_scores = []
        psnr_mouth_scores = []
        mse_mouth_scores = []
        lpips_mouth_scores = []
        
        for i, (gen, gt) in enumerate(zip(gen_frames, gt_frames)):
            # Full frame metrics
            ssim_scores.append(self.compute_ssim(gen, gt))
            psnr_scores.append(self.compute_psnr(gen, gt))
            mse_scores.append(self.compute_mse(gen, gt))
            
            if self.lpips_model is not None:
                lpips_scores.append(self.compute_lpips(gen, gt))
            
            # Mouth region metrics
            if self.mouth_region_only:
                gen_mouth = self.extract_mouth_region(gen)
                gt_mouth = self.extract_mouth_region(gt)
                
                ssim_mouth_scores.append(self.compute_ssim(gen_mouth, gt_mouth))
                psnr_mouth_scores.append(self.compute_psnr(gen_mouth, gt_mouth))
                mse_mouth_scores.append(self.compute_mse(gen_mouth, gt_mouth))
                
                if self.lpips_model is not None:
                    lpips_mouth_scores.append(self.compute_lpips(gen_mouth, gt_mouth))
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_frames} frames...")
        
        results = {
            'n_frames': n_frames,
            'gen_fps': gen_fps,
            'gt_fps': gt_fps,
            'full_frame': {
                'ssim_mean': float(np.mean(ssim_scores)),
                'ssim_std': float(np.std(ssim_scores)),
                'ssim_min': float(np.min(ssim_scores)),
                'psnr_mean': float(np.mean(psnr_scores)),
                'psnr_std': float(np.std(psnr_scores)),
                'mse_mean': float(np.mean(mse_scores)),
            },
        }
        
        if lpips_scores:
            results['full_frame']['lpips_mean'] = float(np.mean(lpips_scores))
            results['full_frame']['lpips_std'] = float(np.std(lpips_scores))
        
        if self.mouth_region_only:
            results['mouth_region'] = {
                'ssim_mean': float(np.mean(ssim_mouth_scores)),
                'ssim_std': float(np.std(ssim_mouth_scores)),
                'ssim_min': float(np.min(ssim_mouth_scores)),
                'psnr_mean': float(np.mean(psnr_mouth_scores)),
                'psnr_std': float(np.std(psnr_mouth_scores)),
                'mse_mean': float(np.mean(mse_mouth_scores)),
            }
            
            if lpips_mouth_scores:
                results['mouth_region']['lpips_mean'] = float(np.mean(lpips_mouth_scores))
                results['mouth_region']['lpips_std'] = float(np.std(lpips_mouth_scores))
        
        return results


def main():
    # Video paths
    comparison_dir = Path.home() / "Desktop" / "lip_sync_comparison"
    ground_truth = comparison_dir / "1_GROUND_TRUTH.mp4"
    generated = comparison_dir / "2_GENERATED_60fps.mp4"
    
    # Verify files exist
    if not ground_truth.exists():
        print(f"ERROR: Ground truth not found: {ground_truth}")
        sys.exit(1)
    if not generated.exists():
        print(f"ERROR: Generated video not found: {generated}")
        sys.exit(1)
    
    print("=" * 60)
    print("VISUAL QUALITY COMPARISON (60fps)")
    print("=" * 60)
    print(f"Ground Truth: {ground_truth.name}")
    print(f"Generated:    {generated.name}")
    print("=" * 60)
    
    # Create evaluator - skip LPIPS for speed (very slow per-frame)
    evaluator = EnhancedVisualQualityEvaluator(
        mouth_region_only=True,
        use_lpips=False  # LPIPS is too slow for 1200 frames
    )
    
    # Run comparison - sample every 6th frame (10fps equivalent from 60fps)
    # This gives us 200 frames which is statistically representative
    results = evaluator.compare(str(generated), str(ground_truth), sample_every=6)
    
    # Print results
    print("\n" + "=" * 60)
    print("VISUAL QUALITY RESULTS")
    print("=" * 60)
    
    print("\n--- Full Frame ---")
    ff = results['full_frame']
    print(f"  SSIM:  {ff['ssim_mean']:.4f} ± {ff['ssim_std']:.4f}")
    print(f"  PSNR:  {ff['psnr_mean']:.2f} ± {ff['psnr_std']:.2f} dB")
    print(f"  MSE:   {ff['mse_mean']:.2f}")
    if 'lpips_mean' in ff:
        print(f"  LPIPS: {ff['lpips_mean']:.4f} ± {ff['lpips_std']:.4f}")
    
    if 'mouth_region' in results:
        print("\n--- Mouth Region (Key Metric) ---")
        mr = results['mouth_region']
        print(f"  SSIM:  {mr['ssim_mean']:.4f} ± {mr['ssim_std']:.4f}")
        print(f"  PSNR:  {mr['psnr_mean']:.2f} ± {mr['psnr_std']:.2f} dB")
        print(f"  MSE:   {mr['mse_mean']:.2f}")
        if 'lpips_mean' in mr:
            print(f"  LPIPS: {mr['lpips_mean']:.4f} ± {mr['lpips_std']:.4f}")
    
    # Quality interpretation
    ssim_val = results.get('mouth_region', results['full_frame'])['ssim_mean']
    print("\n--- Quality Rating ---")
    if ssim_val >= 0.95:
        rating = "Excellent (SSIM >= 0.95)"
    elif ssim_val >= 0.90:
        rating = "Very Good (SSIM >= 0.90)"
    elif ssim_val >= 0.80:
        rating = "Good (SSIM >= 0.80)"
    elif ssim_val >= 0.70:
        rating = "Fair (SSIM >= 0.70)"
    else:
        rating = "Poor (SSIM < 0.70)"
    print(f"  {rating}")
    
    print("=" * 60)
    
    # Save results
    output_file = comparison_dir / "visual_quality_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
