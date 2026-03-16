"""
Visual Quality Evaluator

Compares generated video to ground truth using:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- LPIPS (optional, requires pretrained network)

Focused on mouth region comparison for lip-sync evaluation.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available, SSIM/PSNR metrics disabled")


class VisualQualityEvaluator:
    """
    Compare generated video frames to ground truth.
    
    Computes visual quality metrics focused on lip/mouth region.
    
    Usage:
        evaluator = VisualQualityEvaluator()
        results = evaluator.compare(generated_video, ground_truth_video)
    """
    
    def __init__(self, mouth_region_only: bool = True):
        """
        Initialize evaluator.
        
        Args:
            mouth_region_only: If True, compute metrics only on mouth region
        """
        self.mouth_region_only = mouth_region_only
        
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required: pip install scikit-image")
    
    def extract_frames(self, 
                      video_path: str, 
                      max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """
        Extract frames from video.
        
        Returns:
            frames: List of RGB frames
            fps: Video framerate
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            count += 1
            if max_frames and count >= max_frames:
                break
        
        cap.release()
        return frames, fps
    
    def extract_mouth_region(self, 
                            frame: np.ndarray,
                            relative_bounds: Tuple[float, float, float, float] = (0.3, 0.5, 0.7, 0.85)
                            ) -> np.ndarray:
        """
        Extract mouth region from frame using relative coordinates.
        
        Args:
            frame: RGB frame
            relative_bounds: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
        
        Returns:
            Cropped mouth region
        """
        h, w = frame.shape[:2]
        x1 = int(w * relative_bounds[0])
        y1 = int(h * relative_bounds[1])
        x2 = int(w * relative_bounds[2])
        y2 = int(h * relative_bounds[3])
        
        return frame[y1:y2, x1:x2]
    
    def align_frames(self,
                    gen_frames: List[np.ndarray],
                    gt_frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align frame lists to same length and resize if needed.
        """
        min_len = min(len(gen_frames), len(gt_frames))
        gen_frames = gen_frames[:min_len]
        gt_frames = gt_frames[:min_len]
        
        # Resize generated to match ground truth
        if gen_frames[0].shape != gt_frames[0].shape:
            target_h, target_w = gt_frames[0].shape[:2]
            gen_frames = [cv2.resize(f, (target_w, target_h)) for f in gen_frames]
        
        return gen_frames, gt_frames
    
    def compute_ssim(self, 
                    img1: np.ndarray, 
                    img2: np.ndarray) -> float:
        """Compute SSIM between two images."""
        # Ensure same size
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # Compute SSIM
        if len(img1.shape) == 3:
            # Multichannel
            return ssim(img1, img2, channel_axis=2, data_range=255)
        else:
            return ssim(img1, img2, data_range=255)
    
    def compute_psnr(self, 
                    img1: np.ndarray, 
                    img2: np.ndarray) -> float:
        """Compute PSNR between two images."""
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        return psnr(img1, img2, data_range=255)
    
    def compute_mse(self, 
                   img1: np.ndarray, 
                   img2: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def compare(self,
               generated_video: str,
               ground_truth_video: str,
               max_frames: Optional[int] = None) -> Dict:
        """
        Compare generated video to ground truth.
        
        Args:
            generated_video: Path to generated video
            ground_truth_video: Path to ground truth video
            max_frames: Max frames to compare (None for all)
        
        Returns:
            Dictionary with comparison metrics
        """
        print(f"Loading generated video: {generated_video}")
        gen_frames, gen_fps = self.extract_frames(generated_video, max_frames)
        
        print(f"Loading ground truth video: {ground_truth_video}")
        gt_frames, gt_fps = self.extract_frames(ground_truth_video, max_frames)
        
        # Align frames
        gen_frames, gt_frames = self.align_frames(gen_frames, gt_frames)
        n_frames = len(gen_frames)
        print(f"Comparing {n_frames} frames...")
        
        # Compute metrics
        ssim_scores = []
        psnr_scores = []
        mse_scores = []
        
        ssim_mouth_scores = []
        psnr_mouth_scores = []
        mse_mouth_scores = []
        
        for i, (gen, gt) in enumerate(zip(gen_frames, gt_frames)):
            # Full frame metrics
            ssim_scores.append(self.compute_ssim(gen, gt))
            psnr_scores.append(self.compute_psnr(gen, gt))
            mse_scores.append(self.compute_mse(gen, gt))
            
            # Mouth region metrics
            if self.mouth_region_only:
                gen_mouth = self.extract_mouth_region(gen)
                gt_mouth = self.extract_mouth_region(gt)
                
                ssim_mouth_scores.append(self.compute_ssim(gen_mouth, gt_mouth))
                psnr_mouth_scores.append(self.compute_psnr(gen_mouth, gt_mouth))
                mse_mouth_scores.append(self.compute_mse(gen_mouth, gt_mouth))
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_frames} frames...")
        
        results = {
            'n_frames': n_frames,
            'gen_fps': gen_fps,
            'gt_fps': gt_fps,
            
            # Full frame metrics
            'full_frame': {
                'ssim_mean': float(np.mean(ssim_scores)),
                'ssim_std': float(np.std(ssim_scores)),
                'ssim_min': float(np.min(ssim_scores)),
                'psnr_mean': float(np.mean(psnr_scores)),
                'psnr_std': float(np.std(psnr_scores)),
                'mse_mean': float(np.mean(mse_scores)),
            },
        }
        
        if self.mouth_region_only:
            results['mouth_region'] = {
                'ssim_mean': float(np.mean(ssim_mouth_scores)),
                'ssim_std': float(np.std(ssim_mouth_scores)),
                'ssim_min': float(np.min(ssim_mouth_scores)),
                'psnr_mean': float(np.mean(psnr_mouth_scores)),
                'psnr_std': float(np.std(psnr_mouth_scores)),
                'mse_mean': float(np.mean(mse_mouth_scores)),
            }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "="*60)
        print("Visual Quality Comparison Results")
        print("="*60)
        print(f"Frames Compared: {results['n_frames']}")
        print(f"FPS - Generated: {results['gen_fps']:.1f}, Ground Truth: {results['gt_fps']:.1f}")
        
        print("\n--- Full Frame Metrics ---")
        ff = results['full_frame']
        print(f"  SSIM:  {ff['ssim_mean']:.4f} ± {ff['ssim_std']:.4f} (min: {ff['ssim_min']:.4f})")
        print(f"  PSNR:  {ff['psnr_mean']:.2f} ± {ff['psnr_std']:.2f} dB")
        print(f"  MSE:   {ff['mse_mean']:.2f}")
        
        if 'mouth_region' in results:
            print("\n--- Mouth Region Metrics ---")
            mr = results['mouth_region']
            print(f"  SSIM:  {mr['ssim_mean']:.4f} ± {mr['ssim_std']:.4f} (min: {mr['ssim_min']:.4f})")
            print(f"  PSNR:  {mr['psnr_mean']:.2f} ± {mr['psnr_std']:.2f} dB")
            print(f"  MSE:   {mr['mse_mean']:.2f}")
        
        print("\n--- Interpretation ---")
        ssim_val = results.get('mouth_region', results['full_frame'])['ssim_mean']
        if ssim_val >= 0.95:
            print("  Quality: Excellent (SSIM >= 0.95)")
        elif ssim_val >= 0.90:
            print("  Quality: Very Good (SSIM >= 0.90)")
        elif ssim_val >= 0.80:
            print("  Quality: Good (SSIM >= 0.80)")
        elif ssim_val >= 0.70:
            print("  Quality: Fair (SSIM >= 0.70)")
        else:
            print("  Quality: Poor (SSIM < 0.70)")
        
        print("="*60 + "\n")


def compare_videos(generated: str, ground_truth: str, max_frames: int = None) -> Dict:
    """
    Convenience function to compare two videos.
    
    Args:
        generated: Path to generated video
        ground_truth: Path to ground truth video
        max_frames: Maximum frames to compare
    
    Returns:
        Dictionary with comparison metrics
    """
    evaluator = VisualQualityEvaluator(mouth_region_only=True)
    return evaluator.compare(generated, ground_truth, max_frames)
