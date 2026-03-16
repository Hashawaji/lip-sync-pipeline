#!/usr/bin/env python3
"""
SyncNet Comparison Script

Compares generated video against ground truth using SyncNet metrics.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.syncnet import SyncNetEvaluator


def main():
    # Get base directory
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)
    
    # Paths (relative to base_dir)
    gt_video = "evaluation/ground_truth/actor_5_master_20s.mp4"
    gt_audio = "evaluation/ground_truth/actor_5_audio_20s.wav"
    gen_video = "evaluation/final_v1_first_30sec.mp4"
    
    print("="*60)
    print("SyncNet Comparison: Ground Truth vs Generated")
    print("="*60)
    
    # Initialize evaluator
    evaluator = SyncNetEvaluator()
    
    # Evaluate ground truth
    print("\n[1/2] Evaluating Ground Truth (RealTalk master)...")
    gt_results = evaluator.evaluate(gt_video, gt_audio)
    
    # Evaluate generated
    print("\n[2/2] Evaluating Generated (Triphone concatenation)...")
    gen_results = evaluator.evaluate(gen_video)
    
    # Compile comparison
    comparison = {
        'ground_truth': {
            'video': gt_video,
            'audio': gt_audio,
            **gt_results
        },
        'generated': {
            'video': gen_video,
            'audio': '(embedded)',
            **gen_results
        },
        'summary': {
            'lse_d_diff': gen_results['lse_d'] - gt_results['lse_d'],
            'lse_c_diff': gen_results['lse_c'] - gt_results['lse_c'],
            'winner': 'ground_truth' if gt_results['lse_d'] < gen_results['lse_d'] else 'generated',
        }
    }
    
    # Save results
    output_path = "evaluation/syncnet_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'Ground Truth':>15} {'Generated':>15} {'Diff':>10}")
    print("-"*60)
    print(f"{'LSE-D (lower=better)':<20} {gt_results['lse_d']:>15.4f} {gen_results['lse_d']:>15.4f} {comparison['summary']['lse_d_diff']:>+10.4f}")
    print(f"{'LSE-C (higher=better)':<20} {gt_results['lse_c']:>15.4f} {gen_results['lse_c']:>15.4f} {comparison['summary']['lse_c_diff']:>+10.4f}")
    print(f"{'Offset (ms)':<20} {gt_results['offset_ms']:>15.1f} {gen_results['offset_ms']:>15.1f}")
    print("-"*60)
    print(f"Winner: {comparison['summary']['winner'].upper()}")
    print("="*60)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
