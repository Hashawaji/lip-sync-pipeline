#!/usr/bin/env python3
"""
Lip-Sync Evaluation Runner

Supports two modes:
1. SyncNet evaluation: Audio-visual synchronization metrics (LSE-D, LSE-C)
2. Visual quality comparison: SSIM, PSNR between generated and ground truth

Usage:
    # SyncNet evaluation (measures A/V sync quality)
    python run_syncnet_eval.py --video path/to/video.mp4 --mode syncnet
    
    # Simple A/V correlation (no pretrained weights needed)
    python run_syncnet_eval.py --video path/to/video.mp4 --mode simple
    
    # Visual quality comparison (generated vs ground truth)
    python run_syncnet_eval.py --compare generated.mp4 ground_truth.mp4 --mode visual
    
    # Compare multiple videos
    python run_syncnet_eval.py --compare video1.mp4 video2.mp4 video3.mp4 --mode simple
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.syncnet import SyncNetEvaluator, SimpleAVEvaluator


def main():
    parser = argparse.ArgumentParser(description='Lip-Sync Evaluation')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', '-v', help='Single video to evaluate')
    group.add_argument('--compare', '-c', nargs='+', help='Multiple videos to compare')
    
    parser.add_argument('--mode', '-m', default='syncnet', 
                       choices=['simple', 'syncnet', 'visual'],
                       help='Evaluation mode: simple (A/V correlation), syncnet (needs model), or visual (SSIM/PSNR)')
    parser.add_argument('--audio', '-a', help='Optional separate audio file')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--device', '-d', default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device for SyncNet mode (default: auto)')
    parser.add_argument('--labels', '-l', nargs='+', help='Labels for comparison videos')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Initialize evaluator based on mode
    print(f"Initializing {args.mode} evaluator...")
    
    if args.mode == 'simple':
        evaluator = SimpleAVEvaluator()
    elif args.mode == 'syncnet':
        evaluator = SyncNetEvaluator(device=args.device)
    elif args.mode == 'visual':
        from evaluation.visual_quality import VisualQualityEvaluator
        evaluator = VisualQualityEvaluator(mouth_region_only=True)
    
    if args.video:
        # Single video evaluation
        if args.mode == 'visual':
            print("Error: Visual mode requires --compare with generated and ground truth videos")
            return 1
        
        if args.mode == 'simple':
            results = evaluator.evaluate(args.video)
        else:
            results = evaluator.evaluate(args.video, args.audio)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")
    
    elif args.compare:
        if args.mode == 'visual':
            # Visual comparison: first video is generated, second is ground truth
            if len(args.compare) != 2:
                print("Error: Visual mode requires exactly 2 videos: --compare generated.mp4 ground_truth.mp4")
                return 1
            
            results = evaluator.compare(args.compare[0], args.compare[1], args.max_frames)
        else:
            # Multi-video A/V sync comparison
            labels = args.labels if args.labels else None
            results = evaluator.compare_videos(args.compare, labels)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Comparison results saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
