#!/usr/bin/env python3
"""
Lip-Sync Evaluation Runner

Usage:
    # Simple mode (no pretrained weights needed)
    python run_syncnet_eval.py --video path/to/video.mp4 --mode simple
    
    # SyncNet mode (requires pretrained weights)
    python run_syncnet_eval.py --video video.mp4 --mode syncnet
    
    # Compare multiple videos
    python run_syncnet_eval.py --compare video1.mp4 video2.mp4 video3.mp4 --mode simple
"""

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
    
    parser.add_argument('--mode', '-m', default='simple', choices=['simple', 'syncnet'],
                       help='Evaluation mode: simple (no weights) or syncnet (needs model)')
    parser.add_argument('--audio', '-a', help='Optional separate audio file')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--device', '-d', default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device for SyncNet mode (default: auto)')
    parser.add_argument('--labels', '-l', nargs='+', help='Labels for comparison videos')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    print(f"Initializing {args.mode} evaluator...")
    
    if args.mode == 'simple':
        evaluator = SimpleAVEvaluator()
    else:
        evaluator = SyncNetEvaluator(device=args.device)
    
    if args.video:
        # Single video evaluation
        if args.mode == 'simple':
            results = evaluator.evaluate(args.video)
        else:
            results = evaluator.evaluate(args.video, args.audio)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")
    
    elif args.compare:
        # Multi-video comparison
        labels = args.labels if args.labels else None
        results = evaluator.compare_videos(args.compare, labels)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Comparison results saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
