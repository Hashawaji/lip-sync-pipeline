#!/usr/bin/env python3
"""Run SyncNet comparison on final 60fps videos"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.syncnet import SyncNetEvaluator

def main():
    evaluator = SyncNetEvaluator()
    
    gt_path = os.path.expanduser('~/Desktop/lip_sync_comparison/1_GROUND_TRUTH.mp4')
    gen_path = os.path.expanduser('~/Desktop/lip_sync_comparison/2_GENERATED_60fps.mp4')
    
    print('=== GROUND TRUTH (60fps) ===')
    gt = evaluator.evaluate(gt_path)
    
    print()
    print('=== GENERATED (60fps) ===')
    gen = evaluator.evaluate(gen_path)
    
    # Create comparison
    results = {
        'ground_truth': {
            'file': '1_GROUND_TRUTH.mp4',
            'fps': 60,
            'lse_d': gt['lse_d'],
            'lse_c': gt['lse_c'],
            'offset_frames': gt.get('offset_frames', 0),
            'quality': gt.get('quality_rating', 'N/A')
        },
        'generated': {
            'file': '2_GENERATED_60fps.mp4',
            'fps': 60,
            'lse_d': gen['lse_d'],
            'lse_c': gen['lse_c'],
            'offset_frames': gen.get('offset_frames', 0),
            'quality': gen.get('quality_rating', 'N/A')
        },
        'comparison': {
            'lse_d_diff': gen['lse_d'] - gt['lse_d'],
            'lse_c_diff': gen['lse_c'] - gt['lse_c'],
            'lse_d_winner': 'GROUND_TRUTH' if gt['lse_d'] < gen['lse_d'] else 'GENERATED',
            'lse_c_winner': 'GROUND_TRUTH' if gt['lse_c'] > gen['lse_c'] else 'GENERATED'
        }
    }
    
    # Save to comparison folder
    output_path = os.path.expanduser('~/Desktop/lip_sync_comparison/syncnet_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to: {output_path}')
    
    # Print summary
    print('\n' + '='*60)
    print('SYNCNET COMPARISON RESULTS (60fps)')
    print('='*60)
    print(f'{"Metric":<25} {"Ground Truth":>12} {"Generated":>12} {"Winner":>12}')
    print('-'*60)
    print(f'{"LSE-D (lower=better)":<25} {gt["lse_d"]:>12.4f} {gen["lse_d"]:>12.4f} {results["comparison"]["lse_d_winner"]:>12}')
    print(f'{"LSE-C (higher=better)":<25} {gt["lse_c"]:>12.4f} {gen["lse_c"]:>12.4f} {results["comparison"]["lse_c_winner"]:>12}')
    print(f'{"Offset (frames)":<25} {gt.get("offset_frames", 0):>12} {gen.get("offset_frames", 0):>12}')
    print('='*60)

if __name__ == '__main__':
    main()
