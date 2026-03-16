#!/usr/bin/env python3
"""Final SyncNet comparison: Ground Truth vs Generated (timing-fixed)"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.syncnet import SyncNetEvaluator

def main():
    e = SyncNetEvaluator()

    print('=== GROUND TRUTH ===')
    gt = e.evaluate('evaluation/ground_truth/actor_5_master_20s.mp4', 'evaluation/ground_truth/actor_5_audio_20s.wav')
    
    print()
    print('=== NEW GENERATED (timing-fixed) ===')
    new = e.evaluate('outputs/actor_5_duration_fixed.mp4')
    
    print()
    print('=' * 60)
    print('FINAL COMPARISON: Ground Truth vs Generated (timing-fixed)')
    print('=' * 60)
    print(f'{"Metric":<25} {"Ground Truth":>12} {"Generated":>12} {"Winner":>10}')
    print('-' * 60)
    
    # LSE-D (lower is better)
    lse_d_winner = "GT" if gt["lse_d"] < new["lse_d"] else "GEN"
    print(f'{"LSE-D (lower=better)":<25} {gt["lse_d"]:>12.4f} {new["lse_d"]:>12.4f} {lse_d_winner:>10}')
    
    # LSE-C (higher is better)
    lse_c_winner = "GT" if gt["lse_c"] > new["lse_c"] else "GEN"
    print(f'{"LSE-C (higher=better)":<25} {gt["lse_c"]:>12.4f} {new["lse_c"]:>12.4f} {lse_c_winner:>10}')
    
    # Offset (0 is best)
    gt_offset = gt.get("offset_frames", 0)
    new_offset = new.get("offset_frames", 0)
    offset_winner = "GT" if abs(gt_offset) < abs(new_offset) else "GEN"
    print(f'{"Offset (frames)":<25} {gt_offset:>12} {new_offset:>12} {offset_winner:>10}')
    
    print('-' * 60)
    
    # Summary
    print()
    v1_offset = -57  # From earlier test
    print('IMPROVEMENT vs OLD Generated:')
    print(f'  Offset: {v1_offset} frames -> {new_offset} frames ({abs(v1_offset) - abs(new_offset)} frames better)')
    print(f'  This reduced A/V desync from {abs(v1_offset)*40}ms to {abs(new_offset)*40}ms')

if __name__ == '__main__':
    main()
