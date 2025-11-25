#!/usr/bin/env python3
"""
Quick evaluation runner for testing a specific video.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import evaluator
sys.path.insert(0, str(Path(__file__).parent))

from lip_sync_evaluator import LipSyncEvaluator


def main():
    """Evaluate the output video."""
    
    video_path = "/Users/admin/thesis/lip-sync-pipeline/evaluation/istockphoto-2018929375-640_adpp_is.mp4"
    output_json = "/Users/admin/thesis/lip-sync-pipeline/evaluation/evaluation_report.json"
    
    print("=" * 70)
    print("EVALUATING YOUR LIP SYNC VIDEO")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Report will be saved to: {output_json}")
    print("=" * 70)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"\n❌ ERROR: Video not found at {video_path}")
        return 1
    
    # Create evaluator (without face detection for speed)
    print("\nInitializing evaluator...")
    evaluator = LipSyncEvaluator(use_face_detection=False)
    
    # Run comprehensive evaluation
    report = evaluator.generate_comprehensive_report(video_path)
    
    # Save report
    evaluator.save_report(report, output_json)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Full report saved to: {output_json}")
    print(f"✓ Overall Quality: {report['quality_score']['percentage']:.1f}%")
    print(f"✓ Grade: {report['quality_score']['grade']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
