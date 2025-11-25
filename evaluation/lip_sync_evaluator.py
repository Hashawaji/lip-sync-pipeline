#!/usr/bin/env python3
"""
Lip Sync Quality Evaluator

Provides multiple metrics for evaluating lip sync accuracy:
1. Mouth Movement Analysis (LSE-style)
2. Temporal Consistency
3. Audio-Visual Correlation
4. Perceptual Quality Metrics

Designed for comparing video generators objectively.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.signal
import scipy.fft


class LipSyncEvaluator:
    """Comprehensive lip sync quality evaluation."""
    
    def __init__(self, use_face_detection=True):
        """
        Initialize evaluator.
        
        Args:
            use_face_detection: Whether to use face detection for mouth region (requires dlib)
        """
        self.use_face_detection = use_face_detection
        self.detector = None
        self.predictor = None
        
        if use_face_detection:
            try:
                import dlib
                self.detector = dlib.get_frontal_face_detector()
                
                # Try to load facial landmark predictor
                predictor_path = Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"
                if predictor_path.exists():
                    self.predictor = dlib.shape_predictor(str(predictor_path))
                    print("✓ Facial landmark detector loaded")
                else:
                    print("⚠ Face landmark detector not found. Using simplified analysis.")
                    self.use_face_detection = False
            except ImportError:
                print("⚠ dlib not installed. Using simplified analysis without face detection.")
                self.use_face_detection = False
    
    def extract_mouth_region_simple(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract mouth region using simple heuristics (no face detection).
        Assumes face is centered in frame.
        """
        h, w = frame.shape[:2]
        
        # Assume mouth is in lower-middle area
        mouth_y_start = int(h * 0.6)
        mouth_y_end = int(h * 0.85)
        mouth_x_start = int(w * 0.3)
        mouth_x_end = int(w * 0.7)
        
        mouth_region = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        return mouth_region
    
    def extract_mouth_region_with_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth region using facial landmarks (requires dlib)."""
        if not self.use_face_detection or self.predictor is None:
            return self.extract_mouth_region_simple(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return self.extract_mouth_region_simple(frame)
        
        # Use first detected face
        landmarks = self.predictor(gray, faces[0])
        
        # Extract mouth region (landmarks 48-68)
        mouth_points = np.array([
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in range(48, 68)
        ])
        
        # Get bounding box
        x_min = np.min(mouth_points[:, 0])
        x_max = np.max(mouth_points[:, 0])
        y_min = np.min(mouth_points[:, 1])
        y_max = np.max(mouth_points[:, 1])
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        mouth_region = frame[y_min:y_max, x_min:x_max]
        return mouth_region
    
    def calculate_mouth_openness(self, mouth_region: np.ndarray) -> float:
        """
        Calculate mouth openness score from mouth region.
        Uses edge detection and contour analysis.
        """
        if mouth_region is None or mouth_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY) if len(mouth_region.shape) == 3 else mouth_region
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate openness as ratio of edge pixels (simplified metric)
        openness = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return openness
    
    def analyze_mouth_movements(self, video_path: str) -> Dict:
        """
        Analyze mouth movements throughout video.
        
        Returns:
            Dictionary with movement statistics and frame-by-frame data
        """
        print(f"\n=== Analyzing Mouth Movements ===")
        print(f"Video: {video_path}")
        
        if not os.path.exists(video_path):
            return {'error': f'Video not found: {video_path}'}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Could not open video: {video_path}'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {duration:.2f}s")
        
        mouth_openness_values = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract mouth region
            if self.use_face_detection:
                mouth_region = self.extract_mouth_region_with_landmarks(frame)
            else:
                mouth_region = self.extract_mouth_region_simple(frame)
            
            # Calculate openness
            openness = self.calculate_mouth_openness(mouth_region)
            mouth_openness_values.append(openness)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{frame_count} frames...")
        
        cap.release()
        
        if len(mouth_openness_values) == 0:
            return {'error': 'No frames processed'}
        
        # Calculate statistics
        openness_array = np.array(mouth_openness_values)
        
        # Calculate movement velocity (frame-to-frame changes)
        velocities = np.abs(np.diff(openness_array))
        
        # Calculate acceleration (velocity changes)
        accelerations = np.abs(np.diff(velocities))
        
        results = {
            'frame_count': len(mouth_openness_values),
            'fps': fps,
            'duration': duration,
            'mouth_openness': {
                'mean': float(np.mean(openness_array)),
                'std': float(np.std(openness_array)),
                'min': float(np.min(openness_array)),
                'max': float(np.max(openness_array)),
                'range': float(np.max(openness_array) - np.min(openness_array)),
            },
            'movement_velocity': {
                'mean': float(np.mean(velocities)),
                'std': float(np.std(velocities)),
                'max': float(np.max(velocities)),
            },
            'movement_acceleration': {
                'mean': float(np.mean(accelerations)),
                'std': float(np.std(accelerations)),
                'max': float(np.max(accelerations)),
            },
            'raw_openness_values': mouth_openness_values[:1000]  # Store first 1000 for plotting
        }
        
        print(f"\n  ✓ Mouth Movement Analysis Complete")
        print(f"    Openness Range: {results['mouth_openness']['min']:.4f} - {results['mouth_openness']['max']:.4f}")
        print(f"    Average Movement: {results['mouth_openness']['mean']:.4f} ± {results['mouth_openness']['std']:.4f}")
        print(f"    Movement Velocity: {results['movement_velocity']['mean']:.4f} (max: {results['movement_velocity']['max']:.4f})")
        
        return results
    
    def analyze_temporal_consistency(self, video_path: str) -> Dict:
        """
        Analyze temporal consistency (smoothness) of video.
        Detects jitter, flickering, and unnatural transitions.
        """
        print(f"\n=== Analyzing Temporal Consistency ===")
        
        if not os.path.exists(video_path):
            return {'error': f'Video not found: {video_path}'}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Could not open video: {video_path}'}
        
        prev_frame = None
        frame_diffs = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                frame_diffs.append(diff)
            
            prev_frame = frame.copy()
            frame_idx += 1
        
        cap.release()
        
        if len(frame_diffs) == 0:
            return {'error': 'No frame differences calculated'}
        
        diffs_array = np.array(frame_diffs)
        
        # Calculate temporal smoothness metrics
        results = {
            'frame_difference': {
                'mean': float(np.mean(diffs_array)),
                'std': float(np.std(diffs_array)),
                'min': float(np.min(diffs_array)),
                'max': float(np.max(diffs_array)),
            },
            'temporal_smoothness_score': float(1.0 / (1.0 + np.std(diffs_array))),  # Higher = smoother
            'jitter_score': float(np.std(diffs_array)),  # Lower = less jitter
        }
        
        print(f"  ✓ Temporal Analysis Complete")
        print(f"    Smoothness Score: {results['temporal_smoothness_score']:.4f} (higher is better)")
        print(f"    Jitter Score: {results['jitter_score']:.2f} (lower is better)")
        
        return results
    
    def analyze_audio_visual_correlation(self, video_path: str, audio_path: Optional[str] = None) -> Dict:
        """
        Analyze correlation between audio energy and visual movement.
        Higher correlation suggests better lip sync.
        """
        print(f"\n=== Analyzing Audio-Visual Correlation ===")
        
        if not os.path.exists(video_path):
            return {'error': f'Video not found: {video_path}'}
        
        # Extract audio from video if not provided separately
        if audio_path is None:
            audio_path = video_path
        
        # First, get visual movement signal
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        mouth_movements = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            mouth_region = self.extract_mouth_region_simple(frame)
            openness = self.calculate_mouth_openness(mouth_region)
            mouth_movements.append(openness)
        
        cap.release()
        
        if len(mouth_movements) == 0:
            return {'error': 'No visual movement extracted'}
        
        # Try to extract audio using ffmpeg
        try:
            import tempfile
            import subprocess
            
            # Extract audio to temporary WAV file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', temp_audio.name
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Load audio
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(temp_audio.name)
            
            # Clean up
            os.unlink(temp_audio.name)
            
            # Calculate audio energy envelope
            audio_energy = np.abs(audio_data.astype(float))
            
            # Downsample to match video frame rate
            samples_per_frame = len(audio_energy) // len(mouth_movements)
            audio_envelope = []
            
            for i in range(len(mouth_movements)):
                start_idx = i * samples_per_frame
                end_idx = min((i + 1) * samples_per_frame, len(audio_energy))
                frame_energy = np.mean(audio_energy[start_idx:end_idx])
                audio_envelope.append(frame_energy)
            
            # Normalize both signals
            visual_signal = np.array(mouth_movements)
            audio_signal = np.array(audio_envelope[:len(visual_signal)])
            
            visual_signal = (visual_signal - np.mean(visual_signal)) / (np.std(visual_signal) + 1e-10)
            audio_signal = (audio_signal - np.mean(audio_signal)) / (np.std(audio_signal) + 1e-10)
            
            # Calculate correlation
            correlation = np.corrcoef(visual_signal, audio_signal)[0, 1]
            
            results = {
                'correlation': float(correlation),
                'visual_frames': len(mouth_movements),
                'audio_samples': len(audio_data),
                'sync_quality': 'Good' if correlation > 0.5 else 'Fair' if correlation > 0.3 else 'Poor'
            }
            
            print(f"  ✓ Audio-Visual Correlation: {correlation:.4f}")
            print(f"  Quality: {results['sync_quality']}")
            
            return results
            
        except Exception as e:
            print(f"  ⚠ Could not extract audio: {e}")
            return {'error': f'Audio extraction failed: {str(e)}'}
    
    def generate_comprehensive_report(self, video_path: str, audio_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive lip sync quality report.
        
        Args:
            video_path: Path to video file
            audio_path: Optional separate audio file
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "=" * 70)
        print("LIP SYNC QUALITY EVALUATION")
        print("=" * 70)
        print(f"Video: {video_path}")
        print("=" * 70)
        
        report = {
            'video_path': video_path,
            'mouth_movement_analysis': self.analyze_mouth_movements(video_path),
            'temporal_consistency': self.analyze_temporal_consistency(video_path),
            'audio_visual_correlation': self.analyze_audio_visual_correlation(video_path, audio_path),
        }
        
        # Calculate overall quality score (0-100)
        score_components = []
        
        # Movement quality (30 points)
        mouth_analysis = report['mouth_movement_analysis']
        if 'error' not in mouth_analysis:
            movement_range = mouth_analysis['mouth_openness']['range']
            movement_score = min(30, movement_range * 1000)  # Scale appropriately
            score_components.append(('Movement Quality', movement_score, 30))
        
        # Temporal smoothness (30 points)
        temporal = report['temporal_consistency']
        if 'error' not in temporal:
            smoothness = temporal['temporal_smoothness_score']
            smoothness_score = smoothness * 30
            score_components.append(('Temporal Smoothness', smoothness_score, 30))
        
        # Audio-visual sync (40 points)
        av_corr = report['audio_visual_correlation']
        if 'error' not in av_corr and 'correlation' in av_corr:
            correlation = av_corr['correlation']
            # Normalize correlation from [-1, 1] to [0, 40]
            sync_score = max(0, (correlation + 1) / 2 * 40)
            score_components.append(('Audio-Visual Sync', sync_score, 40))
        
        # Calculate total score
        total_score = sum(score for _, score, _ in score_components)
        max_score = sum(max_val for _, _, max_val in score_components)
        
        report['quality_score'] = {
            'total': float(total_score),
            'max_possible': float(max_score),
            'percentage': float(total_score / max_score * 100) if max_score > 0 else 0,
            'components': score_components,
            'grade': self._get_grade(total_score / max_score * 100) if max_score > 0 else 'N/A'
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("QUALITY SCORE SUMMARY")
        print("=" * 70)
        for component_name, score, max_val in score_components:
            print(f"  {component_name:.<50} {score:.1f}/{max_val}")
        print("-" * 70)
        print(f"  {'TOTAL SCORE':.<50} {total_score:.1f}/{max_score} ({total_score/max_score*100:.1f}%)")
        print(f"  {'GRADE':.<50} {report['quality_score']['grade']}")
        print("=" * 70)
        
        return report
    
    def _get_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade."""
        if percentage >= 90:
            return 'A (Excellent)'
        elif percentage >= 80:
            return 'B (Good)'
        elif percentage >= 70:
            return 'C (Fair)'
        elif percentage >= 60:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def save_report(self, report: Dict, output_path: str):
        """Save evaluation report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to: {output_path}")


def main():
    """CLI for lip sync evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate lip sync quality of video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--audio', help='Optional separate audio file')
    parser.add_argument('--output', help='Output JSON path for report')
    parser.add_argument('--no-face-detection', action='store_true', 
                       help='Disable face detection (faster but less accurate)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = LipSyncEvaluator(use_face_detection=not args.no_face_detection)
    
    # Generate report
    report = evaluator.generate_comprehensive_report(args.video, args.audio)
    
    # Save report if requested
    if args.output:
        evaluator.save_report(report, args.output)
    
    return report


if __name__ == '__main__':
    main()
