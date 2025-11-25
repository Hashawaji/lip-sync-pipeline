#!/usr/bin/env python3
"""
BlinkScheduler - Natural Blink Timing with Pause-Snapping
==========================================================

Generates natural blink schedules based on speech pauses and 
physiological inter-blink intervals (IBI).

Algorithm:
1. Find all speech pauses (sil, sp) from phoneme data
2. Generate baseline blinks based on natural IBI (2.5-7.0 seconds)
3. "Snap" baseline blinks to nearby pauses when available
4. Enforce minimum IBI to prevent clustering
"""

import random
import json
from typing import List, Tuple, Set, Dict, Any, Optional
from pathlib import Path


class BlinkScheduler:
    """
    Generates a natural blink schedule based on speech pauses and a 
    physiological inter-blink interval (IBI).
    
    This uses a "pause-snapping" algorithm:
    1. A baseline IBI is used to schedule blinks.
    2. These scheduled blinks are "snapped" to the closest speech pause
       (from phonemes) if one is within a small time window.
    """
    
    def __init__(self, phoneme_sequence: List[Tuple[str, float, float]], 
                 fps: float, 
                 total_duration_sec: float,
                 min_ibi: float = 2.5, 
                 max_ibi: float = 7.0, 
                 pause_phonemes: Optional[List[str]] = None,
                 pause_snap_window: float = 0.5, 
                 pause_min_duration: float = 0.15):
        """
        Initializes the blink scheduler.

        Args:
            phoneme_sequence: A list of (phoneme, start_sec, end_sec) tuples.
            fps: The video's frames per second.
            total_duration_sec: The total duration of the video.
            min_ibi: The shortest-allowed time (in seconds) between blinks.
            max_ibi: The longest-allowed time (in seconds) between blinks.
            pause_phonemes: List of strings identifying pause phonemes.
            pause_snap_window: Time (in seconds) to look for a nearby pause.
            pause_min_duration: Pauses shorter than this (in sec) are ignored.
        """
        if pause_phonemes is None:
            pause_phonemes = ['sil', 'sp']
        
        self.phoneme_sequence = phoneme_sequence
        self.fps = fps
        self.total_frames = int(total_duration_sec * fps)
        self.min_ibi_frames = int(min_ibi * fps)
        self.max_ibi_frames = int(max_ibi * fps)
        self.pause_snap_frames = int(pause_snap_window * fps)
        self.pause_min_duration = pause_min_duration
        self.pause_phonemes = set(pause_phonemes)
        
        # 1. Find all "blink opportunities" (pauses)
        # We store the *center* frame of the pause
        pause_opportunities = []
        for ph, start, end in self.phoneme_sequence:
            if ph in self.pause_phonemes and (end - start) >= self.pause_min_duration:
                center_time = (start + end) / 2.0
                pause_opportunities.append(int(center_time * self.fps))
        
        # Use a set for quick O(1) "used" checks and removals
        self.available_pauses = set(pause_opportunities)
        self.pause_opportunities_count = len(pause_opportunities)

    def _find_closest_available_pause(self, frame: int) -> int:
        """
        Finds the closest available pause within the snap window.
        
        Args:
            frame: The target frame number to search around.
            
        Returns:
            Frame number of closest pause, or -1 if none found.
        """
        if not self.available_pauses:
            return -1
            
        # Find the pause with the minimum distance
        best_pause = -1
        min_dist = float('inf')
        
        # Note: This is a simple linear search. For extreme-scale videos with
        # thousands of pauses, you could use binary search (bisect_left)
        # on a sorted list, but for typical speech, this is plenty fast.
        for pause_frame in self.available_pauses:
            dist = abs(frame - pause_frame)
            if dist <= self.pause_snap_frames and dist < min_dist:
                min_dist = dist
                best_pause = pause_frame
        return best_pause

    def generate_blink_frames(self) -> Tuple[Set[int], Dict[str, Any]]:
        """
        Generates a set of frame numbers on which a blink should *start*.
        
        Returns:
            Tuple of:
            - Set of frame numbers where blinks should start
            - Statistics dictionary with scheduling information
        """
        blink_start_frames = []
        snapped_count = 0
        baseline_count = 0
        
        # Start with a random offset to avoid blinking on frame 0
        current_frame = random.randint(0, self.min_ibi_frames)
        last_blink_frame = -self.min_ibi_frames  # Allows an early blink

        while current_frame < self.total_frames:
            # 1. Get a "target" blink frame based on IBI
            ibi = random.randint(self.min_ibi_frames, self.max_ibi_frames)
            target_frame = current_frame + ibi
            
            if target_frame >= self.total_frames:
                break
                
            # 2. Try to "snap" to a nearby pause
            snapped_pause_frame = self._find_closest_available_pause(target_frame)
            
            final_blink_frame = target_frame
            is_snapped = False
            
            if snapped_pause_frame != -1:
                final_blink_frame = snapped_pause_frame
                self.available_pauses.remove(snapped_pause_frame)  # Consume the pause
                is_snapped = True

            # 3. Enforce minimum IBI to prevent blinks from clustering
            if (final_blink_frame - last_blink_frame) >= self.min_ibi_frames:
                blink_start_frames.append(final_blink_frame)
                last_blink_frame = final_blink_frame
                current_frame = final_blink_frame  # Next blink is relative to this one
                
                if is_snapped:
                    snapped_count += 1
                else:
                    baseline_count += 1
            else:
                # This blink was too close to the last one, skip it.
                # Schedule the *next* blink from the *original* target.
                current_frame = target_frame 

        # Calculate statistics
        stats = {
            'total_blinks': len(blink_start_frames),
            'snapped_to_pause': snapped_count,
            'baseline_blinks': baseline_count,
            'total_pauses_available': self.pause_opportunities_count,
            'pauses_used': snapped_count,
            'pauses_unused': len(self.available_pauses),
            'video_duration_sec': self.total_frames / self.fps,
            'average_ibi_sec': (self.total_frames / self.fps) / len(blink_start_frames) if blink_start_frames else 0
        }

        return set(blink_start_frames), stats
    
    @staticmethod
    def from_phoneme_json(json_path: Path, fps: float, total_duration_sec: float, **kwargs) -> 'BlinkScheduler':
        """
        Factory method to create a BlinkScheduler from a phoneme JSON file.
        
        This method handles the MFA v3 comprehensive format with multiple phoneme formats.
        
        Args:
            json_path: Path to the phoneme JSON file (complete_phoneme_alignments.json format)
            fps: Video frames per second
            total_duration_sec: Total video duration in seconds
            **kwargs: Additional arguments to pass to BlinkScheduler constructor
            
        Returns:
            BlinkScheduler instance
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            KeyError: If JSON format is unexpected
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Phoneme JSON not found: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract phoneme sequence from MFA v3 format
        # The format has: data['files']['audio']['arpabet_phonemes'] or ['ipa_phonemes'] or ['viseme_phonemes']
        
        if 'files' not in data:
            raise KeyError(f"'files' key not found in phoneme JSON. Available keys: {list(data.keys())}")
        
        # Get the first file entry (usually 'audio')
        file_key = list(data['files'].keys())[0]
        file_data = data['files'][file_key]
        
        # Try to get phoneme data in priority order: ARPABET > IPA > VISEME
        phoneme_data = None
        if 'arpabet_phonemes' in file_data:
            phoneme_data = file_data['arpabet_phonemes']
        elif 'ipa_phonemes' in file_data:
            phoneme_data = file_data['ipa_phonemes']
        elif 'viseme_phonemes' in file_data:
            phoneme_data = file_data['viseme_phonemes']
        else:
            raise KeyError(f"No phoneme data found. Available keys: {list(file_data.keys())}")
        
        # Convert to (phoneme, start, end) tuples
        phoneme_sequence = []
        for item in phoneme_data:
            phoneme = item.get('phoneme', '')
            start = item.get('start_s', 0.0)
            end = item.get('end_s', 0.0)
            phoneme_sequence.append((phoneme, start, end))
        
        # Create and return BlinkScheduler
        return BlinkScheduler(
            phoneme_sequence=phoneme_sequence,
            fps=fps,
            total_duration_sec=total_duration_sec,
            **kwargs
        )


def main():
    """Example usage of BlinkScheduler"""
    
    # Example phoneme data
    phoneme_data = [
        ('sil', 0.0, 0.22), ('W', 0.22, 0.28), ('IY', 0.28, 0.34), ('sil', 0.34, 0.5), 
        ('AA', 0.5, 0.59), ('R', 0.59, 0.64), ('sp', 0.64, 0.7), ('G', 0.7, 0.75), 
        ('OW', 0.75, 0.86), ('IH', 0.86, 0.92), ('NG', 0.92, 1.08), 
        ('sil', 1.08, 1.5), ('T', 1.5, 1.56), ('UW', 1.56, 1.62), ('sil', 1.62, 2.5)
    ]
    
    video_fps = 25.0
    video_duration = 2.5  # seconds
    
    # Create scheduler
    scheduler = BlinkScheduler(
        phoneme_sequence=phoneme_data,
        fps=video_fps,
        total_duration_sec=video_duration
    )
    
    # Generate blink schedule
    blink_frames, stats = scheduler.generate_blink_frames()
    
    # Display results
    print("=" * 60)
    print("Blink Schedule Generated")
    print("=" * 60)
    print(f"Scheduled blinks at frames: {sorted(list(blink_frames))}")
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
