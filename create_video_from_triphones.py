#!/usr/bin/env python3
"""
Create video from triphone visemes with intelligent concatenation.
This script takes inspiration from sync.py but works with our triphone directory structure.

SMOOTHNESS IMPROVEMENTS:
========================
1. Smooth Transitions: Blends the first frame of each phoneme with the last frame 
   of the previous phoneme for seamless transitions WITHOUT adding extra frames
   (maintains perfect audio sync) (--smooth-transitions flag)

2. Motion Interpolation: Uses FFmpeg's minterpolate filter with motion-compensated 
   frame interpolation for ultra-smooth playback (--motion-interpolation flag)

3. Frame Interpolation: Optionally triples FPS by adding 2 interpolated frames between 
   each existing frame (25 FPS → 75 FPS, 30 FPS → 90 FPS) for ultra-smooth playback
   (--interpolate-frames flag)

4. Better Codec: Uses H.264 (libx264) with high quality settings (CRF 18) 
   instead of basic mp4v codec

5. Smooth Frame Scaling: When scaling frames, uses cubic easing interpolation 
   instead of linear for more natural transitions

6. Motion Blur: Optional subtle motion blur for transition frames 
   (currently disabled to maintain timing accuracy)

USAGE EXAMPLES:
==============
# Ultra smooth (default settings)
python create_video_from_triphones.py --fps 30

# Maximum smoothness (frame interpolation to triple FPS)
python create_video_from_triphones.py --fps 25 --interpolate-frames

# Ultra smooth (90 FPS with motion interpolation)
python create_video_from_triphones.py --fps 30 --interpolate-frames --motion-interpolation

# Faster processing (disable motion interpolation)
python create_video_from_triphones.py --fps 30 --no-motion-interpolation

# Disable transition frames for comparison
python create_video_from_triphones.py --fps 30 --no-smooth-transitions
"""

import cv2
import os
import numpy as np
import json
import argparse
import subprocess
import shutil
import traceback
import random
from pathlib import Path

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("Error: MoviePy is required for smooth video output. Please install it with:")
    print("pip install moviepy")
    exit(1)


# Global list of common phonemes for random mapping
COMMON_PHONEMES = [
    'sil'
]


def map_spn_to_random_phoneme():
    """Map 'spn' (spoken noise) to a random common phoneme."""
    return random.choice(COMMON_PHONEMES)


def load_enriched_phoneme_data_from_json(json_file_path, file_key="master_audio"):
    """Load enriched phoneme sequence from the JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Use enriched_sequence instead of viseme_phonemes
    enriched_sequence = data['files'][file_key]['enriched_sequence']
    
    print(f"Loaded {len(enriched_sequence)} enriched phoneme entries from {json_file_path}")
    return enriched_sequence


def get_phoneme_frame_count(phoneme_entry):
    """
    Determine frame count based on phoneme entry.
    Regular phoneme: 3 frames
    Single underscore (_): 1 frame  
    Double underscore (__): 2 frames
    """
    phoneme = phoneme_entry['phoneme']
    
    if phoneme.endswith('__'):
        return 2
    elif phoneme.endswith('_'):
        return 1
    else:
        return 3


def get_base_phoneme(phoneme_entry):
    """Extract base phoneme without underscore suffixes."""
    phoneme = phoneme_entry['phoneme']
    
    if phoneme.endswith('__'):
        return phoneme[:-2]
    elif phoneme.endswith('_'):
        return phoneme[:-1]
    else:
        return phoneme

def load_triphone_frames(triphone_dir):
    """Load frames from a triphone directory."""
    if not os.path.exists(triphone_dir):
        return [], None

    # Load metadata
    metadata_path = os.path.join(triphone_dir, 'metadata.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    # Load frames
    frame_files = sorted([f for f in os.listdir(triphone_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(triphone_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)

    return frames, metadata

def get_triphone_context_enriched(enriched_sequence, index):
    """
    Get triphone context for a phoneme at given index in enriched sequence.
    Returns (left_context, current_phoneme, right_context, triphone_name)
    Creates proper triphones with 3 phonemes: left + current + right
    Now treats 'sil' as a real phoneme and maps 'spn' to random phonemes.
    """
    current_entry = enriched_sequence[index]
    current = current_entry['phoneme']
    current_base = get_base_phoneme(current_entry)
    
    # Map 'spn' to a random phoneme
    if current_base == 'spn':
        current_base = map_spn_to_random_phoneme()
        if current.endswith('__'):
            current = current_base + '__'
        elif current.endswith('_'):
            current = current_base + '_'
        else:
            current = current_base
    
    # Define garbage phonemes to skip - removed 'sil' and 'spn'
    garbage_phonemes = {'', 'LEFTOVER', 'FINAL_LEFTOVER'}
    
    # Get left context (scan backwards for different base phoneme)
    left = None
    for i in range(index - 1, -1, -1):
        prev_base = get_base_phoneme(enriched_sequence[i])
        # Map spn to random phoneme for context too
        if prev_base == 'spn':
            prev_base = map_spn_to_random_phoneme()
        if prev_base not in garbage_phonemes:
            left = prev_base
            break
    
    # Get right context (next phoneme, regardless of base)
    right = None
    for i in range(index + 1, len(enriched_sequence)):
        next_base = get_base_phoneme(enriched_sequence[i])
        # Map spn to random phoneme for context too
        if next_base == 'spn':
            next_base = map_spn_to_random_phoneme()
        if next_base not in garbage_phonemes:
            right = next_base
            break
    
    # Create triphone name only if we have all three components
    if left and right:
        triphone_name = left + current + right  # Use full current phoneme with underscores
    else:
        # Fallback to diphone or monophone
        if left:
            triphone_name = left + current  # Use full current phoneme
        elif right:
            triphone_name = current + right  # Use full current phoneme
        else:
            triphone_name = current  # Use full current phoneme
    
    return left, current, right, triphone_name


def find_best_triphone_match(triphone_visemes_dir, triphone_name, target_frame_count):
    """
    Find the best triphone viseme directory that matches the target frame count.
    Returns the directory path of the best match.
    Now uses cached directory list for better performance.
    """
    if not os.path.exists(triphone_visemes_dir):
        return None
    
    # Use cached directory list  
    triphone_dirs = get_triphone_dirs(triphone_visemes_dir)
    
    # Try exact match - triphone names are the directory names directly
    if triphone_name in triphone_dirs:
        return os.path.join(triphone_visemes_dir, triphone_name)
    
    return None


# Global cache for triphone directories
_triphone_dirs_cache = None

def get_triphone_dirs(triphone_visemes_dir):
    """Get cached list of triphone directories"""
    global _triphone_dirs_cache
    if _triphone_dirs_cache is None:
        print("Caching triphone directories...")
        _triphone_dirs_cache = [item for item in os.listdir(triphone_visemes_dir) 
                               if os.path.isdir(os.path.join(triphone_visemes_dir, item))]
        print(f"Cached {len(_triphone_dirs_cache)} triphone directories")
    return _triphone_dirs_cache


def find_closest_triphone_match(triphone_visemes_dir, triphone_name, target_frame_count, current_phoneme=None):
    """
    Find the best matching triphone viseme using simple wildcard logic.
    For diphones (missing left or right context), find any triphone containing the pattern.
    Priority:
    1. Exact match
    2. For diphones: find triphones with wildcard (*ab or ab*)
    3. Same base with different frame counts
    """
    print(f"Finding closest match for triphone: '{triphone_name}' with current phoneme '{current_phoneme}'")
    if not triphone_name or len(triphone_name) < 2:
        return None, 0
    
    # Use cached directory list
    triphone_dirs = get_triphone_dirs(triphone_visemes_dir)
    
    best_match = None
    best_score = 0
    
    print(f"  Searching for: '{triphone_name}'")
    
    # Priority 1: underscore variants with specific priority order
    if current_phoneme:
        # Get the base phoneme without underscores
        base_current = current_phoneme.rstrip('_')
        
        # Extract left and right context from triphone_name
        left_context = ""
        right_context = ""
        
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            left_context = triphone_name[:current_pos]
            right_context = triphone_name[current_pos + len(current_phoneme):]
        
        
        # Define the priority order based on current phoneme pattern
        if current_phoneme.endswith('__'):
            # Pattern: a__ -> try a, then a_
            priority_variants = [
                (base_current, "no underscore (3 frames)"),           # a (3 frames)
                (base_current + '_', "single underscore (1 frame)")   # a_ (1 frame)
            ]
        elif current_phoneme.endswith('_'):
            # Pattern: a_ -> try a__, then a
            priority_variants = [
                (base_current + '__', "double underscore (2 frames)"), # a__ (2 frames)
                (base_current, "no underscore (3 frames)")             # a (3 frames)
            ]
        else:
            # Pattern: a -> try a__, then a_
            priority_variants = [
                (base_current + '__', "double underscore (2 frames)"), # a__ (2 frames)  
                (base_current + '_', "single underscore (1 frame)")    # a_ (1 frame)
            ]
        
        # Try variants in priority order
        for variant, description in priority_variants:
            test_triphone = left_context + variant + right_context
            print(f"    Trying variant: '{test_triphone}' ({description})")
            
            for dir_name in triphone_dirs:
                if dir_name == test_triphone:
                    best_match = os.path.join(triphone_visemes_dir, dir_name)
                    best_score = 98  # High priority but not exact match
                    print(f"    Found underscore variant match: {dir_name}")
                    return best_match, best_score
    
    # Priority 2: Match left + center with any right context
    print(f"  Priority 2: Left + Center match with any right")
    if current_phoneme and len(triphone_name) >= 2:
        # Extract left and right context
        left_context = ""
        right_context = ""
        
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            left_context = triphone_name[:current_pos]
            right_context = triphone_name[current_pos + len(current_phoneme):]
        
        if left_context:
            # Create left + center pattern (without right context)
            left_center_pattern = left_context + current_phoneme
            print(f"    Looking for pattern starting with: '{left_center_pattern}' + any right")
            
            for dir_name in triphone_dirs:
                if dir_name.startswith(left_center_pattern) and len(dir_name) > len(left_center_pattern):
                    if best_score < 85:
                        best_match = os.path.join(triphone_visemes_dir, dir_name)
                        best_score = 85
                        print(f"    Found left+center match: {dir_name}")
                        return best_match, best_score
    
    # Priority 3: Match center + right with any left context  
    print(f"  Priority 3: Center + Right match with any left")
    if current_phoneme and len(triphone_name) >= 2:
        # Extract left and right context
        left_context = ""
        right_context = ""
        
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            left_context = triphone_name[:current_pos]
            right_context = triphone_name[current_pos + len(current_phoneme):]
        
        if right_context:
            # Create center + right pattern (without left context)
            center_right_pattern = current_phoneme + right_context
            print(f"    Looking for pattern ending with: any left + '{center_right_pattern}'")
            
            for dir_name in triphone_dirs:
                if dir_name.endswith(center_right_pattern) and len(dir_name) > len(center_right_pattern):
                    if best_score < 80:
                        best_match = os.path.join(triphone_visemes_dir, dir_name)
                        best_score = 80
                        print(f"    Found center+right match: {dir_name}")
                        return best_match, best_score
    
    # Priority 4: Combine Priority 2 (Left+Center) with Priority 1 (underscore variants)
    print(f"  Priority 4: Left + Center match with underscore variants")
    if current_phoneme and len(triphone_name) >= 2:
        # Get the base phoneme without underscores
        base_current = current_phoneme.rstrip('_')
        
        # Extract left context
        left_context = ""
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            left_context = triphone_name[:current_pos]
        
        if left_context:
            # Define underscore variants based on current phoneme pattern
            if current_phoneme.endswith('__'):
                # Pattern: a__ -> try a, then a_
                priority_variants = [
                    (base_current, "no underscore"),
                    (base_current + '_', "single underscore")
                ]
            elif current_phoneme.endswith('_'):
                # Pattern: a_ -> try a__, then a
                priority_variants = [
                    (base_current + '__', "double underscore"),
                    (base_current, "no underscore")
                ]
            else:
                # Pattern: a -> try a__, then a_
                priority_variants = [
                    (base_current + '__', "double underscore"),
                    (base_current + '_', "single underscore")
                ]
            
            # Try each underscore variant with left+center pattern
            for variant, description in priority_variants:
                left_center_pattern = left_context + variant
                print(f"    Trying left+center pattern: '{left_center_pattern}*' ({description})")
                
                for dir_name in triphone_dirs:
                    if dir_name.startswith(left_center_pattern) and len(dir_name) > len(left_center_pattern):
                        if best_score < 75:
                            best_match = os.path.join(triphone_visemes_dir, dir_name)
                            best_score = 75
                            print(f"    Found left+center underscore variant match: {dir_name}")
                            return best_match, best_score
    
    # Priority 5: Combine Priority 3 (Center+Right) with Priority 1 (underscore variants)
    print(f"  Priority 5: Center + Right match with underscore variants")
    if current_phoneme and len(triphone_name) >= 2:
        # Get the base phoneme without underscores
        base_current = current_phoneme.rstrip('_')
        
        # Extract right context
        right_context = ""
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            right_context = triphone_name[current_pos + len(current_phoneme):]
        
        if right_context:
            # Define underscore variants based on current phoneme pattern
            if current_phoneme.endswith('__'):
                # Pattern: a__ -> try a, then a_
                priority_variants = [
                    (base_current, "no underscore"),
                    (base_current + '_', "single underscore")
                ]
            elif current_phoneme.endswith('_'):
                # Pattern: a_ -> try a__, then a
                priority_variants = [
                    (base_current + '__', "double underscore"),
                    (base_current, "no underscore")
                ]
            else:
                # Pattern: a -> try a__, then a_
                priority_variants = [
                    (base_current + '__', "double underscore"),
                    (base_current + '_', "single underscore")
                ]
            
            # Try each underscore variant with center+right pattern
            for variant, description in priority_variants:
                center_right_pattern = variant + right_context
                print(f"    Trying center+right pattern: '*{center_right_pattern}' ({description})")
                
                for dir_name in triphone_dirs:
                    if dir_name.endswith(center_right_pattern) and len(dir_name) > len(center_right_pattern):
                        if best_score < 70:
                            best_match = os.path.join(triphone_visemes_dir, dir_name)
                            best_score = 70
                            print(f"    Found center+right underscore variant match: {dir_name}")
                            return best_match, best_score
    
    # Priority 6: Final fallback - match just the middle (current) phoneme
    print(f"  Priority 6: Middle phoneme only match")
    if current_phoneme:
        print(f"    Looking for any triphone containing middle phoneme: '{current_phoneme}'")
        
        # First try exact middle phoneme match
        for dir_name in triphone_dirs:
            if current_phoneme in dir_name:
                if best_score < 50:
                    best_match = os.path.join(triphone_visemes_dir, dir_name)
                    best_score = 50
                    print(f"    Found middle phoneme match: {dir_name}")
                    return best_match, best_score
        
        # If still no match, try underscore variants of middle phoneme
        base_current = current_phoneme.rstrip('_')
        middle_variants = [
            base_current,           # No underscores
            base_current + '_',     # Single underscore  
            base_current + '__'     # Double underscore
        ]
        
        for variant in middle_variants:
            print(f"    Trying middle phoneme variant: '{variant}'")
            for dir_name in triphone_dirs:
                if variant in dir_name:
                    if best_score < 45:
                        best_match = os.path.join(triphone_visemes_dir, dir_name)
                        best_score = 45
                        print(f"    Found middle variant match: {dir_name}")
                        return best_match, best_score
    
    return best_match, best_score


def load_best_triphone_frames(triphone_visemes_dir, triphone_name, target_frame_count, current_phoneme=None):
    """
    Load frames from the best matching triphone viseme based on target frame count.
    If exact match not found, falls back to closest phoneme match.
    Now includes advanced fallback to ensure we always find something.
    """
    # First try exact triphone match
    best_dir = find_best_triphone_match(triphone_visemes_dir, triphone_name, target_frame_count)
    match_type = 'exact_triphone'
    
    if best_dir is None:
        # Try closest match using pattern matching (now with advanced fallback)
        best_dir, similarity = find_closest_triphone_match(triphone_visemes_dir, triphone_name, target_frame_count, current_phoneme)
        match_type = f'closest_pattern_match_score_{similarity}'
    
    # This should now never happen due to advanced fallback, but keep as safety
    if best_dir is None:
        print(f"  CRITICAL: Still no match found for '{triphone_name}' - this should not happen!")
        return [], None, None, 'no_match'
    
    frames, metadata = load_triphone_frames(best_dir)
    return frames, metadata, best_dir, match_type


def interpolate_frames(frame1, frame2, alpha):
    """Interpolate between two frames using alpha blending."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    frame1_float = frame1.astype(np.float32)
    frame2_float = frame2.astype(np.float32)
    
    interpolated = (1 - alpha) * frame1_float + alpha * frame2_float
    return interpolated.astype(np.uint8)


def create_transition_frames(prev_frame, next_frame, num_frames=3):
    """Create smooth transition frames between two visemes with motion blur."""
    if prev_frame is None or next_frame is None:
        return []
    
    transition_frames = []
    for i in range(num_frames):
        # Use smooth easing function for more natural transitions
        t = (i + 1) / (num_frames + 1)
        # Apply ease-in-out cubic function for smoother transitions
        alpha = t * t * (3.0 - 2.0 * t)
        
        transition_frame = interpolate_frames(prev_frame, next_frame, alpha)
        
        # Apply subtle motion blur for smoother appearance
        transition_frame = apply_motion_blur(transition_frame, intensity=0.3)
        
        transition_frames.append(transition_frame)
    
    return transition_frames


def apply_motion_blur(frame, intensity=0.3):
    """Apply subtle motion blur to a frame for smoother appearance."""
    if intensity <= 0:
        return frame
    
    # Apply Gaussian blur for motion blur effect
    kernel_size = max(3, int(5 * intensity))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    # Blend original with blurred (subtle effect)
    result = cv2.addWeighted(frame, 1.0 - intensity, blurred, intensity, 0)
    
    return result


def interpolate_all_frames(frames, interpolation_factor=2):
    """
    Interpolate frames to increase FPS.
    interpolation_factor=1: doubles FPS (adds 1 frame between each pair)
    interpolation_factor=2: triples FPS (adds 2 frames between each pair)
    interpolation_factor=3: quadruples FPS (adds 3 frames between each pair)
    
    Example with factor=2: [A, B, C] -> [A, A1, A2, B, B1, B2, C] (triples frame count)
    """
    if not frames or len(frames) < 2:
        return frames
    
    if interpolation_factor <= 0:
        return frames
    
    interpolated = []
    
    for i in range(len(frames)):
        # Add the original frame
        interpolated.append(frames[i])
        
        # Add interpolated frames between this and next (except for last frame)
        if i < len(frames) - 1:
            for j in range(1, interpolation_factor + 1):
                alpha = j / (interpolation_factor + 1)
                interp_frame = interpolate_frames(frames[i], frames[i + 1], alpha)
                interpolated.append(interp_frame)
    
    multiplier = interpolation_factor + 1
    print(f"  Frame interpolation: {len(frames)} -> {len(interpolated)} frames ({multiplier}x FPS)")
    return interpolated


def scale_frames_to_duration(frames, current_duration, target_duration, fps):
    """Scale frames to match target duration with smooth interpolation."""
    if not frames or target_duration <= 0:
        return frames
    
    target_frame_count = max(1, int(target_duration * fps))
    
    if target_frame_count == len(frames):
        return frames
    elif target_frame_count < len(frames):
        # Sample frames evenly with interpolation for smoother result
        step = len(frames) / target_frame_count
        scaled_frames = []
        for i in range(target_frame_count):
            pos = i * step
            frame_idx = int(pos)
            alpha = pos - frame_idx
            
            frame_idx = min(frame_idx, len(frames) - 1)
            
            # Interpolate between frames even when downsampling for smoother result
            if alpha > 0.01 and frame_idx < len(frames) - 1:
                interpolated = interpolate_frames(frames[frame_idx], frames[frame_idx + 1], alpha)
                scaled_frames.append(interpolated)
            else:
                scaled_frames.append(frames[frame_idx])
        return scaled_frames
    else:
        # Interpolate to create more frames with smooth transitions
        scaled_frames = []
        for i in range(target_frame_count):
            # Find position in original sequence
            pos = (i / target_frame_count) * len(frames)
            frame_idx = int(pos)
            alpha = pos - frame_idx
            
            # Apply smooth easing for more natural motion
            alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            
            if frame_idx >= len(frames) - 1:
                scaled_frames.append(frames[-1])
            elif alpha < 0.01:  # Very close to an existing frame
                scaled_frames.append(frames[frame_idx])
            else:
                # Interpolate between adjacent frames with smooth easing
                interpolated = interpolate_frames(frames[frame_idx], frames[frame_idx + 1], alpha_smooth)
                scaled_frames.append(interpolated)
        
        return scaled_frames


def add_audio_to_video(video_path, audio_path, output_path, start_time=None, end_time=None):
    """Add audio to video using MoviePy for better quality and smoothing."""
    try:
        # Load video clip
        video_clip = VideoFileClip(video_path)
        
        # Load and prepare audio
        if start_time is not None and end_time is not None:
            audio_clip = AudioFileClip(audio_path).subclip(start_time, end_time)
        else:
            audio_clip = AudioFileClip(audio_path)
        
        # Set audio to video
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write with high quality settings and smoothing
        final_clip.write_videofile(
            output_path,
            fps=video_clip.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            bitrate="8000k",  # High bitrate for quality
            ffmpeg_params=['-crf', '18']  # Low CRF for high quality
        )
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        print(f"High-quality video with audio created successfully: {output_path}")
        
    except Exception as e:
        print(f"MoviePy failed: {e}")
        # Fallback to FFmpeg
        import subprocess
        
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            ffmpeg_cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', audio_path,
                '-t', str(duration),
                '-i', video_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '1:v',
                '-map', '0:a',
                '-shortest',
                '-y',
                output_path
            ]
        else:
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y',
                output_path
            ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Fallback: Video with audio created: {output_path}")


def filter_enriched_sequence_by_time(enriched_sequence, start_time, end_time):
    """
    Filter enriched sequence to only include entries within the specified time range.
    This is useful when processing only a segment of the audio.
    """
    if start_time is None and end_time is None:
        return enriched_sequence
    
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = float('inf')
    
    filtered_sequence = []
    cumulative_time = 0.0
    fps = 25.0  # Assume 25 fps for time calculations
    
    for entry in enriched_sequence:
        # Calculate entry duration based on frame count
        frame_count = get_phoneme_frame_count(entry)
        entry_duration = frame_count / fps
        entry_start = cumulative_time
        entry_end = cumulative_time + entry_duration
        
        # Check if this entry overlaps with our time range
        if entry_start < end_time and entry_end > start_time:
            filtered_sequence.append(entry)
        
        cumulative_time = entry_end
        
        # Stop if we've passed the end time
        if cumulative_time > end_time:
            break
    
    return filtered_sequence


def create_video_from_enriched_sequence(triphone_visemes_dir, enriched_sequence, audio_path, output_path, 
                                        fps=25, start_time=None, end_time=None, 
                                        smooth_transitions=True, motion_interpolation=True,
                                        interpolate_frames_flag=False):
    """Create video by processing enriched phoneme sequence frame by frame."""
    print(f"Creating video from enriched sequence in: {triphone_visemes_dir}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")
    print(f"Settings: FPS={fps}, Smooth Transitions={smooth_transitions}, Motion Interpolation={motion_interpolation}, Interpolate Frames={interpolate_frames_flag}")
    
    # If interpolating frames, we'll triple the effective FPS (adds 2 frames between each pair)
    if interpolate_frames_flag:
        effective_fps = fps * 3
        print(f"Frame interpolation enabled: Effective FPS will be {effective_fps} (3x)")
    else:
        effective_fps = fps
    
    # Get audio duration
    if os.path.exists(audio_path):
        try:
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            audio_clip.close()
        except:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ], capture_output=True, text=True)
            try:
                audio_duration = float(result.stdout.strip())
            except Exception:
                audio_duration = 600.0
    else:
        audio_duration = 600.0
        print(f"Using fallback audio duration: {audio_duration}s")

    # If start_time or end_time are not provided, use full audio
    if start_time is None or end_time is None:
        start_time = 0.0
        end_time = audio_duration
        print(f"Using full audio: {start_time}s to {end_time}s")
    else:
        print(f"Using time segment: {start_time}s to {end_time}s")

    # Filter enriched sequence by time if needed
    if start_time is not None or end_time is not None:
        print(f"Filtering enriched sequence for time range {start_time}s to {end_time}s")
        filtered_sequence = filter_enriched_sequence_by_time(enriched_sequence, start_time, end_time)
        print(f"Filtered to {len(filtered_sequence)} entries")
    else:
        filtered_sequence = enriched_sequence

    print(f"Processing {len(filtered_sequence)} enriched phoneme entries")
    
    # Determine frame size from first available triphone
    frame_size = None
    for i, phoneme_entry in enumerate(filtered_sequence[:20]):  # Check first 20 entries
        current_base = get_base_phoneme(phoneme_entry)
        
        # Skip silence or special tokens
        if current_base in ['spn', '', 'LEFTOVER', 'FINAL_LEFTOVER']:
            continue
            
        # Get triphone context
        left, current, right, triphone_name = get_triphone_context_enriched(filtered_sequence, i)
        
        # Try to load frames to determine size
        frames, metadata, best_dir, match_type = load_best_triphone_frames(triphone_visemes_dir, triphone_name, 3)
        if frames:
            frame_size = (frames[0].shape[1], frames[0].shape[0])
            print(f"Detected frame size: {frame_size}")
            break
    
    if frame_size is None:
        print("Error: Could not determine frame size from triphone frames")
        return
    
    # Process each phoneme entry in the filtered sequence
    all_frames = []
    frames_written = 0
    last_frame = None
    prev_triphone_frames = None  # Track previous phoneme's frames for transitions
    
    # Statistics tracking
    total_phonemes = 0
    exact_triphone_matches = 0
    closest_phoneme_matches = 0
    no_matches = 0
    silence_frames = 0
    transitions_added = 0
    
    for i, phoneme_entry in enumerate(filtered_sequence):
        current_base = get_base_phoneme(phoneme_entry)
        frame_count = get_phoneme_frame_count(phoneme_entry)
        
        # Get triphone context first for logging
        left, current, right, triphone_name = get_triphone_context_enriched(filtered_sequence, i)
        
        print(f"Processing entry {i+1}/{len(filtered_sequence)}: {phoneme_entry['phoneme']} ({frame_count} frames)")
        print(f"  Triphone context: left='{left}', current='{current}', right='{right}'")
        print(f"  Looking for triphone: '{triphone_name}'")
        
        # Skip only specific garbage phonemes - removed 'sil' and 'spn'
        if current_base in ['', 'LEFTOVER', 'FINAL_LEFTOVER']:
            silence_frames += 1
            # Skip these entirely - no frame processing
            print(f"  Skipped garbage phoneme: {current}")
            continue

        total_phonemes += 1
        
        # Load triphone frames
        frames, metadata, best_dir, match_type = load_best_triphone_frames(triphone_visemes_dir, triphone_name, frame_count, current)
        
        # Update match statistics
        if match_type == 'exact_triphone':
            exact_triphone_matches += 1
        elif match_type.startswith('closest_pattern'):
            closest_phoneme_matches += 1
        elif match_type == 'no_match':
            no_matches += 1
        
        # If no frames found at all, this should not happen due to advanced fallback
        if not frames:
            print(f"  CRITICAL ERROR: No frames found for triphone '{triphone_name}' - advanced fallback failed!")
            print(f"  This should not happen! Skipping this phoneme.")
            continue
        
        # Resize frames if needed
        if frames[0].shape[:2][::-1] != frame_size:
            frames = [cv2.resize(frame, frame_size) for frame in frames]
        
        # Scale frames to match required frame count
        if len(frames) != frame_count:
            if frame_count == 0:
                continue
            scaled_frames = []
            for j in range(frame_count):
                orig_idx = (j * len(frames)) / frame_count
                frame_idx = int(orig_idx)
                frame_idx = min(frame_idx, len(frames) - 1)
                scaled_frames.append(frames[frame_idx])
            frames = scaled_frames
        
        # Add smooth transition from previous phoneme (blend first frame instead of adding extra)
        if smooth_transitions and prev_triphone_frames is not None and len(prev_triphone_frames) > 0 and len(frames) > 0:
            # Instead of adding extra frames, blend the first frame with the previous last frame
            # This maintains timing while providing smooth transitions
            frames[0] = interpolate_frames(prev_triphone_frames[-1], frames[0], 0.5)
            transitions_added += 1
            print(f"  Applied smooth transition blend (no extra frames)")
        
        # Add frames to video
        for frame in frames:
            all_frames.append(frame)
            frames_written += 1
            last_frame = frame
        
        # Store current frames for next transition
        prev_triphone_frames = frames
        
        # Log the match result
        if best_dir:
            dir_name = os.path.basename(best_dir)
            print(f"  Added {len(frames)} frames for '{phoneme_entry['phoneme']}' ({match_type}) from '{dir_name}'")
        else:
            print(f"  Added {len(frames)} frames for '{phoneme_entry['phoneme']}' ({match_type})")
    
    print(f"\nTotal frames collected: {len(all_frames)}")
    print(f"Video duration at base FPS: {len(all_frames) / fps:.2f}s")
    print(f"Smooth transitions applied: {transitions_added} (blended, no extra frames)")
    print(f"Expected audio duration: {end_time - start_time:.2f}s")
    
    # Apply frame interpolation if requested (triples FPS for smoother motion)
    if interpolate_frames_flag:
        print(f"\nApplying frame interpolation to triple FPS ({fps} -> {effective_fps})...")
        all_frames = interpolate_all_frames(all_frames, interpolation_factor=2)
        print(f"After interpolation: {len(all_frames)} frames")
        print(f"New video duration: {len(all_frames) / effective_fps:.2f}s")
    
    # Use effective FPS for video creation
    output_fps = effective_fps if interpolate_frames_flag else fps
    
    # Validate FPS before video creation
    if output_fps is None:
        print("ERROR: FPS is None! Setting default to 25")
        output_fps = 25
    print(f"Validated output FPS: {output_fps}")
    
    # Print matching statistics
    print(f"\n=== Phoneme Matching Statistics ===")
    print(f"Total phonemes processed: {total_phonemes}")
    print(f"Exact triphone matches: {exact_triphone_matches} ({exact_triphone_matches/total_phonemes*100:.1f}%)")
    print(f"Closest phoneme matches: {closest_phoneme_matches} ({closest_phoneme_matches/total_phonemes*100:.1f}%)")
    print(f"No matches (neutral frames): {no_matches} ({no_matches/total_phonemes*100:.1f}%)")
    print(f"Silence/special tokens: {silence_frames}")
    total_mismatches = closest_phoneme_matches + no_matches
    print(f"Total mismatches: {total_mismatches} ({total_mismatches/total_phonemes*100:.1f}%)")
    print("=====================================\n")
    
        # Create video using OpenCV initially, then re-encode with better codec
    print("Creating video with high-quality codec...")
    print(f"FPS: {output_fps}, Total frames: {len(all_frames)}")
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    
    # Get frame dimensions
    height, width = all_frames[0].shape[:2]
    
    # First pass: Create video with OpenCV (fast)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, output_fps, (width, height))
    
    # Write frames
    for i, frame in enumerate(all_frames):
        if i % 10 == 0:
            print(f"  Writing frame {i+1}/{len(all_frames)}")
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Temporary video created: {temp_video_path}")
    
    # Second pass: Re-encode with high-quality settings for smooth playback
    temp_hq_video = output_path.replace('.mp4', '_temp_hq.mp4')
    
    if motion_interpolation:
        try:
            print("Re-encoding with high-quality settings and motion interpolation for smoothness...")
            reencode_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-c:v', 'libx264',        # High-quality H.264 codec
                '-preset', 'slow',         # Better compression, higher quality
                '-crf', '18',              # High quality (lower = better, 18 is visually lossless)
                '-pix_fmt', 'yuv420p',    # Compatibility
                '-vf', 'minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'.format(output_fps),  # Motion interpolation for smoothness
                temp_hq_video
            ]
            
            print(f"Running: {' '.join(reencode_cmd)}")
            result = subprocess.run(reencode_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("High-quality re-encoding successful")
                # Use the HQ version as temp for audio addition
                os.remove(temp_video_path)
                temp_video_path = temp_hq_video
            else:
                print(f"Re-encoding warning: {result.stderr}")
                # Continue with original temp video
                if os.path.exists(temp_hq_video):
                    os.remove(temp_hq_video)
        except Exception as e:
            print(f"Re-encoding skipped: {e}")
            # Continue with original temp video
            if os.path.exists(temp_hq_video):
                os.remove(temp_hq_video)
    else:
        # Still re-encode with high quality codec, just without motion interpolation
        try:
            print("Re-encoding with high-quality H.264 codec...")
            reencode_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
                temp_hq_video
            ]
            
            result = subprocess.run(reencode_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                os.remove(temp_video_path)
                temp_video_path = temp_hq_video
            else:
                if os.path.exists(temp_hq_video):
                    os.remove(temp_hq_video)
        except Exception as e:
            print(f"Re-encoding skipped: {e}")
    
    # Add audio using time segment
    if os.path.exists(audio_path):
        print("Adding audio to video...")
        try:
            # Create command to add audio to video for time segment
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', temp_video_path,  # Input video
                '-ss', str(start_time),  # Start time for audio
                '-t', str(end_time - start_time),  # Duration
                '-i', audio_path,  # Input audio
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-map', '0:v',  # Map video from first input
                '-map', '1:a',  # Map audio from second input
                '-shortest',  # End when shortest stream ends
                output_path
            ]
            
            print(f"Adding audio: {' '.join(cmd)}")
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Audio added successfully: {output_path}")
                # Clean up temporary video file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            else:
                print(f"Error adding audio: {result.stderr}")
                # Copy video without audio
                shutil.copy2(temp_video_path, output_path)
                print(f"Video saved without audio: {output_path}")
                # Clean up
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    
        except Exception as e:
            print(f"Error in adding audio: {e}")
            # Copy video without audio as fallback
            shutil.copy2(temp_video_path, output_path)
            print(f"Video saved without audio: {output_path}")
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    else:
        shutil.move(temp_video_path, output_path)
        print(f"No audio file found. Final video saved as: {output_path}")
    
    print(f"Video creation completed: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create video from triphone visemes')
    parser.add_argument('--triphones', default='/home/ist/Desktop/synth_audios/master_audios/actor_3/viseme_library',
                       help='Path to triphone visemes directory')
    parser.add_argument('--json', default='/home/ist/Desktop/synth_audios/gregory_test_2/complete_phoneme_alignments_w_reps_fixed_len.json',
                       help='Path to phoneme alignment JSON file')
    parser.add_argument('--audio', default='/home/ist/Desktop/synth_audios/gregory_test_2/test_audio.mp3',
                       help='Path to audio file (can be video file with audio)')
    parser.add_argument('--output', default='/home/ist/Desktop/synth_audios/gregory_test_2/test_video.mp4',
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=25,
                       help='Output video FPS (default: 25, higher = smoother)')
    parser.add_argument('--smooth-transitions', action='store_true', default=True,
                       help='Add smooth transition frames between phonemes (default: True)')
    parser.add_argument('--motion-interpolation', action='store_true', default=True,
                       help='Use FFmpeg motion interpolation for extra smoothness (default: True)')
    parser.add_argument('--interpolate-frames', action='store_true', default=False,
                       help='Triple FPS by adding 2 interpolated frames between each frame (25->75, 30->90, etc.)')
    parser.add_argument('--start-time', type=float,
                       help='Start time of audio segment in seconds (optional)')
    parser.add_argument('--end-time', type=float,
                       help='End time of audio segment in seconds (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.triphones):
        print(f"Error: Triphones directory not found: {args.triphones}")
        return 1
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        return 1
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return 1
    
    if args.start_time is not None and args.end_time is not None:
        if args.start_time >= args.end_time:
            print(f"Error: Start time must be less than end time")
            return 1
    
    print(f"Creating video from triphones...")
    print(f"Triphones: {args.triphones}")
    print(f"JSON: {args.json}")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.output}")
    print(f"FPS: {args.fps}")
    print(f"Time segment: {args.start_time}s to {args.end_time}s")
    print()
    
    try:
        # Load enriched phoneme data
        enriched_sequence = load_enriched_phoneme_data_from_json(args.json, file_key="audio")
        
        # Create video from enriched sequence
        create_video_from_enriched_sequence(
            args.triphones, 
            enriched_sequence, 
            args.audio, 
            args.output, 
            fps=args.fps,
            start_time=args.start_time,
            end_time=args.end_time,
            smooth_transitions=args.smooth_transitions,
            motion_interpolation=args.motion_interpolation,
            interpolate_frames_flag=args.interpolate_frames
        )
        
        print("\nVideo creation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
