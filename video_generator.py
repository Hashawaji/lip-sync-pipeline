#!/usr/bin/env python3
"""
Video Generator from Triphone Visemes
Supports I-frame + P-frame structure for efficient storage.
"""
import cv2
import os
import numpy as np
import json
import subprocess
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

# Global cache for triphone directories and I-frame
_triphone_dirs_cache = None
_triphone_dirs_set = None  # Fast O(1) lookup set
_triphone_dirs_by_prefix = None  # Dictionary for prefix matching
_triphone_dirs_by_suffix = None  # Dictionary for suffix matching
_triphone_dirs_by_content = None  # Dictionary for content matching
_iframe_cache = None
_iframe_cache_path = None
_triphone_frames_cache = {}  # Cache reconstructed frames to avoid re-decoding P-frames


def load_iframe(actor_library_path):
    """Load the master reference I-frame for an actor library."""
    global _iframe_cache, _iframe_cache_path
    
    iframe_path = os.path.join(actor_library_path, 'master_reference.npz')
    
    # Return cached I-frame if already loaded for this path
    if _iframe_cache is not None and _iframe_cache_path == iframe_path:
        return _iframe_cache
    
    if not os.path.exists(iframe_path):
        raise FileNotFoundError(f"Master reference I-frame not found: {iframe_path}")
    
    try:
        # Load NPZ file containing the I-frame
        data = np.load(iframe_path, allow_pickle=True)
        
        # Try different possible key names
        if 'i_frame' in data:
            encoded_iframe = data['i_frame']
        elif 'iframe' in data:
            encoded_iframe = data['iframe']
        elif 'iframe_data' in data:
            encoded_iframe = data['iframe_data']
        elif 'encoded_frame' in data:
            encoded_iframe = data['encoded_frame']
        else:
            available_keys = list(data.keys())
            raise KeyError(f"No I-frame key found in master_reference.npz. Available keys: {available_keys}")
        
        # Decode the I-frame
        # Handle numpy scalar (0-d array) containing bytes
        if isinstance(encoded_iframe, np.ndarray):
            if encoded_iframe.ndim == 0:
                # Scalar array, extract the value
                encoded_iframe = encoded_iframe.item()
        
        # Now decode the bytes
        if isinstance(encoded_iframe, bytes):
            arr = np.frombuffer(encoded_iframe, dtype=np.uint8)
        elif isinstance(encoded_iframe, np.ndarray):
            if encoded_iframe.dtype == np.uint8:
                arr = encoded_iframe
            else:
                arr = encoded_iframe.astype(np.uint8)
        else:
            arr = np.frombuffer(bytes(encoded_iframe), dtype=np.uint8)
        
        iframe = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if iframe is None:
            raise ValueError("Failed to decode master reference I-frame")
        
        # Cache the I-frame
        _iframe_cache = iframe
        _iframe_cache_path = iframe_path
        
        print(f"Loaded master I-frame from {iframe_path}, shape: {iframe.shape}")
        return iframe
        
    except Exception as e:
        raise RuntimeError(f"Failed to load master I-frame: {e}")


def load_enriched_phoneme_data(json_path):
    """Load enriched phoneme sequence from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    enriched_sequence = data['files']['audio']['enriched_sequence']
    print(f"Loaded {len(enriched_sequence)} phoneme entries from {json_path}")
    return enriched_sequence


def get_phoneme_frame_count(phoneme_entry):
    """
    Get frame count for phoneme:
    - Regular phoneme: 3 frames
    - Single underscore (_): 1 frame  
    - Double underscore (__): 2 frames
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
    return phoneme.rstrip('_')


def load_triphone_frames(triphone_dir, iframe):
    """
    Load P-frames from a triphone directory and reconstruct full frames using I-frame.
    Uses aggressive caching to avoid re-decoding P-frames.
    
    Args:
        triphone_dir: Path to triphone directory containing p_frames.npz
        iframe: Master reference I-frame to apply P-frames to
        
    Returns:
        List of reconstructed frames
    """
    global _triphone_frames_cache
    
    # Check cache first - this is the key optimization!
    if triphone_dir in _triphone_frames_cache:
        return _triphone_frames_cache[triphone_dir]
    
    if not os.path.exists(triphone_dir):
        return []

    # Look for p_frames.npz file (new structure) or frames.npz (legacy)
    p_frame_path = os.path.join(triphone_dir, 'p_frames.npz')
    legacy_path = os.path.join(triphone_dir, 'frames.npz')
    
    npz_path = p_frame_path if os.path.exists(p_frame_path) else legacy_path
    
    if not os.path.exists(npz_path):
        _triphone_frames_cache[triphone_dir] = []
        return []
    
    try:
        # Load NPZ file
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Find the encoded frames array (key might be 'p_frames', 'encoded_frames', 'j', etc.)
        encoded_frames = None
        
        # Try known P-frame keys first
        for key in ['p_frames', 'encoded_frames', 'j']:
            if key in data:
                encoded_frames = data[key]
                break
        
        # If still not found, search for any bytes/string array
        if encoded_frames is None:
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.dtype.kind in ('S', 'O'):  # String/bytes or Object
                    encoded_frames = value
                    break
        
        if encoded_frames is None or len(encoded_frames) == 0:
            _triphone_frames_cache[triphone_dir] = []
            return []
        
        # ASSUME ALL FRAMES ARE P-FRAMES in the new multi-actor structure
        # (For backward compatibility with old full-frame libraries, we'd need different detection)
        # Since user has master_reference.npz, this is definitely P-frame format
        frames = reconstruct_frames_from_pframes(encoded_frames, iframe)
        
        # Cache the reconstructed frames!
        _triphone_frames_cache[triphone_dir] = frames
        
        return frames
        
    except Exception as e:
        # Silently fail - will be too verbose otherwise
        _triphone_frames_cache[triphone_dir] = []
        return []


def _decode_single_pframe(p_frame_data, iframe):
    """
    Decode a single P-frame. Used for parallel processing.
    
    Args:
        p_frame_data: Encoded P-frame bytes
        iframe: Master reference I-frame
        
    Returns:
        Reconstructed frame or None if failed
    """
    try:
        # Handle numpy.bytes_ (scalar bytes) or regular bytes
        if isinstance(p_frame_data, (np.bytes_, bytes)):
            arr = np.frombuffer(bytes(p_frame_data), dtype=np.uint8)
        # Handle numpy array types
        elif isinstance(p_frame_data, np.ndarray):
            if p_frame_data.dtype == np.uint8:
                arr = p_frame_data
            else:
                arr = p_frame_data.astype(np.uint8)
        else:
            arr = np.frombuffer(bytes(p_frame_data), dtype=np.uint8)
        
        # Decode JPEG difference frame (this contains diff + 128)
        diff_shifted = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if diff_shifted is None:
            return None
        
        # Ensure same dimensions
        if diff_shifted.shape != iframe.shape:
            diff_shifted = cv2.resize(diff_shifted, (iframe.shape[1], iframe.shape[0]))
        
        # Shift back: subtract 128 to get the actual signed difference
        diff = diff_shifted.astype(np.int16) - 128
        
        # Reconstruct: master I-frame + diff
        reconstructed = iframe.astype(np.int16) + diff
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return reconstructed
        
    except Exception as e:
        # Skip problematic frames
        return None


def reconstruct_frames_from_pframes(p_frames, iframe):
    """
    Reconstruct full frames by applying P-frame deltas to I-frame.
    Uses ThreadPoolExecutor for parallel JPEG decoding (I/O bound operation).
    
    The P-frames are stored as JPEG-encoded differential frames where:
    - The actual diff is calculated as: diff = frame - iframe
    - To store in JPEG (which needs positive values), it's shifted: diff_shifted = diff + 128
    - To reconstruct: frame = iframe + (diff_shifted - 128)
    
    Args:
        p_frames: List of encoded P-frames (differential data as JPEG bytes)
        iframe: Master reference I-frame
        
    Returns:
        List of reconstructed full frames
    """
    # For small frame counts, parallel overhead isn't worth it
    if len(p_frames) <= 3:
        frames = []
        for p_frame_data in p_frames:
            frame = _decode_single_pframe(p_frame_data, iframe)
            if frame is not None:
                frames.append(frame)
        return frames
    
    # Use ThreadPoolExecutor for parallel JPEG decoding
    # JPEG decoding is I/O-bound, so threads work well here (no GIL issues)
    # Increased workers from 4 to 16 for better I/O saturation
    frames = []
    max_workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all decoding tasks
        future_to_index = {
            executor.submit(_decode_single_pframe, p_frame_data, iframe): idx
            for idx, p_frame_data in enumerate(p_frames)
        }
        
        # Collect results in order
        results = [None] * len(p_frames)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    results[idx] = result
            except Exception as e:
                pass  # Skip failed frames
        
        # Filter out None values and maintain order
        frames = [frame for frame in results if frame is not None]
    
    return frames


def get_triphone_context(enriched_sequence, index):
    """Get triphone context (left + current + right) for phoneme at index."""
    current_entry = enriched_sequence[index]
    current = current_entry['phoneme']
    current_base = get_base_phoneme(current_entry)
    
    garbage_phonemes = {'', 'LEFTOVER', 'FINAL_LEFTOVER', 'spn'}
    
    # Get left context
    left = None
    for i in range(index - 1, -1, -1):
        prev_base = get_base_phoneme(enriched_sequence[i])
        if prev_base not in garbage_phonemes:
            left = prev_base
            break
    
    # Get right context
    right = None
    for i in range(index + 1, len(enriched_sequence)):
        next_base = get_base_phoneme(enriched_sequence[i])
        if next_base not in garbage_phonemes:
            right = next_base
            break
    
    # Create triphone name
    if left and right:
        triphone_name = left + current + right
    elif left:
        triphone_name = left + current
    elif right:
        triphone_name = current + right
    else:
        triphone_name = current
    
    return left, current, right, triphone_name


def get_triphone_dirs(triphone_visemes_dir):
    """Get cached list of triphone directories with optimized lookup structures."""
    global _triphone_dirs_cache, _triphone_dirs_set
    global _triphone_dirs_by_prefix, _triphone_dirs_by_suffix, _triphone_dirs_by_content
    
    if _triphone_dirs_cache is None:
        print("Caching triphone directories with optimized indexes...")
        _triphone_dirs_cache = [item for item in os.listdir(triphone_visemes_dir) 
                               if os.path.isdir(os.path.join(triphone_visemes_dir, item))]
        
        # Create fast lookup structures
        _triphone_dirs_set = set(_triphone_dirs_cache)
        
        # Create prefix index for fast prefix matching
        _triphone_dirs_by_prefix = {}
        for dir_name in _triphone_dirs_cache:
            for i in range(1, len(dir_name) + 1):
                prefix = dir_name[:i]
                if prefix not in _triphone_dirs_by_prefix:
                    _triphone_dirs_by_prefix[prefix] = []
                _triphone_dirs_by_prefix[prefix].append(dir_name)
        
        # Create suffix index for fast suffix matching
        _triphone_dirs_by_suffix = {}
        for dir_name in _triphone_dirs_cache:
            for i in range(len(dir_name)):
                suffix = dir_name[i:]
                if suffix not in _triphone_dirs_by_suffix:
                    _triphone_dirs_by_suffix[suffix] = []
                _triphone_dirs_by_suffix[suffix].append(dir_name)
        
        # Create content index for fast substring matching
        _triphone_dirs_by_content = {}
        for dir_name in _triphone_dirs_cache:
            for i in range(len(dir_name)):
                for j in range(i + 1, len(dir_name) + 1):
                    substring = dir_name[i:j]
                    if substring not in _triphone_dirs_by_content:
                        _triphone_dirs_by_content[substring] = []
                    _triphone_dirs_by_content[substring].append(dir_name)
        
        print(f"Cached {len(_triphone_dirs_cache)} triphone directories with fast lookup indexes")
    
    return _triphone_dirs_cache


def find_best_triphone_match(triphone_visemes_dir, triphone_name, current_phoneme):
    """
    Find best matching triphone using priority-based matching with optimized O(1) lookups:
    1. Exact match
    2. Underscore variants (prioritized by target frame count)
    3. Left + center match
    4. Center + right match
    5. Middle phoneme only
    """
    get_triphone_dirs(triphone_visemes_dir)  # Ensure cache is loaded
    
    global _triphone_dirs_set, _triphone_dirs_by_prefix, _triphone_dirs_by_suffix, _triphone_dirs_by_content
    
    # Safety check (should never happen after get_triphone_dirs call)
    if _triphone_dirs_set is None:
        return None, 0
    
    # Priority 1: Exact match - O(1) lookup with set
    if triphone_name in _triphone_dirs_set:
        return os.path.join(triphone_visemes_dir, triphone_name), 100
    
    # Priority 2: Underscore variants - O(1) lookup with set
    if current_phoneme:
        base_current = current_phoneme.rstrip('_')
        left_context = ""
        right_context = ""
        
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            left_context = triphone_name[:current_pos]
            right_context = triphone_name[current_pos + len(current_phoneme):]
        
        # Define priority order based on current phoneme
        if current_phoneme.endswith('__'):
            priority_variants = [base_current, base_current + '_']
        elif current_phoneme.endswith('_'):
            priority_variants = [base_current + '__', base_current]
        else:
            priority_variants = [base_current + '__', base_current + '_']
        
        # Try variants with O(1) set lookup
        for variant in priority_variants:
            test_triphone = left_context + variant + right_context
            if test_triphone in _triphone_dirs_set:
                return os.path.join(triphone_visemes_dir, test_triphone), 95
    
    # Priority 3: Left + center match - Use prefix index
    if current_phoneme and len(triphone_name) >= 2 and _triphone_dirs_by_prefix is not None:
        left_context = ""
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            left_context = triphone_name[:current_pos]
        
        if left_context:
            left_center_pattern = left_context + current_phoneme
            # Use prefix index for O(1) lookup instead of O(n) iteration
            if left_center_pattern in _triphone_dirs_by_prefix:
                candidates = _triphone_dirs_by_prefix[left_center_pattern]
                for dir_name in candidates:
                    if len(dir_name) > len(left_center_pattern):
                        return os.path.join(triphone_visemes_dir, dir_name), 85
    
    # Priority 4: Center + right match - Use suffix index
    if current_phoneme and len(triphone_name) >= 2 and _triphone_dirs_by_suffix is not None:
        right_context = ""
        if current_phoneme in triphone_name:
            current_pos = triphone_name.find(current_phoneme)
            right_context = triphone_name[current_pos + len(current_phoneme):]
        
        if right_context:
            center_right_pattern = current_phoneme + right_context
            # Use suffix index for O(1) lookup instead of O(n) iteration
            if center_right_pattern in _triphone_dirs_by_suffix:
                candidates = _triphone_dirs_by_suffix[center_right_pattern]
                for dir_name in candidates:
                    if len(dir_name) > len(center_right_pattern):
                        return os.path.join(triphone_visemes_dir, dir_name), 80
    
    # Priority 5: Middle phoneme match - Use content index
    if current_phoneme and _triphone_dirs_by_content is not None:
        # Use content index for O(1) lookup instead of O(n) iteration
        if current_phoneme in _triphone_dirs_by_content:
            candidates = _triphone_dirs_by_content[current_phoneme]
            if candidates:
                return os.path.join(triphone_visemes_dir, candidates[0]), 50
        
        # Try underscore variants with content index
        base_current = current_phoneme.rstrip('_')
        for variant in [base_current, base_current + '_', base_current + '__']:
            if variant in _triphone_dirs_by_content:
                candidates = _triphone_dirs_by_content[variant]
                if candidates:
                    return os.path.join(triphone_visemes_dir, candidates[0]), 45
    
    return None, 0


def preload_triphones_parallel(triphone_dirs, iframe, max_workers=8):
    """
    Preload multiple triphones in parallel using ThreadPoolExecutor.
    This dramatically speeds up video generation by loading all needed triphones upfront.
    
    Args:
        triphone_dirs: List of triphone directory paths to preload
        iframe: Master reference I-frame
        max_workers: Maximum parallel workers (default: 8)
    """
    global _triphone_frames_cache
    
    # Filter out already cached triphones
    dirs_to_load = [d for d in triphone_dirs if d not in _triphone_frames_cache]
    
    if not dirs_to_load:
        return  # All already cached
    
    print(f"Preloading {len(dirs_to_load)} triphones in parallel (using {max_workers} workers)...")
    start_time = time.time()
    
    # Load triphones in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all load tasks
        future_to_dir = {
            executor.submit(load_triphone_frames, triphone_dir, iframe): triphone_dir
            for triphone_dir in dirs_to_load
        }
        
        # Wait for completion with progress tracking
        completed = 0
        for future in as_completed(future_to_dir):
            triphone_dir = future_to_dir[future]
            try:
                future.result()  # This will populate the cache
                completed += 1
                if completed % 50 == 0:
                    print(f"  Loaded {completed}/{len(dirs_to_load)} triphones...")
            except Exception as e:
                pass  # Skip failed loads
    
    elapsed = time.time() - start_time
    print(f"✓ Preloaded {len(dirs_to_load)} triphones in {elapsed:.2f}s ({len(dirs_to_load)/elapsed:.1f} triphones/sec)")
    print()


def interpolate_frames(frame1, frame2, alpha):
    """Interpolate between two frames using alpha blending with optimized NumPy operations."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Optimized: Use cv2.addWeighted for hardware-accelerated blending
    # This is much faster than manual float conversion and multiplication
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


def interpolate_all_frames(frames, interpolation_factor=1):
    """
    Interpolate frames naturally without forcing exact frame counts.
    For n input frames with interpolation_factor=1: creates (2n-1) output frames.
    
    The final FPS will be calculated dynamically to maintain correct duration.
    This ensures perfect audio-video sync without artificial frame duplication.
    """
    if not frames or len(frames) < 1:
        return frames
    
    if interpolation_factor <= 0:
        return frames
    
    # Natural interpolation: add frames between adjacent pairs
    interpolated = []
    
    for i in range(len(frames)):
        # Add the original frame
        interpolated.append(frames[i])
        
        # Add interpolated frames (except for the very last input frame)
        if i < len(frames) - 1:
            for j in range(1, interpolation_factor + 1):
                alpha = j / (interpolation_factor + 1)
                interp_frame = interpolate_frames(frames[i], frames[i + 1], alpha)
                interpolated.append(interp_frame)
    
    # Natural result: n + (n-1) * interpolation_factor frames
    # For factor=1: 2n-1 frames (no artificial duplication)
    multiplier = len(interpolated) / len(frames)
    print(f"  Frame interpolation: {len(frames)} -> {len(interpolated)} frames ({multiplier:.2f}x)")
    return interpolated


def scale_frames_to_duration(frames, target_frame_count, fps):
    """
    Scale frames to match target frame count with smooth interpolation.
    Uses cubic easing for more natural transitions.
    """
    if not frames or target_frame_count <= 0:
        return frames
    
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
            
            # Apply smooth easing for more natural motion (cubic easing)
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


def generate_video(triphone_visemes_dir, json_path, audio_path, output_path, 
                   blink_applier=None, actor_blink_assets_path=None):
    """
    Generate browser-compatible video from phoneme alignments and triphone visemes.
    Optimized for speed and direct H.264 encoding without intermediate files.
    
    Args:
        triphone_visemes_dir: Path to viseme library (actor-specific)
        json_path: Path to phoneme alignment JSON
        audio_path: Path to audio file
        output_path: Output video path
        blink_applier: Optional BlinkApplier instance (if None, no blinks applied)
        actor_blink_assets_path: Optional path to actor's blink assets (for scheduling)
    
    Returns:
        str: Path to generated video
    """
    pipeline_start = time.time()
    timings = {}  # Track timing for each stage
    
    if not os.path.exists(triphone_visemes_dir):
        raise FileNotFoundError(f"Viseme library not found: {triphone_visemes_dir}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Phoneme alignment JSON not found: {json_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    fps = 25  # Base FPS
    interpolation_factor = 1  # Adds interpolated frames between adjacent frames
    
    print(f"=== Fast Browser Video Pipeline ===")
    print(f"Viseme library: {triphone_visemes_dir}")
    print(f"JSON: {json_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")
    print(f"Settings: Base FPS={fps}, Interpolation=enabled")
    print()
    
    # Load master I-frame for this actor library
    stage_start = time.time()
    print("Loading master reference I-frame...")
    iframe = load_iframe(triphone_visemes_dir)
    timings['I-frame loading'] = time.time() - stage_start
    print(f"✓ I-frame loaded: {iframe.shape} ({timings['I-frame loading']:.2f}s)")
    print()
    
    # Load phoneme data
    stage_start = time.time()
    enriched_sequence = load_enriched_phoneme_data(json_path)
    timings['Phoneme data loading'] = time.time() - stage_start
    print(f"Processing {len(enriched_sequence)} phoneme entries ({timings['Phoneme data loading']:.2f}s)")
    
    # Determine frame size from I-frame
    frame_size = (iframe.shape[1], iframe.shape[0])
    print(f"Frame size: {frame_size}")
    
    # === OPTIMIZATION: Preload all needed triphones in parallel ===
    stage_start = time.time()
    print("Analyzing phoneme sequence to identify needed triphones...")
    triphone_dirs_to_load = set()
    
    for i, phoneme_entry in enumerate(enriched_sequence):
        current_base = get_base_phoneme(phoneme_entry)
        if current_base in ['', 'LEFTOVER', 'FINAL_LEFTOVER', 'spn']:
            continue
            
        left, current, right, triphone_name = get_triphone_context(enriched_sequence, i)
        best_dir, score = find_best_triphone_match(triphone_visemes_dir, triphone_name, current)
        
        if best_dir:
            triphone_dirs_to_load.add(best_dir)
    
    timings['Triphone analysis'] = time.time() - stage_start
    print(f"Identified {len(triphone_dirs_to_load)} unique triphones needed ({timings['Triphone analysis']:.2f}s)")
    
    # Preload all triphones in parallel (MASSIVE speedup!)
    stage_start = time.time()
    if triphone_dirs_to_load:
        preload_triphones_parallel(list(triphone_dirs_to_load), iframe, max_workers=8)
    timings['Triphone preloading'] = time.time() - stage_start
    
    # Verify triphone loading with first available phoneme
    for i, phoneme_entry in enumerate(enriched_sequence[:20]):
        current_base = get_base_phoneme(phoneme_entry)
        if current_base in ['spn', '', 'LEFTOVER', 'FINAL_LEFTOVER']:
            continue
            
        left, current, right, triphone_name = get_triphone_context(enriched_sequence, i)
        best_dir, score = find_best_triphone_match(triphone_visemes_dir, triphone_name, current)
        
        if best_dir:
            frames = load_triphone_frames(best_dir, iframe)
            if frames:
                print(f"Verified triphone loading: '{triphone_name}' → {len(frames)} frames")
                break
    
    print()
    
    # Process each phoneme entry
    stage_start = time.time()
    all_frames = []
    prev_frames = None
    
    # Statistics tracking
    stats = {
        'total': 0,
        'exact': 0,
        'fallback': 0,
        'skipped': 0,
        'silence': 0
    }
    
    for i, phoneme_entry in enumerate(enriched_sequence):
        current_base = get_base_phoneme(phoneme_entry)
        frame_count = get_phoneme_frame_count(phoneme_entry)
        
        left, current, right, triphone_name = get_triphone_context(enriched_sequence, i)
        
        # Progress logging
        if (i + 1) % 50 == 0:
            print(f"Processing phoneme {i + 1}/{len(enriched_sequence)}...")
        
        # Handle silence/leftover/spn phonemes by holding last frame
        if current_base in ['', 'LEFTOVER', 'FINAL_LEFTOVER', 'spn']:
            stats['silence'] += 1
            # For silence/spn, use previous frame or find 'sil' triphone
            if prev_frames and len(prev_frames) > 0:
                hold_frame = prev_frames[-1]  # Keep last frame for spn/silence
            else:
                # Try to find 'sil' triphone for neutral pose
                sil_dir, _ = find_best_triphone_match(triphone_visemes_dir, 'sil', 'sil')
                if sil_dir:
                    sil_frames = load_triphone_frames(sil_dir, iframe)
                    if sil_frames:
                        hold_frame = sil_frames[0]
                    else:
                        hold_frame = iframe
                else:
                    hold_frame = iframe
            
            # Add held frames
            for _ in range(frame_count):
                all_frames.append(hold_frame.copy())
            continue
        
        stats['total'] += 1
        
        # Find matching triphone - should always find with comprehensive library
        best_dir, score = find_best_triphone_match(triphone_visemes_dir, triphone_name, current)
        
        if not best_dir:
            print(f"ERROR: No match found for triphone '{triphone_name}' (phoneme: '{current}')")
            stats['skipped'] += 1
            continue
        
        # Track match quality
        if score >= 95:
            stats['exact'] += 1
        else:
            stats['fallback'] += 1
        
        # Load frames with P-frame reconstruction
        frames = load_triphone_frames(best_dir, iframe)
        if not frames:
            print(f"ERROR: Failed to load frames for triphone '{triphone_name}'")
            stats['skipped'] += 1
            continue
        
        # Resize if needed
        if frames[0].shape[:2][::-1] != frame_size:
            frames = [cv2.resize(frame, frame_size) for frame in frames]
        
        # Scale to match required frame count using smooth interpolation
        if len(frames) != frame_count:
            if frame_count == 0:
                continue
            frames = scale_frames_to_duration(frames, frame_count, fps)
        
        # Add frames to collection
        all_frames.extend(frames)
        prev_frames = frames
    
    timings['Frame collection'] = time.time() - stage_start
    
    print(f"\n=== Frame Collection Complete ===")
    print(f"Total frames: {len(all_frames)}")
    print(f"Video duration at base FPS: {len(all_frames) / fps:.2f}s")
    print(f"Time: {timings['Frame collection']:.2f}s")
    print()
    
    # Check if we have frames
    if len(all_frames) == 0:
        raise ValueError(
            f"No frames collected! This indicates an issue with the viseme library.\n"
            f"Statistics: {stats}\n"
            f"Please check:\n"
            f"1. Viseme library path: {triphone_visemes_dir}\n"
            f"2. P-frame files exist and are readable\n"
            f"3. Phoneme sequence contains valid phonemes"
        )
    
    # === APPLY BLINKS BEFORE INTERPOLATION ===
    if blink_applier is not None and actor_blink_assets_path is not None:
        stage_start = time.time()
        print(f"\n=== Applying Blinks to Frames ===")
        print(f"Current frame count: {len(all_frames)} @ {fps} FPS")
        
        # Import BlinkScheduler
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "blink_module"))
        from BlinkScheduler import BlinkScheduler
        
        # Calculate video duration
        video_duration = len(all_frames) / fps
        
        # Generate intelligent blink schedule
        print(f"Generating intelligent blink schedule...")
        blink_scheduler = BlinkScheduler.from_phoneme_json(
            json_path=Path(json_path),  # Convert string to Path
            fps=fps,
            total_duration_sec=video_duration
        )
        
        blink_start_frames, blink_stats = blink_scheduler.generate_blink_frames()
        
        # Log blink statistics
        snapped_percentage = (blink_stats['snapped_to_pause']/blink_stats['total_blinks']*100) if blink_stats['total_blinks'] > 0 else 0
        print(f"Blink schedule generated:")
        print(f"  - Total blinks: {blink_stats['total_blinks']}")
        print(f"  - Snapped to pauses: {blink_stats['snapped_to_pause']} ({snapped_percentage:.1f}%)")
        print(f"  - Baseline blinks: {blink_stats['baseline_blinks']}")
        print(f"  - Average IBI: {blink_stats['average_ibi_sec']:.2f}s")
        print(f"  - Blink frames: {sorted(list(blink_start_frames))}")
        
        # Apply blinks to frames in memory
        print(f"Applying blinks to {len(blink_start_frames)} frame positions...")
        all_frames = blink_applier.apply_blinks_to_frames(
            frames=all_frames,
            blink_start_frames=blink_start_frames,
            fps=fps
        )
        
        timings['Blink application'] = time.time() - stage_start
        print(f"✓ Blinks applied to frames ({timings['Blink application']:.2f}s)")
        print()
    else:
        print(f"\n=== No Blinks Applied ===")
        print(f"Reason: {'No BlinkApplier provided' if blink_applier is None else 'No blink assets path provided'}")
        print()
    
    stage_start = time.time()
    print(f"Applying frame interpolation...")
    base_frame_count = len(all_frames)
    all_frames = interpolate_all_frames(all_frames, interpolation_factor=interpolation_factor)
    timings['Frame interpolation'] = time.time() - stage_start
    
    # Calculate dynamic FPS to maintain exact audio duration
    interpolated_frame_count = len(all_frames)
    
    # Load the phoneme data to get exact duration
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    enriched_sequence = json_data['files']['audio']['enriched_sequence']
    total_duration = sum(entry['duration_s'] for entry in enriched_sequence)
    
    # Calculate the FPS needed to maintain exact duration
    output_fps = interpolated_frame_count / total_duration
    
    print(f"After interpolation: {interpolated_frame_count} frames (was {base_frame_count})")
    print(f"MFA duration: {total_duration:.3f}s")
    print(f"Dynamic output FPS: {output_fps:.2f} (maintains exact sync)")
    print(f"Time: {timings['Frame interpolation']:.2f}s")
    
    print(f"\n=== Phoneme Matching Statistics ===")
    if stats['total'] > 0:
        print(f"Total phonemes processed: {stats['total']}")
        print(f"Exact triphone matches: {stats['exact']} ({stats['exact']/stats['total']*100:.1f}%)")
        print(f"Fallback matches: {stats['fallback']} ({stats['fallback']/stats['total']*100:.1f}%)")
        print(f"Skipped phonemes: {stats['skipped']}")
        print(f"Silence/special tokens: {stats['silence']}")
    else:
        print("No phonemes processed!")
    print("=" * 36)
    print()
    
    stage_start = time.time()
    print(f"Creating browser-compatible H.264 video with direct FFmpeg encoding...")
    print(f"Output FPS: {output_fps}, Total frames: {len(all_frames)}")
    
    # Get FFmpeg path
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        ffmpeg_path = '/opt/homebrew/Caskroom/miniconda/base/envs/mfa-dev/bin/ffmpeg'
    
    if not os.path.exists(ffmpeg_path):
        raise RuntimeError(f"FFmpeg not found at {ffmpeg_path}. Cannot create browser-compatible video.")
    
    height, width = all_frames[0].shape[:2]
    
    # OPTIMIZED: Direct FFmpeg pipe encoding (no temp file!)
    # This eliminates the two-pass encoding bottleneck
    print(f"Starting direct FFmpeg encoding with VideoToolbox...")
    print(f"  Frames: {len(all_frames)}, Size: {width}x{height}, FPS: {output_fps}")
    
    # Build FFmpeg command for direct pipe input
    encoder = 'h264_videotoolbox'  # Fast on macOS
    
    ffmpeg_cmd = [
        ffmpeg_path, '-y',
        # Input: raw video from pipe
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(output_fps),
        '-i', '-',  # Read from stdin
        # Input: audio file
        '-i', audio_path,
        # Output: H.264 encoding
        '-c:v', encoder,
        '-b:v', '2M',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_path
    ]
    
    # Start FFmpeg process
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Write frames directly to FFmpeg stdin
    try:
        for i, frame in enumerate(all_frames):
            if (i + 1) % 100 == 0:
                print(f"  Encoding frame {i + 1}/{len(all_frames)}")
            
            # Write raw frame bytes to FFmpeg
            process.stdin.write(frame.tobytes())
        
        # Close stdin and wait for FFmpeg to finish
        # Don't call communicate() after closing stdin manually
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            timings['H.264 encoding'] = time.time() - stage_start
            print(f"✓ Direct FFmpeg encoding complete ({timings['H.264 encoding']:.2f}s)")
        else:
            # Get stderr for error reporting
            stderr_output = process.stderr.read()
            print(f"✗ VideoToolbox encoding failed (return code: {process.returncode})")
            print(f"Error: {stderr_output.decode()[:500] if stderr_output else 'No error output'}")
            print("Trying fallback encoder (libx264)...")
            
            # Fallback: libx264 encoder
            ffmpeg_cmd_fallback = [
                ffmpeg_path, '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(output_fps),
                '-i', '-',
                '-i', audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-b:v', '2M',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                output_path
            ]
            
            process2 = subprocess.Popen(
                ffmpeg_cmd_fallback,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            for i, frame in enumerate(all_frames):
                if (i + 1) % 100 == 0:
                    print(f"  Encoding frame {i + 1}/{len(all_frames)} (fallback)")
                process2.stdin.write(frame.tobytes())
            
            process2.stdin.close()
            process2.wait()
            
            if process2.returncode == 0:
                timings['H.264 encoding'] = time.time() - stage_start
                print(f"✓ Fallback encoding successful ({timings['H.264 encoding']:.2f}s)")
            else:
                stderr2_output = process2.stderr.read()
                raise RuntimeError(f"Both encoders failed. VideoToolbox: {stderr_output.decode()[:200] if stderr_output else 'unknown'}, libx264: {stderr2_output.decode()[:200] if stderr2_output else 'unknown'}")
    
    except Exception as e:
        process.kill()
        raise RuntimeError(f"FFmpeg encoding error: {e}")
    
    total_time = time.time() - pipeline_start
    
    # Check if the final video file exists and has audio  
    has_audio = False
    is_h264 = False
    if os.path.exists(output_path):
        try:
            probe_cmd = [ffmpeg_path.replace('/ffmpeg', '/ffprobe'), '-v', 'quiet', 
                        '-print_format', 'json', '-show_streams', output_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json as json_module
                streams = json_module.loads(result.stdout)
                
                # Check for audio streams
                for stream in streams.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        has_audio = True
                    elif stream.get('codec_type') == 'video' and stream.get('codec_name') == 'h264':
                        is_h264 = True
        except Exception:
            pass
    
    print(f"\n✓ Fast browser video generation complete: {output_path}")
    print(f"  FPS: {output_fps}, Audio: {'Yes' if has_audio else 'No'}")
    print(f"  Codec: {'H.264 (browser-optimized)' if is_h264 else 'Other'}")
    print(f"  Web streaming: {'Optimized' if is_h264 and has_audio else 'Limited'}")
    
    print("=" * 60)
    print("=== Performance Breakdown ===")
    for stage, duration in timings.items():
        if duration > 0:
            percentage = (duration / total_time) * 100
            print(f"  {stage}: {duration:.2f}s ({percentage:.1f}%)")
    print(f"  TOTAL: {total_time:.2f}s")
    
    # Performance insights
    encoding_time = timings.get('H.264 encoding', 0)
    if encoding_time > 0:
        frames_per_second = len(all_frames) / encoding_time
        print(f"  Encoding speed: {frames_per_second:.1f} frames/sec")
    
    print("=" * 60)
    
    return output_path


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate video from triphone visemes')
    parser.add_argument('--visemes', default='/home/ist/Desktop/lip-sync-pipeline/viseme_library',
                       help='Path to viseme library (actor-specific directory)')
    parser.add_argument('--json', required=True,
                       help='Path to phoneme alignment JSON')
    parser.add_argument('--audio', required=True,
                       help='Path to audio file')
    parser.add_argument('--output', required=True,
                       help='Output video path')
    
    args = parser.parse_args()
    
    generate_video(
        triphone_visemes_dir=args.visemes,
        json_path=args.json,
        audio_path=args.audio,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
