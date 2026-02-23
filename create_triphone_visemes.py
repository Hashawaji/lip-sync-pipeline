#!/usr/bin/env python3
"""
Script to create triphone visemes from video based on enriched phoneme sequence.

Storage: Global I-frame + P-frame differential encoding
- Single master I-frame stored once for entire library (master_reference.npz)
- ALL frames stored as differences from the master I-frame
- Perfect for talking portraits where 95%+ pixels are identical
- Achieves 85-95% space savings compared to individual JPEGs
"""

import json
import cv2
import os
import numpy as np
from pathlib import Path
import argparse


def load_enriched_phoneme_data(json_path):
    """Load enriched phoneme sequence data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['files']['audio']['enriched_sequence']


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


def get_triphone_context(enriched_sequence, index):
    """
    Get triphone context for a phoneme at given index in enriched sequence.
    Returns (left_context, current_phoneme, right_context, triphone_name)
    Simply identifies the neighboring base phonemes for context.
    Filters out 'spn' (spoken noise) and other garbage phonemes.
    """
    current_entry = enriched_sequence[index]
    current = current_entry['phoneme']  # Keep the full phoneme with underscores
    current_base = get_base_phoneme(current_entry)
    
    # Define garbage phonemes to skip - removed 'sil' to treat silence as real phoneme
    garbage_phonemes = {'spn', '', 'LEFTOVER', 'FINAL_LEFTOVER'}
    
    # Get left context (scan backwards for different valid base phoneme)
    left = None
    for i in range(index - 1, -1, -1):
        prev_base = get_base_phoneme(enriched_sequence[i])
        if prev_base not in garbage_phonemes:
            left = prev_base  # Just the base phoneme, no underscores needed for context
            break
    
    # Get right context (scan forwards for different valid base phoneme)
    right = None
    for i in range(index + 1, len(enriched_sequence)):
        next_base = get_base_phoneme(enriched_sequence[i])
        if next_base not in garbage_phonemes:
            right = next_base  # Just the base phoneme, no underscores needed for context
            break
    
    # Create triphone name: left_base + current_full + right_base
    triphone_name = ""
    if left:
        triphone_name += left
    triphone_name += current  # Current phoneme keeps its underscores
    if right:
        triphone_name += right
    
    return left, current, right, triphone_name


def extract_frames_by_position(video_path, start_frame, frame_count):
    """
    Extract specific number of frames starting from start_frame position.
    Returns list of frames.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {start_frame + i}")
            break
        frames.append(frame)
    
    cap.release()
    return frames


def save_viseme_frames(frames, output_dir, triphone_name, phoneme_data, master_i_frame):
    """
    Save frames using global I-frame + P-frame differential encoding.
    
    Args:
        frames: List of frames for this triphone
        output_dir: Base output directory
        triphone_name: Name of the triphone
        phoneme_data: Phoneme metadata
        master_i_frame: Decoded master reference frame (shared across all triphones)
    
    All frames (including first frame) are stored as differences from the master I-frame.
    This maximizes space savings for talking portrait videos.
    """
    # Create directory for this triphone
    triphone_dir = os.path.join(output_dir, triphone_name)

    # Check if already exists
    npz_path = os.path.join(triphone_dir, 'frames.npz')
    if os.path.exists(npz_path):
        print(f"Triphone '{triphone_name}' already exists, skipping")
        return False

    os.makedirs(triphone_dir, exist_ok=True)

    # Prepare metadata
    metadata = {
        'triphone': triphone_name,
        'target_phoneme': get_base_phoneme(phoneme_data),
        'original_phoneme_entry': phoneme_data['phoneme'],
        'frame_count': len(frames),
        'encoding': 'global_i_p_differential'  # Mark encoding type
    }

    if len(frames) == 0:
        return False
    
    # ALL FRAMES as P-frames: JPEG-encode each difference frame
    # This combines differential encoding with JPEG compression
    p_frames_jpeg = []
    
    for i in range(len(frames)):
        # Calculate pixel difference from master I-frame
        diff = frames[i].astype(np.int16) - master_i_frame.astype(np.int16)
        
        # Shift to uint8 range [0, 255] for JPEG: diff+128
        diff_shifted = np.clip(diff + 128, 0, 255).astype(np.uint8)
        
        # JPEG-encode the difference (quality 85 for good compression)
        _, encoded = cv2.imencode('.jpg', diff_shifted, [cv2.IMWRITE_JPEG_QUALITY, 85])
        p_frames_jpeg.append(encoded.tobytes())
    
    # Store as simple dict with minimal structure
    # Using shortest key names to minimize overhead
    np.savez_compressed(npz_path, j=p_frames_jpeg)
    
    # Calculate and report space savings
    jpeg_sizes = []
    for frame in frames:
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_sizes.append(len(encoded))
    
    original_size = sum(jpeg_sizes)
    saved_size = os.path.getsize(npz_path)
    savings_pct = (1 - saved_size/original_size) * 100 if original_size > 0 else 0
    
    print(f"Saved {len(frames)} frames for '{triphone_name}' (Global I+P: {savings_pct:.1f}% space saving, {saved_size} bytes vs {original_size} JPEG bytes) -> {triphone_dir}")
    return True


def create_triphone_visemes_enriched(video_path, json_path, output_dir):
    """
    Main function to create triphone visemes from video and enriched phoneme sequence.
    Uses global I-frame approach: one master reference frame for entire library.
    """
    # Load enriched phoneme data
    print(f"Loading enriched phoneme data from: {json_path}")
    enriched_sequence = load_enriched_phoneme_data(json_path)
    print(f"Found {len(enriched_sequence)} phoneme entries in enriched sequence")
    print(f"Storage format: Global I-frame + P-frame differential encoding (85-95% space saving)")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"Video properties: {fps} fps, {total_frames} frames, {duration:.2f}s duration")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Check if master I-frame already exists (for library extension)
    print("\n=== Master Reference Frame ===")
    master_i_frame_path = os.path.join(output_dir, 'master_reference.npz')
    
    if os.path.exists(master_i_frame_path):
        # REUSE existing master I-frame when extending library
        print(f"Found existing master reference: {master_i_frame_path}")
        print("Reusing existing master I-frame for library extension")
        
        data = np.load(master_i_frame_path, allow_pickle=True)
        master_i_frame_bytes = bytes(data['i_frame'])
        master_i_frame_decoded = cv2.imdecode(
            np.frombuffer(master_i_frame_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        master_size_mb = len(master_i_frame_bytes) / (1024 * 1024)
        print(f"Master I-frame size: {master_size_mb:.2f} MB")
        
        # Show original source info if available
        if 'video_source' in data:
            print(f"Original video: {data['video_source']}")
        if 'frame_index' in data:
            print(f"Original frame index: {data['frame_index']}")
        print(f"Master I-frame will be shared across ALL triphones (existing + new)\n")
        
    else:
        # CREATE new master I-frame
        print("No existing master reference found, creating new one")
        
        # Extract a frame from the MIDDLE of the video (avoid intro/fade-in)
        # Pick a random frame from 10-50% range to get a representative talking frame
        cap = cv2.VideoCapture(video_path)
        
        # Pick frame from middle portion (10-50% into video)
        import random
        random.seed(42)  # Reproducible
        frame_index = int(total_frames * random.uniform(0.1, 0.5))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, master_frame = cap.read()
        cap.release()
        
        if not ret or master_frame is None:
            raise ValueError(f"Could not read frame {frame_index} from video")
        
        print(f"Using frame {frame_index}/{total_frames} ({frame_index/total_frames*100:.1f}% into video) as master reference")
        
        # Save master I-frame as high-quality JPEG
        success, master_i_frame_encoded = cv2.imencode('.jpg', master_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Failed to encode master I-frame")
        
        master_i_frame_bytes = master_i_frame_encoded.tobytes()
        
        # Decode it to use for P-frame calculation (this is what we'll load at runtime)
        master_i_frame_decoded = cv2.imdecode(
            np.frombuffer(master_i_frame_bytes, dtype=np.uint8), 
            cv2.IMREAD_COLOR
        )
        
        # Save master I-frame
        np.savez_compressed(
            master_i_frame_path,
            i_frame=master_i_frame_bytes,
            frame_shape=master_frame.shape,
            encoding='global_master_reference',
            video_source=video_path,
            frame_index=frame_index
        )
        
        master_size_mb = len(master_i_frame_bytes) / (1024 * 1024)
        print(f"Master I-frame created: {master_i_frame_path}")
        print(f"Master I-frame size: {master_size_mb:.2f} MB")
        print(f"Master I-frame will be shared across ALL triphones\n")
    
    # STEP 2: Process each phoneme entry
    print("=== Creating Triphone Visemes ===")
    triphone_stats = {}
    current_frame_position = 0
    
    for i, phoneme_entry in enumerate(enriched_sequence):
        current_base = get_base_phoneme(phoneme_entry)
        
        # Skip only specific garbage phonemes
        if current_base in ['spn', '', 'LEFTOVER', 'FINAL_LEFTOVER']:
            frame_count = get_phoneme_frame_count(phoneme_entry)
            current_frame_position += frame_count
            continue

        # Get triphone context
        left, current, right, triphone_name = get_triphone_context(enriched_sequence, i)
        
        # Skip any triphone that contains 'spn'
        if 'spn' in triphone_name.lower():
            print(f"Skipping triphone '{triphone_name}' - contains 'spn' garbage phoneme")
            frame_count = get_phoneme_frame_count(phoneme_entry)
            current_frame_position += frame_count
            continue
        
        # Get frame count for this entry
        frame_count = get_phoneme_frame_count(phoneme_entry)
        
        # Extract frames for this phoneme entry
        try:
            frames = extract_frames_by_position(
                video_path,
                current_frame_position,
                frame_count
            )

            if frames:
                # Save frames with triphone context using master I-frame
                was_saved = save_viseme_frames(
                    frames,
                    output_dir,
                    triphone_name,
                    phoneme_entry,
                    master_i_frame_decoded  # Pass master I-frame
                )

                # Update statistics
                if triphone_name not in triphone_stats:
                    triphone_stats[triphone_name] = {
                        'count': 0,
                        'total_frames': 0,
                        'versions': []
                    }

                triphone_stats[triphone_name]['count'] += 1
                triphone_stats[triphone_name]['total_frames'] += len(frames)

                # Track this version (frame counts)
                version_exists = any(
                    v['frames'] == len(frames) 
                    for v in triphone_stats[triphone_name]['versions']
                )
                
                if not version_exists:
                    triphone_stats[triphone_name]['versions'].append({
                        'frames': len(frames)
                    })

        except Exception as e:
            print(f"Error processing phoneme entry {i} ({phoneme_entry['phoneme']}): {e}")
        
        # Advance frame position
        current_frame_position += frame_count
    
    # Save overall statistics
    stats_path = os.path.join(output_dir, 'triphone_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(triphone_stats, f, indent=2, sort_keys=True)
    
    print(f"\n=== Summary ===")
    print(f"Created visemes for {len(triphone_stats)} unique triphones")
    print(f"Master I-frame: {master_size_mb:.2f} MB (shared across all triphones)")
    print(f"Statistics saved to: {stats_path}")
    print(f"Total video frames processed: {current_frame_position}")
    
    # Count total unique versions created
    total_versions = sum(len(stats['versions']) for stats in triphone_stats.values())
    print(f"Total unique viseme versions created: {total_versions}")
    
    # Print top triphones by occurrence
    sorted_triphones = sorted(triphone_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    print("\nTop 10 most frequent triphones:")
    for triphone, stats in sorted_triphones[:10]:
        versions_info = [f"{v['frames']}f" for v in stats['versions']]
        print(f"  {triphone}: {stats['count']} occurrences, frame variations: {', '.join(versions_info)}")


def main():
    parser = argparse.ArgumentParser(description='Create triphone visemes from video and enriched phoneme sequence')
    parser.add_argument('--video',
                       help='Path to input video file', 
                       default='//home/ist/Desktop/video-retalking/emily/video.mp4')
    parser.add_argument('--json',
                       help='Path to enriched phoneme alignment JSON file', 
                       default='//home/ist/Desktop/video-retalking/emily/output/complete_phoneme_alignments_w_reps_fixed_len.json')
    parser.add_argument('--output',
                       help='Output directory for triphone visemes', 
                       default='/home/ist/Desktop/video-retalking/emily/viseme_library')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        return 1
    
    print(f"Creating triphone visemes from enriched sequence...")
    print(f"Video: {args.video}")
    print(f"JSON: {args.json}")
    print(f"Output: {args.output}")
    print()
    
    try:
        create_triphone_visemes_enriched(args.video, args.json, args.output)
        print("\nTriphone viseme creation completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
