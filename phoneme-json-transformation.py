import json
import os
import math
from phoneme_preprocessing_utils import fill_gaps_with_silence

def enrich_sequence_with_repetitions(sequence: list, boundaries_dict: dict) -> list:
    """
    For each phoneme occurrence, split into multiple consecutive entries
    based on repetition (1, 2, or 3), dividing duration equally.
    Returns a new enriched sequence.
    """
    enriched_sequence = []
    for occurrence in sequence:
        phoneme = occurrence['phoneme']
        duration = occurrence['duration_s']
        boundaries = boundaries_dict.get(phoneme, [None, None])
        if boundaries[0] is None or boundaries[1] is None:
            repetition = 1
        elif duration < boundaries[0]:
            repetition = 1
        elif duration < boundaries[1]:
            repetition = 2
        else:
            repetition = 3

        split_duration = duration / repetition
        for i in range(repetition):
            enriched_sequence.append({
                'phoneme': phoneme,
                'start_s': occurrence['start_s'] + i * split_duration,
                'end_s': occurrence['start_s'] + (i + 1) * split_duration,
                'duration_s': split_duration
            })
    return enriched_sequence

def enrich_sequence_with_fixed_length(sequence: list, num_frames: int = 3, fps: float = 25.0) -> list:
    """
    Transform each phoneme occurrence into frame-based repetitions.
    Each phoneme occurrence is converted to total frames, then split into repetitions of num_frames each.
    Remainders are handled with special phonemes: 1 frame -> "A_", 2 frames -> "A__"
    Uses cumulative frame tracking to ensure perfect accuracy and duration preservation.
    
    Args
        sequence: List of phoneme occurrences
        num_frames: Number of frames per phoneme repetition (default 3)
        fps: Frames per second (default 25.0)
    
    Returns:
        New enriched sequence with frame-based repetitions
    """
    frame_duration = num_frames / fps  # Duration per repetition in seconds
    single_frame_duration = 1.0 / fps  # Duration of a single frame
    
    enriched_sequence = []
    
    # Calculate total original duration for final adjustment
    total_original_duration = sum(occ['duration_s'] for occ in sequence)
    
    # Cumulative tracking for perfect precision
    cumulative_duration = 0.0
    cumulative_frames_allocated = 0
    current_time = 0.0  # Track actual time progression
    
    for i, occurrence in enumerate(sequence):
        phoneme = occurrence['phoneme']
        original_duration = occurrence['duration_s']
        is_last_phoneme = (i == len(sequence) - 1)
        
        # Add this phoneme's duration to cumulative total
        cumulative_duration += original_duration
        
        # Calculate total frames that should be allocated up to this point
        cumulative_exact_frames = cumulative_duration * fps
        target_cumulative_frames = round(cumulative_exact_frames)
        
        # Frames for this specific phoneme = target total - already allocated
        frames_for_this_phoneme = target_cumulative_frames - cumulative_frames_allocated
        
        # Ensure we don't go negative (safety check)
        frames_for_this_phoneme = max(0, frames_for_this_phoneme)
        
        # Update cumulative tracking
        cumulative_frames_allocated = target_cumulative_frames
        
        # Calculate repetitions and remainders
        num_repetitions = frames_for_this_phoneme // num_frames
        remaining_frames = frames_for_this_phoneme % num_frames
        
        # Add complete repetitions
        for j in range(num_repetitions):
            # For the very last phoneme entry, adjust duration to preserve total
            if is_last_phoneme and j == num_repetitions - 1 and remaining_frames == 0:
                # This is the last phoneme entry - adjust its duration
                remaining_duration = total_original_duration - current_time
                enriched_sequence.append({
                    'phoneme': phoneme,
                    'start_s': current_time,
                    'end_s': total_original_duration,
                    'duration_s': remaining_duration
                })
                current_time = total_original_duration
            else:
                enriched_sequence.append({
                    'phoneme': phoneme,
                    'start_s': current_time,
                    'end_s': current_time + frame_duration,
                    'duration_s': frame_duration
                })
                current_time += frame_duration
        
        # Handle remaining frames immediately after the repetitions
        if remaining_frames == 1:
            # For last remainder frame, adjust duration to preserve total
            if is_last_phoneme:
                remaining_duration = total_original_duration - current_time
                enriched_sequence.append({
                    'phoneme': f'{phoneme}_',
                    'start_s': current_time,
                    'end_s': total_original_duration,
                    'duration_s': remaining_duration
                })
                current_time = total_original_duration
            else:
                enriched_sequence.append({
                    'phoneme': f'{phoneme}_',
                    'start_s': current_time,
                    'end_s': current_time + single_frame_duration,
                    'duration_s': single_frame_duration
                })
                current_time += single_frame_duration
        elif remaining_frames == 2:
            # For last remainder frames, adjust duration to preserve total
            if is_last_phoneme:
                remaining_duration = total_original_duration - current_time
                enriched_sequence.append({
                    'phoneme': f'{phoneme}__',
                    'start_s': current_time,
                    'end_s': total_original_duration,
                    'duration_s': remaining_duration
                })
                current_time = total_original_duration
            else:
                enriched_sequence.append({
                    'phoneme': f'{phoneme}__',
                    'start_s': current_time,
                    'end_s': current_time + 2 * single_frame_duration,
                    'duration_s': 2 * single_frame_duration
                })
                current_time += 2 * single_frame_duration
    
    return enriched_sequence

def verify_enriched_sequence(original_sequence: list, enriched_sequence: list) -> None:
    """
    Performs sanity checks:
    - Total duration is preserved
    - Number of phonemes is >= original
    - No negative durations
    Prints results of checks.
    """
    orig_total = sum(p['duration_s'] for p in original_sequence)
    enrich_total = sum(p['duration_s'] for p in enriched_sequence)
    print(f"Original total duration: {orig_total:.6f}")
    print(f"Enriched total duration: {enrich_total:.6f}")
    if abs(orig_total - enrich_total) < 1e-6:
        print("PASS: Total duration preserved.")
    else:
        print("FAIL: Total duration mismatch.")

    if len(enriched_sequence) >= len(original_sequence):
        print("PASS: Enriched sequence has equal or more phonemes.")
    else:
        print("FAIL: Enriched sequence has fewer phonemes.")

    negative_durations = [p for p in enriched_sequence if p['duration_s'] < 0]
    if not negative_durations:
        print("PASS: No negative durations.")
    else:
        print(f"FAIL: Found {len(negative_durations)} negative durations.")

def save_json(data, filename, transform_type="clusters", output_dir=None):
    """
    Saves data as a JSON file in the specified output directory.
    Adds transform type suffix to filename.
    
    Args:
        data: JSON data to save
        filename: Base filename
        transform_type: Type of transformation ("clusters" or "fixed-length")
        output_dir: Directory to save the file (if None, uses current directory)
    """
    # Add suffix based on transform type
    name, ext = os.path.splitext(filename)
    if transform_type == "fixed-length":
        filename = f"{name}_fixed_len{ext}"
    else:
        filename = f"{name}_clusters{ext}"
    
    # Construct full output path
    if output_dir:
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich viseme phoneme sequences in JSON using phoneme boundaries or fixed-length segments.")
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--output_json", help="Path to output JSON file (optional - will auto-generate from input filename)")
    parser.add_argument("--output_dir", help="Directory to save output file (default: same as input file)")
    parser.add_argument("--transform-type", choices=["clusters", "fixed-length"], default="clusters", 
                        help="Transformation type: 'clusters' (default) uses phoneme boundaries, 'fixed-length' uses fixed frame duration")
    parser.add_argument("--boundaries_json", help="Path to phoneme boundaries JSON file (required for clusters mode)")
    parser.add_argument("--num-frames", type=int, default=3, help="Number of frames per segment for fixed-length mode (default: 3)")
    parser.add_argument("--fps", type=float, default=25.0, help="Frames per second for fixed-length mode (default: 25.0)")
    args = parser.parse_args()

    # Validate arguments based on transform type
    if args.transform_type == "clusters" and not args.boundaries_json:
        parser.error("--boundaries_json is required when using 'clusters' transform type")

    # Load data
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    phoneme_boundaries = None
    if args.transform_type == "clusters":
        with open(args.boundaries_json, "r", encoding="utf-8") as f:
            phoneme_boundaries = json.load(f)

    # Transform viseme_phonemes for each file
    for file_key, file_data in data.get("files", {}).items():
        if "viseme_phonemes" in file_data:
            original_sequence = file_data["viseme_phonemes"]
            
            print(f"\nProcessing {file_key} with {args.transform_type} transform...")
            
            if args.transform_type == "clusters":
                enriched_sequence = enrich_sequence_with_repetitions(fill_gaps_with_silence(original_sequence), phoneme_boundaries or {})
            else:  # fixed-length
                enriched_sequence = enrich_sequence_with_fixed_length(fill_gaps_with_silence(original_sequence), args.num_frames, args.fps)
                print(f"Using {args.num_frames} frames per segment at {args.fps} FPS (segment duration: {args.num_frames/args.fps:.3f}s)")
            
            verify_enriched_sequence(fill_gaps_with_silence(original_sequence), enriched_sequence)
            file_data["enriched_sequence"] = enriched_sequence

    # Auto-generate output filename if not provided
    if args.output_json:
        output_filename = os.path.basename(args.output_json)
    else:
        # Create output filename from input: input_name_w_reps_[transform_type].json
        input_name, input_ext = os.path.splitext(os.path.basename(args.input_json))
        output_filename = f"{input_name}_w_reps.json"

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as input file
        output_dir = os.path.dirname(args.input_json)
    
    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save output (save_json will add transform type suffix)
    save_json(data, output_filename, args.transform_type, output_dir)

# python phoneme-json-transformation.py --input_json /home/ist/Desktop/synth_audios/master-audio/output_dict_generated/complete_phoneme_alignments.json --output_json /home/ist/Desktop/synth_audios/master-audio/output_dict_generated/complete_phoneme_alignments_w_reps.json --transform-type fixed-length 
# python phoneme-json-transformation.py --input_json /home/ist/Desktop/video-retalking/Actor_1_Female/output/complete_phoneme_alignments.json --transform-type fixed-length