#!/usr/bin/env python3
"""
Section 4: Master Video Comparison
Analyzes triphone coverage, viseme selection accuracy, and temporal alignment.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_viseme_library(viseme_library_path):
    """Analyze the triphone library structure and coverage."""
    print("=" * 60)
    print("SECTION 4.1: TRIPHONE LIBRARY ANALYSIS")
    print("=" * 60)
    
    import numpy as np
    
    triphone_dirs = [d for d in os.listdir(viseme_library_path) 
                     if os.path.isdir(os.path.join(viseme_library_path, d))]
    
    # Parse triphone structure
    phoneme_set = set()
    
    # Count frames from NPZ files (P-frames)
    frames_per_triphone = {}
    
    for triphone in triphone_dirs:
        triphone_path = os.path.join(viseme_library_path, triphone)
        
        # Try p_frames.npz first, then frames.npz
        npz_file = os.path.join(triphone_path, "p_frames.npz")
        if not os.path.exists(npz_file):
            npz_file = os.path.join(triphone_path, "frames.npz")
        
        if os.path.exists(npz_file):
            try:
                data = np.load(npz_file, allow_pickle=True)
                # Find the encoded frames array (key might be 'p_frames', 'encoded_frames', 'j', etc.)
                for key in ['p_frames', 'encoded_frames', 'j']:
                    if key in data:
                        frames_per_triphone[triphone] = len(data[key])
                        break
                else:
                    # Fallback: count any array
                    for key in data.keys():
                        frames_per_triphone[triphone] = len(data[key])
                        break
            except:
                frames_per_triphone[triphone] = 0
        else:
            frames_per_triphone[triphone] = 0
        
        # Extract unique phonemes from triphone name
        parts = triphone.replace('__', '_').split('_') if '_' in triphone else [triphone]
        for part in parts:
            if part:
                phoneme_set.add(part)
    
    # Calculate statistics
    total_triphones = len(triphone_dirs)
    total_frames = sum(frames_per_triphone.values())
    avg_frames = total_frames / total_triphones if total_triphones > 0 else 0
    min_frames = min(frames_per_triphone.values()) if frames_per_triphone else 0
    max_frames = max(frames_per_triphone.values()) if frames_per_triphone else 0
    
    # Frame count distribution
    frame_distribution = defaultdict(int)
    for count in frames_per_triphone.values():
        frame_distribution[count] += 1
    
    print(f"\n📚 Library Overview:")
    print(f"  • Total triphones: {total_triphones:,}")
    print(f"  • Total frames: {total_frames:,}")
    print(f"  • Unique phonemes: {len(phoneme_set)}")
    print(f"  • Avg frames/triphone: {avg_frames:.1f}")
    print(f"  • Frame range: {min_frames} - {max_frames}")
    
    print(f"\n📊 Frame Count Distribution:")
    for frame_count in sorted(frame_distribution.keys()):
        count = frame_distribution[frame_count]
        pct = count / total_triphones * 100
        bar = "█" * int(pct / 2)
        print(f"  {frame_count} frames: {count:4d} ({pct:5.1f}%) {bar}")
    
    return {
        "total_triphones": total_triphones,
        "total_frames": total_frames,
        "unique_phonemes": len(phoneme_set),
        "phoneme_list": sorted(list(phoneme_set)),
        "avg_frames_per_triphone": avg_frames,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "frame_distribution": dict(frame_distribution)
    }


def run_coverage_test(test_sentences):
    """Run video generation on test sentences and collect coverage stats."""
    print("\n" + "=" * 60)
    print("SECTION 4.2: TRIPHONE COVERAGE TEST")
    print("=" * 60)
    
    from actor_manager import ActorManager
    from tts_engine import text_to_speech
    from fast_phoneme_extractor import FastPhonemeExtractor
    from video_generator import generate_video
    import tempfile
    import subprocess
    
    actor_manager = ActorManager()
    actor_id = "actor_5"
    actor_data = actor_manager.get_actor(actor_id)
    
    extractor = FastPhonemeExtractor()
    
    all_coverage_stats = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Test {i}/{len(test_sentences)}: \"{sentence[:50]}...\" ---")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Generate TTS
            audio_path = tmpdir / "audio.mp3"
            text_to_speech(sentence, str(audio_path))
            
            # Save text file for MFA
            text_path = tmpdir / "text.txt"
            with open(text_path, 'w') as f:
                f.write(sentence)
            
            # Run MFA alignment
            phoneme_result = extractor.extract_from_files(audio_path, text_path)
            phoneme_json = tmpdir / "phonemes.json"
            extractor.save_result(phoneme_result, phoneme_json)
            
            # Run transformation
            transform_script = Path(__file__).parent.parent / "phoneme-json-transformation.py"
            subprocess.run([
                sys.executable, str(transform_script),
                "--input_json", str(phoneme_json),
                "--transform-type", "fixed-length"
            ], capture_output=True, check=True)
            
            enriched_json = tmpdir / "phonemes_w_reps_fixed_len.json"
            
            # Generate video and capture stats
            output_path = tmpdir / "output.mp4"
            
            # Capture stdout to parse stats
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                generate_video(
                    triphone_visemes_dir=str(actor_data.viseme_path),
                    json_path=str(enriched_json),
                    audio_path=str(audio_path),
                    output_path=str(output_path)
                )
            
            output = f.getvalue()
            
            # Parse stats from output
            stats = parse_coverage_stats(output)
            stats["sentence"] = sentence
            stats["word_count"] = len(sentence.split())
            all_coverage_stats.append(stats)
            
            print(f"  Exact: {stats.get('exact', 0)}/{stats.get('total', 0)} "
                  f"({stats.get('exact_pct', 0):.1f}%), "
                  f"Fallback: {stats.get('fallback', 0)}")
            stats["word_count"] = len(sentence.split())
            all_coverage_stats.append(stats)
            
            print(f"  Exact: {stats.get('exact', 0)}/{stats.get('total', 0)} "
                  f"({stats.get('exact_pct', 0):.1f}%), "
                  f"Fallback: {stats.get('fallback', 0)}")
    
    # Aggregate statistics
    total_phonemes = sum(s.get('total', 0) for s in all_coverage_stats)
    total_exact = sum(s.get('exact', 0) for s in all_coverage_stats)
    total_fallback = sum(s.get('fallback', 0) for s in all_coverage_stats)
    
    aggregate = {
        "test_sentences": len(test_sentences),
        "total_phonemes": total_phonemes,
        "exact_matches": total_exact,
        "fallback_matches": total_fallback,
        "exact_match_rate": total_exact / total_phonemes * 100 if total_phonemes > 0 else 0,
        "fallback_rate": total_fallback / total_phonemes * 100 if total_phonemes > 0 else 0,
        "per_sentence": all_coverage_stats
    }
    
    print(f"\n📈 Aggregate Coverage:")
    print(f"  • Total phonemes: {total_phonemes}")
    print(f"  • Exact triphone matches: {total_exact} ({aggregate['exact_match_rate']:.1f}%)")
    print(f"  • Fallback matches: {total_fallback} ({aggregate['fallback_rate']:.1f}%)")
    
    return aggregate


def parse_coverage_stats(output):
    """Parse coverage statistics from video generator output."""
    stats = {"total": 0, "exact": 0, "fallback": 0, "skipped": 0, "silence": 0}
    
    for line in output.split('\n'):
        if "Total phonemes processed:" in line:
            try:
                stats["total"] = int(line.split(':')[1].strip())
            except:
                pass
        elif "Exact triphone matches:" in line:
            try:
                # Format: "Exact triphone matches: 25 (96.2%)"
                parts = line.split(':')[1].strip()
                num = parts.split('(')[0].strip()
                stats["exact"] = int(num)
                pct = parts.split('(')[1].rstrip('%)').strip()
                stats["exact_pct"] = float(pct)
            except:
                pass
        elif "Fallback matches:" in line:
            try:
                parts = line.split(':')[1].strip()
                num = parts.split('(')[0].strip()
                stats["fallback"] = int(num)
            except:
                pass
        elif "Skipped phonemes:" in line:
            try:
                stats["skipped"] = int(line.split(':')[1].strip())
            except:
                pass
        elif "Silence/special tokens:" in line:
            try:
                stats["silence"] = int(line.split(':')[1].strip())
            except:
                pass
    
    return stats


def analyze_temporal_alignment(json_path, video_fps=50):
    """Analyze temporal alignment between MFA phoneme boundaries and video frames."""
    print("\n" + "=" * 60)
    print("SECTION 4.3: TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 60)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    enriched_sequence = data['files']['audio']['enriched_sequence']
    
    # Calculate timing precision
    frame_duration = 1.0 / video_fps
    
    alignment_errors = []
    phoneme_durations = []
    
    for entry in enriched_sequence:
        duration = entry['duration_s']
        start = entry['start_s']
        end = entry['end_s']
        
        phoneme_durations.append(duration)
        
        # Check if boundaries align to frame boundaries
        start_frame = start * video_fps
        end_frame = end * video_fps
        
        start_error = abs(start_frame - round(start_frame)) / video_fps
        end_error = abs(end_frame - round(end_frame)) / video_fps
        
        alignment_errors.append({
            "phoneme": entry.get('phoneme', entry.get('current', 'unknown')),
            "start_error_ms": start_error * 1000,
            "end_error_ms": end_error * 1000,
            "duration_ms": duration * 1000
        })
    
    avg_start_error = sum(e['start_error_ms'] for e in alignment_errors) / len(alignment_errors)
    avg_end_error = sum(e['end_error_ms'] for e in alignment_errors) / len(alignment_errors)
    avg_duration = sum(phoneme_durations) / len(phoneme_durations) * 1000
    
    print(f"\n⏱️ Temporal Precision (at {video_fps} FPS):")
    print(f"  • Frame duration: {frame_duration*1000:.2f} ms")
    print(f"  • Avg phoneme duration: {avg_duration:.1f} ms")
    print(f"  • Avg start boundary error: {avg_start_error:.3f} ms")
    print(f"  • Avg end boundary error: {avg_end_error:.3f} ms")
    print(f"  • Max alignment error: {max(e['start_error_ms'] for e in alignment_errors):.3f} ms")
    
    # Duration distribution
    duration_dist = {
        "<50ms": sum(1 for d in phoneme_durations if d < 0.05),
        "50-100ms": sum(1 for d in phoneme_durations if 0.05 <= d < 0.1),
        "100-200ms": sum(1 for d in phoneme_durations if 0.1 <= d < 0.2),
        ">200ms": sum(1 for d in phoneme_durations if d >= 0.2)
    }
    
    print(f"\n📊 Phoneme Duration Distribution:")
    for range_label, count in duration_dist.items():
        pct = count / len(phoneme_durations) * 100
        print(f"  {range_label}: {count} ({pct:.1f}%)")
    
    return {
        "video_fps": video_fps,
        "frame_duration_ms": frame_duration * 1000,
        "avg_phoneme_duration_ms": avg_duration,
        "avg_start_error_ms": avg_start_error,
        "avg_end_error_ms": avg_end_error,
        "max_alignment_error_ms": max(e['start_error_ms'] for e in alignment_errors),
        "duration_distribution": duration_dist,
        "total_phonemes": len(alignment_errors)
    }


def run_same_vs_novel_test(actor_data):
    """Compare quality metrics for same-text (training) vs novel-text generation."""
    print("\n" + "=" * 60)
    print("SECTION 4.4: SAME-TEXT vs NOVEL-TEXT COMPARISON")
    print("=" * 60)
    
    # For this test, we'd need to know which text was used to create the viseme library
    # Since we don't have that info, we'll use coverage rate as proxy
    
    print("\n📝 Coverage Rate Analysis:")
    print("  • High coverage (>95%) suggests text similar to training data")
    print("  • Lower coverage (<90%) suggests novel phoneme combinations")
    print("  • Fallback matches preserve quality but may have subtle differences")
    
    # We can infer from the library what phoneme combinations exist
    viseme_library = str(actor_data.viseme_path)
    triphones = set(os.listdir(viseme_library))
    
    # Sample some common English triphones to check coverage
    common_triphones = [
        "sil_ð_ə",  # "the"
        "sil_h_ɛ",  # "he"
        "ə_n_d",    # "and"
        "t_u_sil",  # "to"
        "ɪ_z_sil",  # "is"
    ]
    
    found = sum(1 for t in common_triphones if t in triphones)
    
    print(f"\n🔍 Common Triphone Spot Check:")
    print(f"  • Checked {len(common_triphones)} common English triphones")
    print(f"  • Found in library: {found}/{len(common_triphones)}")
    
    return {
        "note": "Same-text vs novel-text comparison requires knowing training data",
        "common_triphone_coverage": found / len(common_triphones) * 100,
        "recommendation": "Use coverage rate >90% as quality indicator"
    }


def main():
    print("=" * 60)
    print("SECTION 4: MASTER VIDEO COMPARISON")
    print("Triphone Coverage & Viseme Selection Analysis")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    results_dir = Path.home() / "Desktop" / "lip_sync_comparison"
    results_dir.mkdir(exist_ok=True)
    
    # Load actor data
    sys.path.insert(0, str(base_dir))
    from actor_manager import ActorManager
    actor_manager = ActorManager()
    actor_id = "actor_5"
    actor_data = actor_manager.get_actor(actor_id)
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actor_id": actor_id
    }
    
    # 4.1: Library Analysis
    library_stats = analyze_viseme_library(str(actor_data.viseme_path))
    results["library_analysis"] = library_stats
    
    # 4.2: Coverage Test
    test_sentences = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "I think therefore I am.",
        "To be or not to be, that is the question.",
        "The rain in Spain falls mainly on the plain.",
        "How much wood would a woodchuck chuck?",
    ]
    
    coverage_stats = run_coverage_test(test_sentences)
    results["coverage_test"] = coverage_stats
    
    # 4.3: Temporal Alignment (use one of the test outputs)
    # We'll generate one more time to get the JSON
    import tempfile
    import subprocess
    from tts_engine import text_to_speech
    from fast_phoneme_extractor import FastPhonemeExtractor
    
    extractor = FastPhonemeExtractor()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_path = tmpdir / "audio.mp3"
        text_to_speech("Hello, how are you today?", str(audio_path))
        
        text_path = tmpdir / "text.txt"
        with open(text_path, 'w') as f:
            f.write("Hello, how are you today?")
        
        phoneme_result = extractor.extract_from_files(audio_path, text_path)
        phoneme_json = tmpdir / "phonemes.json"
        extractor.save_result(phoneme_result, phoneme_json)
        
        # Run transformation to get enriched JSON
        transform_script = Path(__file__).parent.parent / "phoneme-json-transformation.py"
        subprocess.run([
            sys.executable, str(transform_script),
            "--input_json", str(phoneme_json),
            "--transform-type", "fixed-length"
        ], capture_output=True, check=True)
        
        enriched_json = tmpdir / "phonemes_w_reps_fixed_len.json"
        
        temporal_stats = analyze_temporal_alignment(str(enriched_json))
        results["temporal_alignment"] = temporal_stats
    
    # 4.4: Same vs Novel Text
    same_novel_analysis = run_same_vs_novel_test(actor_data)
    results["same_vs_novel"] = same_novel_analysis
    
    # Summary and Verdicts
    print("\n" + "=" * 60)
    print("SECTION 4 SUMMARY")
    print("=" * 60)
    
    exact_rate = coverage_stats["exact_match_rate"]
    
    # Determine verdicts
    if exact_rate >= 95:
        coverage_verdict = "Excellent"
    elif exact_rate >= 90:
        coverage_verdict = "Very Good"
    elif exact_rate >= 80:
        coverage_verdict = "Good"
    else:
        coverage_verdict = "Needs Improvement"
    
    alignment_error = temporal_stats["avg_start_error_ms"]
    if alignment_error < 1.0:
        alignment_verdict = "Excellent (<1ms)"
    elif alignment_error < 5.0:
        alignment_verdict = "Very Good (<5ms)"
    else:
        alignment_verdict = "Good"
    
    results["summary"] = {
        "triphone_library_size": library_stats["total_triphones"],
        "exact_match_rate": exact_rate,
        "coverage_verdict": coverage_verdict,
        "temporal_alignment_error_ms": alignment_error,
        "alignment_verdict": alignment_verdict
    }
    
    print(f"\n| Metric | Value | Verdict |")
    print(f"|--------|-------|---------|")
    print(f"| Library Size | {library_stats['total_triphones']:,} triphones | Large |")
    print(f"| Exact Match Rate | {exact_rate:.1f}% | {coverage_verdict} |")
    print(f"| Fallback Rate | {coverage_stats['fallback_rate']:.1f}% | - |")
    print(f"| Temporal Error | {alignment_error:.3f}ms | {alignment_verdict} |")
    print(f"| Unique Phonemes | {library_stats['unique_phonemes']} | Comprehensive |")
    
    # Save results
    output_path = results_dir / "section4_master_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
