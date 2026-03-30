#!/usr/bin/env python3
"""
Performance Benchmark for Triphone-Based Lip-Sync Pipeline

Measures:
- 2.1 Latency Metrics (Text-to-First-Frame, Per-Frame, End-to-End)
- 2.2 Resource Metrics (Peak Memory, Model Size, FPS, CPU-only)
- 2.3 Scalability Metrics (Cold Start, Warm Inference)
"""

import os
import sys
import json
import time
import tracemalloc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_directory_size_mb(path: str) -> float:
    """Get total size of directory in MB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / 1024 / 1024


class PerformanceBenchmark:
    """Benchmark the triphone-based lip-sync pipeline"""
    
    def __init__(self, actor_id: str = "actor_5"):
        self.actor_id = actor_id
        self.results = {}
        
        # Import modules (measure import time)
        self.modules_loaded = False
        
    def _load_modules(self):
        """Load pipeline modules"""
        if self.modules_loaded:
            return
            
        global ActorManager, FastPhonemeExtractor, generate_video
        
        from actor_manager import ActorManager
        from fast_phoneme_extractor import FastPhonemeExtractor
        from video_generator import generate_video
        
        self.modules_loaded = True
    
    def measure_model_sizes(self) -> Dict:
        """Measure size of models and assets"""
        print("\n=== Measuring Model Sizes ===")
        
        sizes = {}
        base_path = Path(__file__).parent.parent
        
        # Viseme library size
        viseme_path = base_path / "actors" / self.actor_id / "triphone_visemes"
        if viseme_path.exists():
            sizes['viseme_library_mb'] = get_directory_size_mb(str(viseme_path))
            print(f"  Viseme Library: {sizes['viseme_library_mb']:.2f} MB")
        
        # MFA models size
        mfa_paths = [
            base_path / "MFA",
            base_path / "Montreal-Forced-Aligner",
            Path.home() / "Documents" / "MFA"
        ]
        for mfa_path in mfa_paths:
            if mfa_path.exists():
                sizes['mfa_models_mb'] = get_directory_size_mb(str(mfa_path))
                print(f"  MFA Models: {sizes['mfa_models_mb']:.2f} MB")
                break
        
        # Total runtime assets
        sizes['total_assets_mb'] = sum(v for k, v in sizes.items() if k.endswith('_mb'))
        print(f"  Total Assets: {sizes['total_assets_mb']:.2f} MB")
        
        return sizes
    
    def measure_cold_start(self) -> Dict:
        """Measure cold start time (loading all models)"""
        print("\n=== Measuring Cold Start Time ===")
        
        # Clear any cached modules
        modules_to_clear = [
            'actor_manager', 'fast_phoneme_extractor', 'video_generator',
            'tts_engine', 'create_video_from_triphones'
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Measure memory before
        mem_before = get_memory_usage_mb()
        
        # Time module imports
        start_import = time.time()
        self._load_modules()
        import_time = time.time() - start_import
        
        # Time ActorManager initialization
        start_actor = time.time()
        from actor_manager import ActorManager
        actor_manager = ActorManager()
        actor_time = time.time() - start_actor
        
        # Time MFA model loading
        start_mfa = time.time()
        from fast_phoneme_extractor import FastPhonemeExtractor
        extractor = FastPhonemeExtractor()
        mfa_time = time.time() - start_mfa
        
        # Memory after loading
        mem_after = get_memory_usage_mb()
        
        results = {
            'import_time_s': import_time,
            'actor_manager_load_s': actor_time,
            'mfa_model_load_s': mfa_time,
            'total_cold_start_s': import_time + actor_time + mfa_time,
            'memory_increase_mb': mem_after - mem_before
        }
        
        print(f"  Module Import: {import_time:.3f}s")
        print(f"  Actor Manager: {actor_time:.3f}s")
        print(f"  MFA Models: {mfa_time:.3f}s")
        print(f"  Total Cold Start: {results['total_cold_start_s']:.3f}s")
        print(f"  Memory Increase: {results['memory_increase_mb']:.2f} MB")
        
        # Store for warm inference
        self.actor_manager = actor_manager
        self.extractor = extractor
        
        return results
    
    def measure_inference_latency(self, test_texts: List[str] = None) -> Dict:
        """Measure inference latency for different text lengths"""
        print("\n=== Measuring Inference Latency ===")
        
        if test_texts is None:
            test_texts = [
                "Hello.",  # Short (1 word)
                "Hello, how are you today?",  # Medium (5 words)
                "The quick brown fox jumps over the lazy dog near the riverbank.",  # Long (12 words)
            ]
        
        results = {'tests': []}
        
        from actor_manager import ActorManager
        from video_generator import generate_video
        
        if not hasattr(self, 'actor_manager'):
            self.actor_manager = ActorManager()
        if not hasattr(self, 'extractor'):
            from fast_phoneme_extractor import FastPhonemeExtractor
            self.extractor = FastPhonemeExtractor()
        
        actor = self.actor_manager.get_actor(self.actor_id)
        
        for i, text in enumerate(test_texts):
            print(f"\n  Test {i+1}: '{text[:50]}...' ({len(text.split())} words)")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                
                # Track memory
                tracemalloc.start()
                mem_start = get_memory_usage_mb()
                
                test_result = {
                    'text': text,
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }
                
                # Step 1: TTS
                tts_start = time.time()
                audio_file = tmpdir / "audio.mp3"
                text_file = tmpdir / "text.txt"
                with open(text_file, 'w') as f:
                    f.write(text)
                self.actor_manager.generate_actor_voice(self.actor_id, text, str(audio_file))
                test_result['tts_time_s'] = time.time() - tts_start
                print(f"    TTS: {test_result['tts_time_s']:.3f}s")
                
                # Step 2: MFA Alignment
                mfa_start = time.time()
                phoneme_result = self.extractor.extract_from_files(audio_file, text_file)
                phoneme_json = tmpdir / "phonemes.json"
                self.extractor.save_result(phoneme_result, phoneme_json)
                
                # Run transformation
                transform_script = Path(__file__).parent.parent / "phoneme-json-transformation.py"
                subprocess.run([
                    sys.executable, str(transform_script),
                    "--input_json", str(phoneme_json),
                    "--transform-type", "fixed-length"
                ], capture_output=True, check=True)
                test_result['mfa_time_s'] = time.time() - mfa_start
                print(f"    MFA: {test_result['mfa_time_s']:.3f}s")
                
                # Step 3: Video Generation
                video_start = time.time()
                enriched_json = tmpdir / "phonemes_w_reps_fixed_len.json"
                output_video = tmpdir / "output.mp4"
                
                video_file = generate_video(
                    triphone_visemes_dir=str(actor.viseme_path),
                    json_path=str(enriched_json),
                    audio_path=str(audio_file),
                    output_path=str(output_video)
                )
                test_result['video_gen_time_s'] = time.time() - video_start
                print(f"    Video Gen: {test_result['video_gen_time_s']:.3f}s")
                
                # Get video info
                if Path(video_file).exists():
                    import cv2
                    cap = cv2.VideoCapture(video_file)
                    test_result['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_result['fps'] = cap.get(cv2.CAP_PROP_FPS)
                    test_result['duration_s'] = test_result['frame_count'] / test_result['fps'] if test_result['fps'] > 0 else 0
                    cap.release()
                
                # Memory tracking
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                test_result['peak_memory_mb'] = peak / 1024 / 1024
                
                # Calculate derived metrics
                test_result['total_time_s'] = (
                    test_result['tts_time_s'] + 
                    test_result['mfa_time_s'] + 
                    test_result['video_gen_time_s']
                )
                test_result['text_to_first_frame_s'] = (
                    test_result['tts_time_s'] + 
                    test_result['mfa_time_s'] + 
                    0.1  # Approximate time to write first frame
                )
                
                if test_result.get('frame_count', 0) > 0:
                    test_result['per_frame_time_ms'] = (
                        test_result['video_gen_time_s'] / test_result['frame_count'] * 1000
                    )
                    test_result['effective_fps'] = (
                        test_result['frame_count'] / test_result['video_gen_time_s']
                    )
                
                print(f"    Total: {test_result['total_time_s']:.3f}s")
                print(f"    Frames: {test_result.get('frame_count', 'N/A')}")
                print(f"    Effective FPS: {test_result.get('effective_fps', 0):.1f}")
                
                results['tests'].append(test_result)
        
        # Compute averages
        if results['tests']:
            results['avg_tts_time_s'] = sum(t['tts_time_s'] for t in results['tests']) / len(results['tests'])
            results['avg_mfa_time_s'] = sum(t['mfa_time_s'] for t in results['tests']) / len(results['tests'])
            results['avg_video_gen_time_s'] = sum(t['video_gen_time_s'] for t in results['tests']) / len(results['tests'])
            results['avg_total_time_s'] = sum(t['total_time_s'] for t in results['tests']) / len(results['tests'])
            
            fps_values = [t.get('effective_fps', 0) for t in results['tests'] if t.get('effective_fps')]
            if fps_values:
                results['avg_effective_fps'] = sum(fps_values) / len(fps_values)
        
        return results
    
    def measure_warm_inference(self, num_runs: int = 3) -> Dict:
        """Measure warm inference time (models already loaded)"""
        print(f"\n=== Measuring Warm Inference ({num_runs} runs) ===")
        
        test_text = "Hello, how are you today?"
        times = []
        
        from video_generator import generate_video
        
        if not hasattr(self, 'actor_manager'):
            from actor_manager import ActorManager
            self.actor_manager = ActorManager()
        if not hasattr(self, 'extractor'):
            from fast_phoneme_extractor import FastPhonemeExtractor
            self.extractor = FastPhonemeExtractor()
        
        actor = self.actor_manager.get_actor(self.actor_id)
        
        for run in range(num_runs):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                
                start = time.time()
                
                # TTS
                audio_file = tmpdir / "audio.mp3"
                text_file = tmpdir / "text.txt"
                with open(text_file, 'w') as f:
                    f.write(test_text)
                self.actor_manager.generate_actor_voice(self.actor_id, test_text, str(audio_file))
                
                # MFA
                phoneme_result = self.extractor.extract_from_files(audio_file, text_file)
                phoneme_json = tmpdir / "phonemes.json"
                self.extractor.save_result(phoneme_result, phoneme_json)
                
                transform_script = Path(__file__).parent.parent / "phoneme-json-transformation.py"
                subprocess.run([
                    sys.executable, str(transform_script),
                    "--input_json", str(phoneme_json),
                    "--transform-type", "fixed-length"
                ], capture_output=True, check=True)
                
                # Video
                enriched_json = tmpdir / "phonemes_w_reps_fixed_len.json"
                output_video = tmpdir / "output.mp4"
                generate_video(
                    triphone_visemes_dir=str(actor.viseme_path),
                    json_path=str(enriched_json),
                    audio_path=str(audio_file),
                    output_path=str(output_video)
                )
                
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Run {run+1}: {elapsed:.3f}s")
        
        results = {
            'test_text': test_text,
            'num_runs': num_runs,
            'times_s': times,
            'mean_time_s': sum(times) / len(times),
            'min_time_s': min(times),
            'max_time_s': max(times)
        }
        
        print(f"  Mean: {results['mean_time_s']:.3f}s")
        print(f"  Min: {results['min_time_s']:.3f}s, Max: {results['max_time_s']:.3f}s")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("="*60)
        print("TRIPHONE LIP-SYNC PERFORMANCE BENCHMARK")
        print("="*60)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'actor_id': self.actor_id,
            'system_info': {
                'cpu_count': os.cpu_count(),
                'total_ram_gb': psutil.virtual_memory().total / 1024**3
            }
        }
        
        # Model sizes
        results['model_sizes'] = self.measure_model_sizes()
        
        # Cold start
        results['cold_start'] = self.measure_cold_start()
        
        # Inference latency
        results['inference'] = self.measure_inference_latency()
        
        # Warm inference
        results['warm_inference'] = self.measure_warm_inference(num_runs=3)
        
        # Summary
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        results['summary'] = {
            'total_assets_mb': results['model_sizes'].get('total_assets_mb', 0),
            'cold_start_s': results['cold_start']['total_cold_start_s'],
            'warm_inference_s': results['warm_inference']['mean_time_s'],
            'avg_effective_fps': results['inference'].get('avg_effective_fps', 0),
            'gpu_required': False,
            'cpu_only': True
        }
        
        print(f"\n  Total Assets Size: {results['summary']['total_assets_mb']:.2f} MB")
        print(f"  Cold Start Time: {results['summary']['cold_start_s']:.2f}s")
        print(f"  Warm Inference Time: {results['summary']['warm_inference_s']:.2f}s")
        print(f"  Effective FPS: {results['summary']['avg_effective_fps']:.1f}")
        print(f"  GPU Required: {results['summary']['gpu_required']}")
        
        # Comparison context
        print("\n  --- Comparison with Neural Methods ---")
        print("  | Method    | VRAM    | FPS  | GPU Required |")
        print("  |-----------|---------|------|--------------|")
        print("  | Wav2Lip   | ~2 GB   | ~25  | Yes          |")
        print("  | SadTalker | ~4 GB   | ~3   | Yes          |")
        print("  | RealTalk  | ~8 GB   | ~0.5 | Yes          |")
        print(f"  | Ours      | 0 GB    | {results['summary']['avg_effective_fps']:.0f}   | No           |")
        
        print("="*60)
        
        self.results = results
        return results
    
    def save_results(self, output_path: str = None):
        """Save benchmark results to JSON"""
        if output_path is None:
            output_path = Path.home() / "Desktop" / "lip_sync_comparison" / "performance_results.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    # Run benchmark
    benchmark = PerformanceBenchmark(actor_id="actor_5")
    results = benchmark.run_full_benchmark()
    benchmark.save_results()


if __name__ == "__main__":
    main()
