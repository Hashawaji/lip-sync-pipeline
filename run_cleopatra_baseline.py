"""
Standalone script: run the full FastLipSynth pipeline on actor_1 (Cleopatra)
with a fixed ~15 s sentence. Produces output_video.mp4 and its audio for
subsequent baseline comparison.
"""
import sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TEXT = (
    "Greetings and welcome, honored guests of the royal court. "
    "Today I shall speak of wisdom, power, and the ancient secrets that have "
    "guided my kingdom for generations. Let the ceremony begin, and may "
    "fortune favor our gathering this glorious afternoon."
)
OUTPUT_NAME = "cleopatra_baseline_15s"
ACTOR_ID    = "actor_1"

def main():
    out_dir = ROOT / "outputs" / OUTPUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    text_file  = out_dir / "text.txt"
    audio_file = out_dir / "audio.mp3"
    phoneme_json = out_dir / "complete_phoneme_alignments.json"
    phoneme_json_fixed = out_dir / "complete_phoneme_alignments_w_reps_fixed_len.json"
    video_file = out_dir / "output_video.mp4"

    text_file.write_text(TEXT)
    print(f"[1/5] wrote text ({len(TEXT)} chars)")

    # Step 2: TTS via actor_manager
    t0 = time.time()
    from actor_manager import ActorManager
    am = ActorManager()
    am.generate_actor_voice(ACTOR_ID, TEXT, str(audio_file))
    print(f"[2/5] audio generated in {time.time()-t0:.1f}s -> {audio_file}")

    # Step 3: MFA phoneme extraction
    t0 = time.time()
    from fast_phoneme_extractor import FastPhonemeExtractor
    extractor = FastPhonemeExtractor()
    result = extractor.extract_from_files(audio_file, text_file)
    extractor.save_result(result, phoneme_json)
    pc = result['files']['audio']['phoneme_count']
    wc = result['files']['audio']['word_count']
    print(f"[3/5] MFA done in {time.time()-t0:.1f}s: {pc} phonemes, {wc} words")

    # Step 3b: phoneme transformation -> fixed-length JSON
    t0 = time.time()
    subprocess.run([
        sys.executable, str(ROOT / "phoneme-json-transformation.py"),
        "--input_json", str(phoneme_json),
        "--transform-type", "fixed-length",
    ], check=True, capture_output=True)
    print(f"[3.5/5] phoneme transformation in {time.time()-t0:.1f}s")

    # Step 4: video generation
    t0 = time.time()
    from video_generator import generate_video
    actor = am.get_actor(ACTOR_ID)
    blink_applier = None
    if actor.has_blink_assets:
        try:
            from blink_module.BlinkApplier import BlinkApplier
            blink_applier = BlinkApplier(str(actor.blink_assets_path))
        except Exception as e:
            print(f"[warn] blink loading failed, continuing without: {e}")

    video_path = generate_video(
        triphone_visemes_dir=str(actor.viseme_path),
        json_path=str(phoneme_json_fixed),
        audio_path=str(audio_file),
        output_path=str(video_file),
        blink_applier=blink_applier,
        actor_blink_assets_path=str(actor.blink_assets_path) if actor.has_blink_assets else None,
    )
    print(f"[4/5] video generated in {time.time()-t0:.1f}s -> {video_path}")

    # Summary
    import subprocess as sp
    p = sp.run(["ffprobe","-v","error","-show_entries","format=duration:stream=width,height,r_frame_rate",
                "-of","default=noprint_wrappers=1", str(video_path)],
               capture_output=True, text=True)
    print(f"[5/5] DONE\n{p.stdout}")

if __name__ == "__main__":
    main()
