"""
Generate a few Cleopatra demo videos for the supervisor.

Uses the full FastLipSynth pipeline with:
  - actor_1 (Cleopatra) viseme library (8,443 triphones)
  - Blink synthesis (Cleopatra has blink_assets)
  - Frame interpolation (smoother motion)
  - Temporal smoothing at viseme transitions
"""
import sys, time, subprocess, json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from actor_manager import ActorManager
from fast_phoneme_extractor import FastPhonemeExtractor
from video_generator import generate_video

OUT_DIR = Path('/Users/admin/thesis/cleopatra_demos')
OUT_DIR.mkdir(exist_ok=True)

ACTOR_ID = 'actor_1'

DEMOS = [
    ('greeting',
     "Greetings, traveller. I am Cleopatra, last queen of the Ptolemaic dynasty of Egypt. "
     "Welcome to the royal court of Alexandria, where the wisdom of the ancients flows like "
     "the Nile itself."),

    ('reign',
     "For three decades I have ruled this kingdom, between the desert and the sea. "
     "My court receives ambassadors from every corner of the known world, from Rome in "
     "the west to the sacred lands of the east."),

    ('contemplation',
     "When the night falls upon Alexandria and the lighthouse burns, I often walk along "
     "the harbour. The stars above are the same that watched over my ancestors a thousand "
     "years before me. What will the next thousand years remember of us?"),
]


def main():
    print('[setup] loading actor + MFA...')
    am = ActorManager()
    actor = am.get_actor(ACTOR_ID)
    if actor is None:
        raise RuntimeError(f'Actor {ACTOR_ID} not found')
    print(f'  actor: {actor.display_name}')
    print(f'  viseme library: {actor.viseme_path}')
    print(f'  has blink assets: {actor.has_blink_assets}')

    extractor = FastPhonemeExtractor()

    blink_applier = None
    if actor.has_blink_assets:
        try:
            from blink_module.BlinkApplier import BlinkApplier
            dlib_model = ROOT / 'blink_module' / 'assets' / 'shape_predictor_68_face_landmarks.dat'
            blink_applier = BlinkApplier(str(dlib_model), str(actor.blink_assets_path))
            print(f'  blink applier loaded')
        except Exception as e:
            print(f'  WARN: blink applier failed to load — proceeding without: {e}')

    timings = {}
    for name, text in DEMOS:
        print(f'\n=== {name} ===')
        wdir = OUT_DIR / name
        wdir.mkdir(exist_ok=True)
        text_file = wdir / 'text.txt'
        audio_file = wdir / 'audio.mp3'
        ph_json = wdir / 'complete_phoneme_alignments.json'
        out_video = wdir / 'output.mp4'

        text_file.write_text(text)
        print(f'  text ({len(text)} chars): "{text[:80]}…"')

        # 1. TTS via the actor's configured voice
        t0 = time.time()
        am.generate_actor_voice(ACTOR_ID, text, str(audio_file))
        t_tts = time.time() - t0
        print(f'  [1] TTS: {t_tts:.1f}s')

        # 2. MFA alignment
        t0 = time.time()
        result = extractor.extract_from_files(audio_file, text_file)
        extractor.save_result(result, ph_json)
        t_mfa = time.time() - t0
        pc = result['files']['audio']['phoneme_count']
        wc = result['files']['audio']['word_count']
        print(f'  [2] MFA: {t_mfa:.1f}s ({pc} phonemes, {wc} words)')

        # 3. Phoneme transformation -> fixed-length triphone enriched JSON
        t0 = time.time()
        subprocess.run([
            sys.executable, str(ROOT / 'phoneme-json-transformation.py'),
            '--input_json', str(ph_json),
            '--transform-type', 'fixed-length',
        ], check=True, capture_output=True)
        t_tx = time.time() - t0
        enriched_json = ph_json.parent / 'complete_phoneme_alignments_w_reps_fixed_len.json'
        print(f'  [3] transform: {t_tx:.1f}s')

        # 4. Video generation (with blinks + interpolation)
        t0 = time.time()
        generate_video(
            triphone_visemes_dir=str(actor.viseme_path),
            json_path=str(enriched_json),
            audio_path=str(audio_file),
            output_path=str(out_video),
            blink_applier=blink_applier,
            actor_blink_assets_path=str(actor.blink_assets_path) if actor.has_blink_assets else None,
        )
        t_vid = time.time() - t0
        total = t_tts + t_mfa + t_tx + t_vid
        timings[name] = {
            'text': text,
            'tts_s': round(t_tts, 2),
            'mfa_s': round(t_mfa, 2),
            'transform_s': round(t_tx, 2),
            'video_gen_s': round(t_vid, 2),
            'total_s': round(total, 2),
            'phoneme_count': pc,
            'word_count': wc,
        }
        # Confirm dimensions/duration of output
        info = subprocess.check_output(
            ['ffprobe', '-v', 'error',
             '-show_entries', 'format=duration:stream=width,height,r_frame_rate',
             '-of', 'csv=p=0:s=x', str(out_video)], text=True).strip().splitlines()
        print(f'  [4] video: {t_vid:.1f}s')
        print(f'      total: {total:.1f}s, output: {info[0] if info else "?"}')
        print(f'      saved: {out_video}')

    (OUT_DIR / 'timings.json').write_text(json.dumps(timings, indent=2))
    print(f'\nAll demos saved under {OUT_DIR}')


if __name__ == '__main__':
    main()
