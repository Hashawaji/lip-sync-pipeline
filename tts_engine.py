"""
Simple Text-to-Speech Engine with Multi-Voice Support
Usage: 
    from tts_engine import text_to_speech
    text_to_speech("Hello world", "output.wav", backend="kokoro", voice="af_bella")
"""
import subprocess
from pathlib import Path
from typing import Literal, Optional


def text_to_speech(
    text: str,
    output_path: str,
    backend: Literal["espeak", "gtts", "kokoro", "edge", "auto"] = "auto",
    speed: int = 175,
    lang: str = "en",
    voice: Optional[str] = None,
    tld: str = "com",
    slow: bool = False
) -> str:
    """
    Convert text to speech audio file with multi-voice support.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save audio file (WAV or MP3 format)
        backend: TTS engine - "espeak" (fast, offline), "gtts" (online), "kokoro" (high-quality), "edge" (Microsoft Edge TTS, fast + quality), "auto" (choose best)
        speed: Speech speed in words per minute (default: 175)
        lang: Language code (default: "en")
        voice: Voice name (backend-specific)
        tld: Top-level domain for gTTS (e.g., "com", "co.uk", "com.au")
        slow: Slow speech for gTTS (default: False)
    
    Returns:
        Path to generated audio file
    
    Example:
        # Edge-TTS with male voice (RECOMMENDED)
        text_to_speech("Hello", "output.mp3", backend="edge", voice="en-US-GuyNeural")
        
        # Kokoro TTS with specific voice
        text_to_speech("Hello world", "output.wav", backend="kokoro", voice="af_bella")
        
        # gTTS with accent
        text_to_speech("Hello", "output.mp3", backend="gtts", tld="co.uk")
        
    Available Edge-TTS Voices (Male):
        - en-US-GuyNeural, en-US-DavisNeural (American Male)
        - en-GB-RyanNeural (British Male)
        - en-AU-WilliamNeural (Australian Male)
        
    Available Edge-TTS Voices (Female):
        - en-US-JennyNeural, en-US-AriaNeural (American Female)
        - en-GB-SoniaNeural (British Female)
        - en-AU-NatashaNeural (Australian Female)
    
    Available Kokoro Voices:
        - af_bella, af_nicole, af_sarah, af_sky (American Female)
        - am_adam, am_michael (American Male)
        - bf_emma, bf_isabella (British Female)
        - bm_george, bm_lewis (British Male)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-select backend
    if backend == "auto":
        if _check_edge():
            backend = "edge"
        elif _check_kokoro():
            backend = "kokoro"
        elif _check_espeak():
            backend = "espeak"
        else:
            backend = "gtts"
    
    if backend == "edge":
        return _synthesize_edge(text, output_path, voice, lang, speed)
    elif backend == "kokoro":
        return _synthesize_kokoro(text, output_path, voice, lang, speed)
    elif backend == "espeak":
        return _synthesize_espeak(text, output_path, speed, lang)
    else:  # gtts
        return _synthesize_gtts(text, output_path, lang, tld, slow)


def _check_edge() -> bool:
    """Check if Edge-TTS is available."""
    try:
        import edge_tts
        return True
    except ImportError:
        return False


def _check_kokoro() -> bool:
    """Check if Kokoro TTS is available."""
    try:
        import kokoro_onnx
        return True
    except ImportError:
        return False


def _check_espeak() -> bool:
    """Check if espeak is available."""
    try:
        result = subprocess.run(
            ["espeak", "--version"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _synthesize_espeak(text: str, output_path: Path, speed: int, lang: str) -> str:
    """Synthesize using espeak (fast, offline)."""
    cmd = [
        "espeak",
        "-v", f"{lang}+f3",
        "-s", str(speed),
        "-w", str(output_path),
        text
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"espeak failed: {e.stderr.decode()}")


def _synthesize_edge(text: str, output_path: Path, voice: Optional[str], lang: str, speed: int) -> str:
    """Synthesize using Microsoft Edge TTS (fast, high-quality, male/female voices)."""
    try:
        import edge_tts
        import asyncio
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
    
    # Default voice based on language if not specified
    if voice is None:
        if lang.startswith("en"):
            voice = "en-US-GuyNeural"  # Default male American voice
        else:
            voice = f"{lang}-Neural"
    
    # Map speed (words per minute) to Edge-TTS rate parameter
    # Normal speech is ~175 WPM, Edge-TTS uses percentage (-50% to +100%)
    # 175 WPM = 0%, 100 WPM = -43%, 300 WPM = +71%
    rate_percent = int(((speed - 175) / 175) * 100)
    rate_percent = max(-50, min(100, rate_percent))  # Clamp to valid range
    rate_str = f"{rate_percent:+d}%" if rate_percent != 0 else "+0%"
    
    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        await communicate.save(str(output_path))
    
    try:
        # Run async function
        asyncio.run(_generate())
        return str(output_path)
    except Exception as e:
        raise RuntimeError(f"Edge-TTS failed: {e}")


def _synthesize_gtts(text: str, output_path: Path, lang: str, tld: str = "com", slow: bool = False) -> str:
    """Synthesize using Google TTS (quality, online)."""
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("gTTS not installed. Run: pip install gTTS")
    
    try:
        tts = gTTS(text=text, lang=lang, slow=slow, tld=tld)
        
        if output_path.suffix.lower() == '.wav':
            temp_mp3 = output_path.with_suffix('.temp.mp3')
            tts.save(str(temp_mp3))
            
            # Convert to WAV if ffmpeg available
            if _check_ffmpeg():
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(temp_mp3),
                    "-ar", "22050", "-ac", "1",
                    str(output_path)
                ], check=True, capture_output=True)
                temp_mp3.unlink()
            else:
                temp_mp3.rename(output_path.with_suffix('.mp3'))
                output_path = output_path.with_suffix('.mp3')
        else:
            tts.save(str(output_path))
        
        return str(output_path)
    except Exception as e:
        raise RuntimeError(f"gTTS failed: {e}")


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Global Kokoro instance cache (models are large, load once)
_kokoro_instance = None


def _get_kokoro_instance():
    """Get or create cached Kokoro TTS instance."""
    global _kokoro_instance
    
    if _kokoro_instance is None:
        try:
            from kokoro_onnx import Kokoro
            from pathlib import Path
            import os
            
            # Get model directory (Kokoro auto-downloads to ~/.cache/kokoro/)
            cache_dir = Path.home() / ".cache" / "kokoro"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize Kokoro with default model
            # The library will auto-download the model on first use
            model_path = cache_dir / "kokoro-v0_19.onnx"
            voices_path = cache_dir / "voices.bin"
            
            # If models don't exist, provide helpful message
            if not model_path.exists() or not voices_path.exists():
                raise RuntimeError(
                    f"Kokoro TTS models not found. Please download them:\n"
                    f"mkdir -p ~/.cache/kokoro && cd ~/.cache/kokoro\n"
                    f"curl -L -o kokoro-v0_19.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx\n"
                    f"curl -L -o voices.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
                )
            
            _kokoro_instance = Kokoro(str(model_path), str(voices_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro TTS: {e}")
    
    return _kokoro_instance


def _synthesize_kokoro(text: str, output_path: Path, voice: Optional[str], lang: str, speed: int) -> str:
    """Synthesize using Kokoro TTS (high-quality, multi-voice)."""
    try:
        kokoro = _get_kokoro_instance()
        
        # Default voice if not specified
        if voice is None:
            voice = "af_bella"  # Default to American Female voice
        
        # Map speed (words per minute) to Kokoro's speed parameter (0.5 to 2.0)
        # Normal speech is ~150-175 WPM, map to speed=1.0
        kokoro_speed = speed / 175.0
        kokoro_speed = max(0.5, min(2.0, kokoro_speed))  # Clamp to valid range
        
        # Kokoro expects language codes like "en-us", "en-gb", etc.
        if lang == "en":
            lang = "en-us"  # Default to American English
        
        # Generate speech
        samples, sample_rate = kokoro.create(text, voice=voice, speed=kokoro_speed, lang=lang)
        
        # Save to file
        if output_path.suffix.lower() in ['.wav', '.mp3']:
            # Save as WAV using scipy or soundfile
            try:
                import soundfile as sf
                sf.write(str(output_path), samples, sample_rate)
            except ImportError:
                # Fallback to scipy if soundfile not available
                try:
                    from scipy.io import wavfile
                    import numpy as np
                    # Convert float samples to int16
                    samples_int16 = (samples * 32767).astype(np.int16)
                    wavfile.write(str(output_path), sample_rate, samples_int16)
                except ImportError:
                    raise RuntimeError("Please install soundfile or scipy: pip install soundfile")
            
            return str(output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
            
    except Exception as e:
        raise RuntimeError(f"Kokoro TTS failed: {e}")


# Legacy class-based interface (optional)
class FastTTS:
    """Legacy class interface (use text_to_speech() function instead)."""
    def __init__(self, backend: Literal["espeak", "gtts", "auto"] = "auto"):
        self.backend = backend
    
    def synthesize(self, text: str, output_path: str, speed: int = 175, lang: str = "en") -> str:
        return text_to_speech(text, output_path, backend=self.backend, speed=speed, lang=lang)


if __name__ == "__main__":
    import time
    
    print("Simple TTS Test")
    print("=" * 50)
    
    text = "Hello! This is a fast text to speech system."
    output_file = "test_output.wav"
    
    print(f"Text: '{text}'")
    print(f"Output: {output_file}\n")
    
    start = time.time()
    result = text_to_speech(text, output_file)
    elapsed = time.time() - start
    
    print(f"✓ Success!")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  File: {result}")
