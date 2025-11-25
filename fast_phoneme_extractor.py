#!/usr/bin/env python3
"""
Fast MFA Phoneme Extractor - Python API
========================================

A programmatic interface to FastMFAAligner for phoneme extraction.
Use this in your Python code instead of the CLI.

"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add Montreal-Forced-Aligner to path BEFORE importing mfa_fast_aligner
sys.path.insert(0, str(Path(__file__).parent / "Montreal-Forced-Aligner"))

try:
    from mfa_fast_aligner import FastMFAAligner
except ImportError as e:
    raise ImportError(
        f"Could not import FastMFAAligner: {e}\n"
        "Make sure mfa_fast_aligner.py is in the same directory and "
        "the Montreal-Forced-Aligner folder is present."
    )

logger = logging.getLogger(__name__)


# Phoneme conversion mappings (ARPABET -> IPA -> VISEME)
ARPABET_TO_IPA = {
            # Vowels
            'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
            'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IY': 'i', 'OW': 'oʊ',
            'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
            # Consonants
            'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'g',
            'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
            'NG': 'ŋ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't',
            'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
            # Additional ARPABET phones
            'AX': 'ə', 'IX': 'ɨ', 'DX': 'ɾ',
            # Silence markers
            'SIL': 'sil', 'SP': 'sp'
        }

IPA_TO_VISEME = {
            # p viseme (bilabial)
            'b': 'b', 'B': 'b', 'm': 'b', 'M': 'b', 'p': 'b', 'P': 'b',
            # t viseme (alveolar)
            'd': 'd', 'D': 'd', 'l': 'd', 'L': 'd', 'n': 'd', 'N': 'd', 't': 'd', 'T': 'd',
            # S viseme (postalveolar)
            'ʃ': 'ʃ', 'ʒ': 'ʃ', 'tʃ': 'ʃ', 'dʒ': 'ʃ',
            # T viseme (dental)
            'ð': 'ð', 'Ð': 'ð', 'θ': 'ð',
            # f viseme (labiodental)
            'f': 'f', 'F': 'f', 'v': 'f', 'V': 'f',
            # k viseme (velar/glottal)
            'ɡ': 'ɡ', 'g': 'ɡ', 'h': 'ɡ', 'H': 'ɡ', 'k': 'ɡ', 'K': 'ɡ', 'ŋ': 'ɡ', 'Ŋ': 'ɡ',
            # i viseme (high front)
            'j': 'j', 'i': 'j', 'I': 'j', 'ɪ': 'j', 'Ɪ': 'j', 'Iː': 'j',
            # r viseme (rhotic)
            'ɹ': 'ɹ', 'ɝ': 'ɹ', 'ɚ': 'ɹ',
            # s viseme (sibilant)
            's': 's', 'S': 's', 'z': 's', 'Z': 's',
            # u viseme (high back rounded)
            'w': 'w', 'W': 'w', 'u': 'w', 'ʊ': 'w',
            # @ viseme (schwa/central)
            'ə': 'ə', 'Ə': 'ə',
            # a viseme (low front/central)
            'æ': 'æ', 'Æ': 'æ', 'aɪ': 'æ', 'aʊ': 'æ', 'ɑ': 'æ',
            # e viseme (mid front)
            'eɪ': 'eɪ',
            # E viseme (mid-low front/central)
            'ɛ': 'ɛ', 'Ɛ': 'ɛ', 'ʌ': 'ɛ', 'Ʌ': 'ɛ',
            # o viseme (mid back rounded)
            'oʊ': 'oʊ',
            # O viseme (mid-low back rounded)
            'ɔ': 'ɔ', 'Ɔ': 'ɔ', 'ɔɪ': 'ɔ',
            # Silence markers
            'sil': 'sil', 'sp': 'spn'
        }


def convert_phoneme_formats(phone: str) -> Tuple[str, str, str]:
    """
    Convert phoneme from ARPABET to IPA and VISEME formats.
    
    Args:
        phone: ARPABET phoneme (with optional stress digit)
        
    Returns:
        Tuple of (arpabet, ipa, viseme)
    """
    # Remove stress markers (0, 1, 2) from ARPABET
    base_phone = re.sub(r'[012]$', '', phone.upper())
    
    # Convert to IPA
    ipa = ARPABET_TO_IPA.get(base_phone, phone)
    
    # Convert to VISEME
    viseme = IPA_TO_VISEME.get(ipa, ipa)
    
    return phone, ipa, viseme


class TextCleaner:
    """Text cleaning with pre-compiled regex patterns"""
    
    def __init__(self):
        # Pre-compile all regex patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.quotes_pattern = re.compile(r'["""'']')
        self.hyphen_pattern = re.compile(r'(\w+)-(\w+)')
        self.numbers_pattern = re.compile(r'\b\d+\b')
        self.punct_pattern = re.compile(r'[^\w\s\.\!\?\,\-]')
        
        # Pre-compiled contraction patterns
        self.contractions = [
            (re.compile(r"\bwon't\b", re.I), "will not"),
            (re.compile(r"\bcan't\b", re.I), "cannot"),
            (re.compile(r"\bain't\b", re.I), "am not"),
            (re.compile(r"\bn't\b", re.I), " not"),
            (re.compile(r"'ll\b", re.I), " will"),
            (re.compile(r"'ve\b", re.I), " have"),
            (re.compile(r"'re\b", re.I), " are"),
            (re.compile(r"'d\b", re.I), " would"),
            (re.compile(r"'m\b", re.I), " am"),
            (re.compile(r"'s\b", re.I), " is"),
        ]
    
    def clean(self, text: str, minimal: bool = False) -> str:
        """
        Clean text for MFA alignment.
        
        Args:
            text: Input text
            minimal: If True, skip some cleaning steps for extra speed
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic whitespace normalization
        text = self.whitespace_pattern.sub(' ', text.strip())
        
        if minimal:
            return text.lower()
        
        # Full cleaning
        text = self.quotes_pattern.sub('', text)
        
        # Expand contractions
        for pattern, replacement in self.contractions:
            text = pattern.sub(replacement, text)
        
        # Clean punctuation, hyphens, numbers
        text = self.punct_pattern.sub('', text)
        text = self.hyphen_pattern.sub(r'\1 \2', text)
        text = self.numbers_pattern.sub('', text)
        
        # Lowercase and final cleanup
        text = text.lower()
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text


class FastPhonemeExtractor:
    """
    Fast phoneme extraction using FastMFAAligner.
    
    This class provides a high-level interface for extracting phoneme alignments
    from audio files. Models are loaded once and kept in memory for fast repeated use.
    
    Example:
        >>> extractor = FastPhonemeExtractor()
        >>> result = extractor.extract_from_files("audio.mp3", "text.txt")
        >>> print(f"Found {len(result['phonemes'])} phonemes")
    """
    
    def __init__(self, 
                 dictionary_path: Optional[Path] = None,
                 acoustic_model_path: Optional[Path] = None,
                 g2p_model_path: Optional[Path] = None,
                 beam: int = 10,
                 acoustic_scale: float = 0.1):
        """
        Initialize the fast phoneme extractor.
        
        Args:
            dictionary_path: Path to MFA dictionary (default: english_us_arpa)
            acoustic_model_path: Path to MFA acoustic model (default: english_us_arpa)
            g2p_model_path: Optional path to G2P model for OOV words
            beam: Beam width for alignment (higher = more accurate but slower)
            acoustic_scale: Acoustic score scaling
        """
        # Use default paths if not provided - point to local MFA folder in repo
        if dictionary_path is None:
            dictionary_path = Path(__file__).parent / "MFA" / "pretrained_models" / "dictionary" / "english_us_arpa.dict"
        
        if acoustic_model_path is None:
            acoustic_model_path = Path(__file__).parent / "MFA" / "pretrained_models" / "acoustic" / "english_us_arpa.zip"
        
        # Validate paths
        if not dictionary_path.exists():
            raise FileNotFoundError(
                f"Dictionary not found: {dictionary_path}\n"
                "Download with: mfa model download dictionary english_us_arpa"
            )
        
        if not acoustic_model_path.exists():
            raise FileNotFoundError(
                f"Acoustic model not found: {acoustic_model_path}\n"
                "Download with: mfa model download acoustic english_us_arpa"
            )
        
        logger.info("Initializing FastMFAAligner (loading models into memory)...")
        start_time = time.time()
        
        # Initialize the aligner (models loaded once)
        self.aligner = FastMFAAligner(
            dictionary_path=str(dictionary_path),
            acoustic_model_path=str(acoustic_model_path),
            g2p_model_path=str(g2p_model_path) if g2p_model_path else None,
            beam=beam,
            acoustic_scale=acoustic_scale,
            ignore_case=True
        )
        
        init_time = time.time() - start_time
        logger.info(f"✓ FastMFAAligner initialized in {init_time:.2f}s")
        
        self.text_cleaner = TextCleaner()
    
    def extract(self, audio_path: Path, text: str, 
                clean_text: bool = True) -> Dict[str, Any]:
        """
        Extract phoneme alignments from audio and text.
        Returns data in the comprehensive format compatible with mfa_phoneme_cli_v3.py
        
        Args:
            audio_path: Path to audio file
            text: Transcription text
            clean_text: Whether to clean the text first
            
        Returns:
            Dictionary with comprehensive phoneme alignments in MFA v3 format
        """
        # Clean text if requested
        if clean_text:
            cleaned_text = self.text_cleaner.clean(text)
        else:
            cleaned_text = text
        
        # Perform alignment using FastMFAAligner
        result = self.aligner.align(
            audio_path=str(audio_path),
            text=cleaned_text
        )
        
        # Handle different result formats - FastMFAAligner may return 'phones' or 'phonemes'
        phonemes_list = result.get('phonemes') or result.get('phones') or []
        words_list = result.get('words') or []
        
        # Calculate total duration from last phoneme
        total_duration = 0.0
        if phonemes_list:
            # Check if phoneme has 'end' or 'end_time' field
            last_phoneme = phonemes_list[-1]
            total_duration = last_phoneme.get('end') or last_phoneme.get('end_time') or 0.0
        
        # Convert phonemes to all three formats
        arpabet_phonemes = []
        ipa_phonemes = []
        viseme_phonemes = []
        
        for phone_data in phonemes_list:
            # Handle different key names
            phone = phone_data.get('phone') or phone_data.get('phoneme') or phone_data.get('label', '')
            start = phone_data.get('begin') or phone_data.get('start') or phone_data.get('start_time') or 0.0
            end = phone_data.get('end') or phone_data.get('end_time') or 0.0
            duration = end - start
            
            # Convert to all formats
            arpabet, ipa, viseme = convert_phoneme_formats(phone)
            
            # ARPABET format (original)
            arpabet_phonemes.append({
                'phoneme': arpabet,
                'start_s': start,
                'end_s': end,
                'duration_s': duration
            })
            
            # IPA format
            ipa_phonemes.append({
                'phoneme': ipa,
                'start_s': start,
                'end_s': end,
                'duration_s': duration
            })
            
            # VISEME format
            viseme_phonemes.append({
                'phoneme': viseme,
                'start_s': start,
                'end_s': end,
                'duration_s': duration
            })
        
        # Format words
        words = []
        for word_data in words_list:
            word_text = word_data.get('word') or word_data.get('text') or word_data.get('label', '')
            word_start = word_data.get('begin') or word_data.get('start') or word_data.get('start_time') or 0.0
            word_end = word_data.get('end') or word_data.get('end_time') or 0.0
            words.append({
                'word': word_text,
                'start_s': word_start,
                'end_s': word_end,
                'duration_s': word_end - word_start
            })
        
        # Build comprehensive result in mfa_phoneme_cli_v3 format
        comprehensive_result = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_files': 1,
                'format': 'MFA_phoneme_alignment_comprehensive_v2',
                'includes_all_formats': True,
                'phoneme_formats': ['ARPABET', 'IPA', 'VISEME']
            },
            'files': {
                'audio': {
                    'total_duration': total_duration,
                    'phoneme_count': len(arpabet_phonemes),
                    'word_count': len(words),
                    'tier_names': ['words', 'phones'],
                    'arpabet_phonemes': arpabet_phonemes,
                    'ipa_phonemes': ipa_phonemes,
                    'viseme_phonemes': viseme_phonemes,
                    'words': words
                }
            }
        }
        
        return comprehensive_result
    
    def extract_from_files(self, audio_path: Path, text_path: Path,
                          clean_text: bool = True) -> Dict[str, Any]:
        """
        Extract phonemes from audio file and text file.
        
        Args:
            audio_path: Path to audio file
            text_path: Path to text file
            clean_text: Whether to clean text
            
        Returns:
            Dictionary with comprehensive phoneme alignments
        """
        # Read text from file
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        return self.extract(audio_path, text, clean_text)
    
    def extract_batch(self, audio_text_pairs: List[Tuple[Path, str]],
                     clean_text: bool = True) -> List[Dict[str, Any]]:
        """
        Extract phonemes from multiple audio-text pairs efficiently.
        
        Args:
            audio_text_pairs: List of (audio_path, text) tuples
            clean_text: Whether to clean texts
            
        Returns:
            List of comprehensive phoneme alignment results
        """
        results = []
        
        for audio_path, text in audio_text_pairs:
            try:
                result = self.extract(audio_path, text, clean_text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results.append({
                    'file': str(audio_path),
                    'text': text,
                    'error': str(e)
                })
        
        return results
    
    def extract_from_directory(self, audio_dir: Path, text_dir: Path,
                              output_dir: Optional[Path] = None,
                              clean_text: bool = True) -> List[Dict[str, Any]]:
        """
        Extract phonemes from all matching audio-text file pairs in directories.
        
        Args:
            audio_dir: Directory containing audio files
            text_dir: Directory containing text files
            output_dir: Optional directory to save JSON results
            clean_text: Whether to clean text
            
        Returns:
            List of comprehensive phoneme alignment results
        """
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process each file
        results = []
        
        for i, audio_path in enumerate(audio_files, 1):
            # Find matching text file
            text_path = text_dir / f"{audio_path.stem}.txt"
            
            if not text_path.exists():
                logger.warning(f"[{i}/{len(audio_files)}] Skipping {audio_path.name} - no matching text file")
                continue
            
            logger.info(f"[{i}/{len(audio_files)}] Processing {audio_path.name}...")
            
            try:
                # Read text
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # Extract phonemes
                start_time = time.time()
                result = self.extract(audio_path, text, clean_text)
                elapsed = time.time() - start_time
                
                phoneme_count = result['files']['audio']['phoneme_count']
                word_count = result['files']['audio']['word_count']
                
                logger.info(f"  ✓ Complete in {elapsed:.3f}s - {phoneme_count} phonemes, {word_count} words")
                
                results.append(result)
                
                # Save to file if output directory specified
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{audio_path.stem}.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                results.append({
                    'file': str(audio_path),
                    'error': str(e)
                })
        
        logger.info(f"\nProcessed {len(results)}/{len(audio_files)} files successfully")
        
        return results
    
    def save_result(self, result: Dict[str, Any], output_path: Path) -> None:
        """
        Save extraction result to JSON file in comprehensive format.
        
        Args:
            result: Result from extract() or extract_from_files()
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
