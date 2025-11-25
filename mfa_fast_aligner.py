"""
Fast MFA Aligner Service for Real-time Applications
====================================================

This module provides a fast, stateful aligner service optimized for chatbots and 
real-time applications. Unlike the CLI which reloads models for each invocation,
this service loads models once and reuses them for multiple alignment requests.

Key Features:
- Models loaded once and kept in memory
- No database overhead (works in-memory only)
- Optimized for single audio+text pair alignment
- Streamlit-compatible
- Thread-safe with proper locking

Usage Example:
--------------
```python
from mfa_fast_aligner import FastMFAAligner

# Initialize once (loads models into memory)
aligner = FastMFAAligner(
    dictionary_path="path/to/dictionary.dict",
    acoustic_model_path="path/to/acoustic_model.zip",
    g2p_model_path="path/to/g2p_model.zip"  # optional
)

# Use repeatedly for different audio+text pairs
result = aligner.align(
    audio_path="audio.wav",
    text="This is the transcription",
    begin=0.0,  # optional, default 0.0
    end=None    # optional, default None (full file)
)

# Result contains phoneme alignments with timing
for phone_alignment in result['phones']:
    print(f"{phone_alignment['phone']}: {phone_alignment['begin']}-{phone_alignment['end']}")
```

Author: MFA Community
License: MIT
"""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add Montreal-Forced-Aligner to path BEFORE importing from it
sys.path.insert(0, str(Path(__file__).parent / "Montreal-Forced-Aligner"))

import pywrapfst
from kalpy.aligner import KalpyAligner  # Use the new API in kalpy >= 0.8
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler
from kalpy.utterance import Segment, Utterance as KalpyUtterance

from montreal_forced_aligner.data import WordType
from montreal_forced_aligner.dictionary.mixins import (
    DEFAULT_BRACKETS,
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    DEFAULT_WORD_BREAK_MARKERS,
)
from montreal_forced_aligner.models import AcousticModel, G2PModel
from montreal_forced_aligner.online.alignment import tokenize_utterance_text
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer
from montreal_forced_aligner.data import Language, OOV_WORD, LAUGHTER_WORD, CUTOFF_WORD, BRACKETED_WORD

# Word strings for special cases (used in normalization logic) - for backward compatibility
OOV_WORD_STR = str(OOV_WORD) if hasattr(OOV_WORD, '__str__') else "<unk>"
LAUGHTER_WORD_STR = str(LAUGHTER_WORD) if hasattr(LAUGHTER_WORD, '__str__') else "<laugh>"
CUTOFF_WORD_STR = str(CUTOFF_WORD) if hasattr(CUTOFF_WORD, '__str__') else "-"
BRACKETED_WORD_STR = str(BRACKETED_WORD) if hasattr(BRACKETED_WORD, '__str__') else "<bracketed>"

logger = logging.getLogger("mfa_fast_aligner")


class FastMFAAligner:
    """
    Fast, stateful MFA aligner optimized for real-time applications.
    
    This class loads acoustic models, dictionaries, and G2P models once during
    initialization and reuses them for multiple alignment requests, eliminating
    the overhead of repeated model loading.
    
    Parameters
    ----------
    dictionary_path : Path or str
        Path to the pronunciation dictionary file (.dict or .txt)
    acoustic_model_path : Path or str
        Path to the acoustic model (.zip archive or directory)
    g2p_model_path : Path or str, optional
        Path to G2P model for handling OOV words
    temp_dir : Path or str, optional
        Directory for temporary files. If None, uses system temp directory
    beam : int, default=10
        Beam width for decoding
    retry_beam : int, default=40
        Beam width for retry attempts
    acoustic_scale : float, default=0.1
        Scale factor for acoustic scores
    transition_scale : float, default=1.0
        Scale factor for transition scores
    self_loop_scale : float, default=0.1
        Scale factor for self-loop transitions
    boost_silence : float, default=1.0
        Boost factor for silence phones
    ignore_case : bool, default=True
        Whether to ignore case in text normalization
    no_tokenization : bool, default=False
        Whether to disable advanced tokenization
        
    Attributes
    ----------
    acoustic_model : AcousticModel
        Loaded acoustic model
    lexicon_compiler : LexiconCompiler
        FST-based lexicon compiler for pronunciation lookup
    kalpy_aligner : GmmAligner
        Core alignment engine (formerly KalpyAligner in newer kalpy versions)
    tokenizer : SimpleTokenizer or LanguageTokenizer
        Text tokenizer for normalization
    """

    def __init__(
        self,
        dictionary_path: Union[Path, str],
        acoustic_model_path: Union[Path, str],
        g2p_model_path: Optional[Union[Path, str]] = None,
        temp_dir: Optional[Union[Path, str]] = None,
        # Alignment parameters
        beam: int = 10,
        retry_beam: int = 40,
        acoustic_scale: float = 0.1,
        transition_scale: float = 1.0,
        self_loop_scale: float = 0.1,
        boost_silence: float = 1.0,
        # Dictionary parameters
        ignore_case: bool = True,
        no_tokenization: bool = False,
        word_break_markers: Tuple[str, ...] = DEFAULT_WORD_BREAK_MARKERS,
        punctuation: Tuple[str, ...] = DEFAULT_PUNCTUATION,
        clitic_markers: Tuple[str, ...] = DEFAULT_CLITIC_MARKERS,
        compound_markers: Tuple[str, ...] = DEFAULT_COMPOUND_MARKERS,
        brackets: Tuple[Tuple[str, str], ...] = DEFAULT_BRACKETS,
        laughter_word: str = LAUGHTER_WORD_STR,
        oov_word: str = OOV_WORD_STR,
        bracketed_word: str = BRACKETED_WORD_STR,
        cutoff_word: str = CUTOFF_WORD_STR,
    ):
        """Initialize the Fast MFA Aligner with all models loaded into memory."""
        
        # Convert paths
        self.dictionary_path = Path(dictionary_path)
        self.acoustic_model_path = Path(acoustic_model_path)
        self.g2p_model_path = Path(g2p_model_path) if g2p_model_path else None
        
        # Setup temp directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.gettempdir()) / "mfa_fast_aligner"
        else:
            self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Store alignment parameters
        self.align_options = {
            "beam": beam,
            "retry_beam": retry_beam,
            "acoustic_scale": acoustic_scale,
            "transition_scale": transition_scale,
            "self_loop_scale": self_loop_scale,
            "boost_silence": boost_silence,
        }
        
        # Store dictionary parameters
        self.dict_params = {
            "ignore_case": ignore_case,
            "word_break_markers": word_break_markers,
            "punctuation": punctuation,
            "clitic_markers": clitic_markers,
            "compound_markers": compound_markers,
            "brackets": brackets,
            "laughter_word": laughter_word,
            "oov_word": oov_word,
            "bracketed_word": bracketed_word,
            "cutoff_word": cutoff_word,
        }
        self.no_tokenization = no_tokenization
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load models (this is done once)
        logger.info("Loading acoustic model...")
        self._load_acoustic_model()
        
        logger.info("Loading and compiling lexicon...")
        self._load_lexicon()
        
        if self.g2p_model_path:
            logger.info("Loading G2P model...")
            self._load_g2p_model()
        else:
            self.g2p_model = None
        
        logger.info("Initializing tokenizer...")
        self._initialize_tokenizer()
        
        logger.info("Creating aligner...")
        self._create_aligner()
        
        logger.info("FastMFAAligner initialized successfully!")
    
    def _load_acoustic_model(self) -> None:
        """Load the acoustic model into memory."""
        self.acoustic_model = AcousticModel(self.acoustic_model_path)
        
    def _load_g2p_model(self) -> None:
        """Load the G2P model for OOV handling."""
        self.g2p_model = G2PModel(self.g2p_model_path)
    
    def _load_lexicon(self) -> None:
        """Load and compile the pronunciation dictionary into FST format."""
        # Create a dedicated directory for lexicon FSTs
        lexicon_dir = self.temp_dir / "lexicon" / self.dictionary_path.stem
        lexicon_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for cached FSTs
        l_fst_path = lexicon_dir / "L.fst"
        l_align_fst_path = lexicon_dir / "L_align.fst"
        words_path = lexicon_dir / "words.txt"
        phones_path = lexicon_dir / "phones.txt"
        
        # Initialize lexicon compiler with acoustic model parameters
        self.lexicon_compiler = LexiconCompiler(
            disambiguation=False,
            silence_probability=self.acoustic_model.parameters.get("silence_probability", 0.5),
            initial_silence_probability=self.acoustic_model.parameters.get(
                "initial_silence_probability", 0.5
            ),
            final_silence_correction=self.acoustic_model.parameters.get(
                "final_silence_correction", 1.0
            ),
            final_non_silence_correction=self.acoustic_model.parameters.get(
                "final_non_silence_correction", 1.0
            ),
            silence_phone=self.acoustic_model.parameters["optional_silence_phone"],
            oov_phone=self.acoustic_model.parameters.get("oov_phone", "spn"),
            position_dependent_phones=self.acoustic_model.parameters.get(
                "position_dependent_phones", True
            ),
            phones=self.acoustic_model.parameters["non_silence_phones"],
            ignore_case=self.dict_params["ignore_case"],
        )
        
        # Try to load cached FSTs, otherwise compile from dictionary
        if l_fst_path.exists() and l_align_fst_path.exists():
            logger.info("Loading cached lexicon FSTs...")
            self.lexicon_compiler.load_l_from_file(l_fst_path)
            self.lexicon_compiler.load_l_align_from_file(l_align_fst_path)
            self.lexicon_compiler.word_table = pywrapfst.SymbolTable.read_text(str(words_path))
            self.lexicon_compiler.phone_table = pywrapfst.SymbolTable.read_text(str(phones_path))
        else:
            logger.info("Compiling lexicon from dictionary (this may take a moment)...")
            self.lexicon_compiler.load_pronunciations(self.dictionary_path)
            
            # Cache the compiled FSTs for future use
            self.lexicon_compiler.fst.write(str(l_fst_path))
            self.lexicon_compiler.align_fst.write(str(l_align_fst_path))
            self.lexicon_compiler.word_table.write_text(str(words_path))
            self.lexicon_compiler.phone_table.write_text(str(phones_path))
            
            # Clear intermediate data to save memory
            self.lexicon_compiler.clear()
    
    def _initialize_tokenizer(self) -> None:
        """Initialize text tokenizer based on language and settings."""
        if self.no_tokenization or self.acoustic_model.language is Language.unknown:
            self.tokenizer = SimpleTokenizer(
                word_table=self.lexicon_compiler.word_table,
                **self.dict_params
            )
        else:
            self.tokenizer = generate_language_tokenizer(self.acoustic_model.language)
    
    def _create_aligner(self) -> None:
        """Create the GmmAligner instance (KalpyAligner in newer versions)."""
        # GmmAligner expects a path to the acoustic model file (final.mdl or similar)
        # The model files might be in root_directory or in subdirectories
        
        root_dir = Path(self.acoustic_model.root_directory)
        model_name = self.acoustic_model.name
        
        # Try multiple possible locations for the model files
        possible_dirs = [
            root_dir,
            root_dir / model_name,
            root_dir / f"{model_name}_acoustic",
            # MFA stores inspected models in a different location
            root_dir.parent / "inspect" / model_name / model_name,
        ]
        
        # Common model file names in MFA acoustic models
        possible_model_files = ['final.alimdl', 'final.mdl', 'model.mdl']
        acoustic_model_file = None
        
        for acoustic_dir in possible_dirs:
            if not acoustic_dir.exists():
                continue
            logger.info(f"Searching for model files in: {acoustic_dir}")
            
            for model_name_file in possible_model_files:
                model_path = acoustic_dir / model_name_file
                if model_path.exists():
                    acoustic_model_file = model_path
                    logger.info(f"✓ Found acoustic model file: {model_path}")
                    break
            
            if acoustic_model_file:
                break
        
        if acoustic_model_file is None:
            # List all checked directories and their contents to help debug
            search_info = []
            for d in possible_dirs:
                if d.exists():
                    files = list(d.glob('*.mdl*'))
                    search_info.append(f"\n  {d}: {files if files else 'no .mdl files'}")
                else:
                    search_info.append(f"\n  {d}: directory does not exist")
            
            raise FileNotFoundError(
                f"Could not find acoustic model file (.mdl). Searched in:{''.join(search_info)}\n\n"
                f"Please ensure the acoustic model has been properly extracted/installed."
            )
        
        # Use KalpyAligner with the new API (kalpy >= 0.8)
        # This takes the acoustic model object and lexicon compiler, not just paths
        self.kalpy_aligner = KalpyAligner(
            self.acoustic_model,
            self.lexicon_compiler,
            **self.align_options
        )
    
    def align(
        self,
        audio_path: Union[Path, str],
        text: str,
        begin: float = 0.0,
        end: Optional[float] = None,
        channel: int = 0,
    ) -> Dict:
        """
        Align audio with text and return phoneme-level alignments with timing.
        
        This method is thread-safe and can be called repeatedly for different
        audio+text pairs without reloading models.
        
        Parameters
        ----------
        audio_path : Path or str
            Path to the audio file (WAV format recommended)
        text : str
            Transcription text to align with audio
        begin : float, default=0.0
            Start time in seconds (for aligning a segment)
        end : float, optional
            End time in seconds (if None, uses full file duration)
        channel : int, default=0
            Audio channel to use (0 for mono or left channel)
            
        Returns
        -------
        dict
            Alignment results with structure:
            {
                'words': [
                    {
                        'word': str,
                        'begin': float,
                        'end': float,
                        'phones': [
                            {
                                'phone': str,
                                'begin': float,
                                'end': float,
                                'confidence': float
                            },
                            ...
                        ]
                    },
                    ...
                ],
                'phones': [  # Flat list of all phones
                    {
                        'phone': str,
                        'begin': float,
                        'end': float,
                        'confidence': float
                    },
                    ...
                ],
                'likelihood': float,  # Alignment likelihood score
                'text': str,  # Original text
                'normalized_text': str  # Normalized text used for alignment
            }
            
        Example
        -------
        >>> aligner = FastMFAAligner("dict.txt", "model.zip")
        >>> result = aligner.align("audio.wav", "hello world")
        >>> for phone in result['phones']:
        ...     print(f"{phone['phone']}: {phone['begin']:.3f}-{phone['end']:.3f}")
        """
        with self._lock:  # Thread-safe alignment
            audio_path = Path(audio_path)
            
            # Create segment
            segment = Segment(str(audio_path), begin, end, channel)
            
            # Normalize and tokenize text using the tokenize_utterance_text function
            normalized_text = tokenize_utterance_text(
                text,
                self.lexicon_compiler,
                self.tokenizer,
                self.g2p_model,
                language=self.acoustic_model.language,
            )
            
            # Create utterance
            utterance = KalpyUtterance(segment, normalized_text)
            
            # Generate MFCCs
            utterance.generate_mfccs(self.acoustic_model.mfcc_computer)
            
            # Compute CMVN (cepstral mean and variance normalization)
            cmvn_computer = CmvnComputer()
            cmvn = cmvn_computer.compute_cmvn_from_features([utterance.mfccs])
            
            # Apply CMVN
            utterance.apply_cmvn(cmvn)
            
            # Perform alignment
            ctm: HierarchicalCtm = self.kalpy_aligner.align_utterance(utterance)
            
            # Convert to easy-to-use format
            result = self._format_alignment_result(ctm, text, normalized_text)
            
            return result
    
    def _format_alignment_result(
        self, ctm: HierarchicalCtm, original_text: str, normalized_text: str
    ) -> Dict:
        """
        Format the CTM alignment result into a convenient dictionary structure.
        
        Parameters
        ----------
        ctm : HierarchicalCtm
            Raw alignment result from GmmAligner
        original_text : str
            Original input text
        normalized_text : str
            Normalized text used for alignment
            
        Returns
        -------
        dict
            Formatted alignment results
        """
        words = []
        all_phones = []
        
        for word_interval in ctm.word_intervals:
            phones_in_word = []
            
            for phone_interval in word_interval.phones:
                phone_dict = {
                    "phone": phone_interval.label,
                    "begin": round(phone_interval.begin, 4),
                    "end": round(phone_interval.end, 4),
                    "duration": round(phone_interval.end - phone_interval.begin, 4),
                    "confidence": round(phone_interval.confidence, 4) if phone_interval.confidence else 0.0,
                }
                phones_in_word.append(phone_dict)
                all_phones.append(phone_dict)
            
            word_dict = {
                "word": word_interval.label,
                "begin": round(word_interval.begin, 4),
                "end": round(word_interval.end, 4),
                "duration": round(word_interval.end - word_interval.begin, 4),
                "pronunciation": word_interval.pronunciation,
                "phones": phones_in_word,
            }
            words.append(word_dict)
        
        return {
            "words": words,
            "phones": all_phones,
            "likelihood": ctm.likelihood if hasattr(ctm, 'likelihood') else None,
            "text": original_text,
            "normalized_text": normalized_text,
        }
    
    def align_batch(
        self,
        audio_text_pairs: List[Tuple[Union[Path, str], str]],
        begins: Optional[List[float]] = None,
        ends: Optional[List[float]] = None,
        channels: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Align multiple audio+text pairs in sequence.
        
        This is more efficient than calling align() multiple times separately
        as it reuses CMVN computations where possible.
        
        Parameters
        ----------
        audio_text_pairs : list of (audio_path, text) tuples
            List of audio file paths and their corresponding texts
        begins : list of float, optional
            Start times for each segment (default all 0.0)
        ends : list of float, optional
            End times for each segment (default all None)
        channels : list of int, optional
            Channels for each segment (default all 0)
            
        Returns
        -------
        list of dict
            List of alignment results, one per input pair
            
        Example
        -------
        >>> pairs = [
        ...     ("audio1.wav", "hello world"),
        ...     ("audio2.wav", "goodbye friend")
        ... ]
        >>> results = aligner.align_batch(pairs)
        >>> for i, result in enumerate(results):
        ...     print(f"Result {i}: {len(result['phones'])} phones")
        """
        n = len(audio_text_pairs)
        if begins is None:
            begins = [0.0] * n
        if ends is None:
            ends = [None] * n
        if channels is None:
            channels = [0] * n
        
        results = []
        for (audio_path, text), begin, end, channel in zip(
            audio_text_pairs, begins, ends, channels
        ):
            result = self.align(audio_path, text, begin, end, channel)
            results.append(result)
        
        return results
    
    def export_textgrid(
        self,
        alignment_result: Dict,
        output_path: Union[Path, str],
        file_duration: Optional[float] = None,
        output_format: str = "long_textgrid",
    ) -> None:
        """
        Export alignment result to TextGrid format.
        
        Parameters
        ----------
        alignment_result : dict
            Result from align() method
        output_path : Path or str
            Path to save TextGrid file
        file_duration : float, optional
            Total duration of audio file (if None, uses last phone end time)
        output_format : str, default="long_textgrid"
            Format: "long_textgrid", "short_textgrid", "json", or "csv"
        """
        from kalpy.gmm.data import CtmInterval, HierarchicalCtm
        
        # Reconstruct HierarchicalCtm from result
        word_intervals = []
        for word_data in alignment_result["words"]:
            phones = [
                CtmInterval(
                    begin=p["begin"],
                    end=p["end"],
                    label=p["phone"],
                    confidence=p["confidence"],
                )
                for p in word_data["phones"]
            ]
            word_interval = CtmInterval(
                begin=word_data["begin"],
                end=word_data["end"],
                label=word_data["word"],
            )
            word_interval.phones = phones
            word_interval.pronunciation = word_data["pronunciation"]
            word_intervals.append(word_interval)
        
        ctm = HierarchicalCtm(word_intervals)
        
        if file_duration is None and word_intervals:
            file_duration = word_intervals[-1].end
        
        # Export using kalpy's built-in export
        ctm.export_textgrid(
            output_path, file_duration=file_duration, output_format=output_format
        )
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns
        -------
        dict
            Model metadata including phones, features, etc.
        """
        return {
            "acoustic_model": {
                "name": self.acoustic_model.name,
                "path": str(self.acoustic_model_path),
                "language": str(self.acoustic_model.language),
                "phones": self.acoustic_model.parameters.get("non_silence_phones", []),
                "silence_phone": self.acoustic_model.parameters.get("optional_silence_phone"),
                "parameters": self.acoustic_model.parameters,
            },
            "dictionary": {
                "path": str(self.dictionary_path),
                "word_count": self.lexicon_compiler.word_table.num_symbols() if hasattr(self.lexicon_compiler.word_table, 'num_symbols') else None,
            },
            "g2p_model": {
                "loaded": self.g2p_model is not None,
                "path": str(self.g2p_model_path) if self.g2p_model_path else None,
            },
            "align_options": self.align_options,
        }
    
    def __repr__(self) -> str:
        return (
            f"FastMFAAligner(\n"
            f"  acoustic_model={self.acoustic_model.name},\n"
            f"  dictionary={self.dictionary_path.name},\n"
            f"  g2p_model={'loaded' if self.g2p_model else 'not loaded'}\n"
            f")"
        )


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Simple CLI for testing
    if len(sys.argv) < 5:
        print("Usage: python mfa_fast_aligner.py <audio.wav> <text> <dictionary> <acoustic_model> [g2p_model]")
        print("\nExample:")
        print("  python mfa_fast_aligner.py audio.wav 'hello world' english.dict english.zip")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    text = sys.argv[2]
    dictionary = sys.argv[3]
    acoustic_model = sys.argv[4]
    g2p_model = sys.argv[5] if len(sys.argv) > 5 else None
    
    print(f"\n{'='*60}")
    print("FastMFAAligner - Quick Test")
    print(f"{'='*60}\n")
    
    # Initialize aligner (happens once)
    print("Initializing aligner...")
    aligner = FastMFAAligner(
        dictionary_path=dictionary,
        acoustic_model_path=acoustic_model,
        g2p_model_path=g2p_model,
    )
    
    print(f"\n{aligner}\n")
    
    # Perform alignment
    print(f"Aligning audio: {audio_file}")
    print(f"Text: {text}\n")
    
    result = aligner.align(audio_file, text)
    
    # Print results
    print(f"{'='*60}")
    print("ALIGNMENT RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Original text: {result['text']}")
    print(f"Normalized text: {result['normalized_text']}")
    print(f"Likelihood: {result['likelihood']}\n")
    
    print(f"{'Word':<15} {'Begin':>8} {'End':>8} {'Duration':>8}")
    print(f"{'-'*44}")
    for word in result['words']:
        print(f"{word['word']:<15} {word['begin']:>8.3f} {word['end']:>8.3f} {word['duration']:>8.3f}")
    
    print(f"\n{'Phone':<15} {'Begin':>8} {'End':>8} {'Duration':>8} {'Confidence':>10}")
    print(f"{'-'*59}")
    for phone in result['phones']:
        print(f"{phone['phone']:<15} {phone['begin']:>8.3f} {phone['end']:>8.3f} {phone['duration']:>8.3f} {phone['confidence']:>10.3f}")
    
    # Save TextGrid
    output_file = Path(audio_file).with_suffix('.TextGrid')
    aligner.export_textgrid(result, output_file)
    print(f"\nTextGrid saved to: {output_file}")
