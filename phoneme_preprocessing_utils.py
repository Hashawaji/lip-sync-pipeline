# This file marks the directory as a Python package.
from typing import Dict, Any, List

def remove_arpabet_and_ipa(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove 'arpabet_phonemes' and 'ipa_phonemes' from each file's data.
    """
    files = data.get('files', {})
    for file_data in files.values():
        file_data.pop('arpabet_phonemes', None)
        file_data.pop('ipa_phonemes', None)
    return data

def fill_gaps_with_silence(phoneme_list: List[Dict[str, Any]], gap_threshold: float = 0.01) -> List[Dict[str, Any]]:
    """
    Fill gaps between phonemes with 'sil' (silence) phonemes.
    """
    if not phoneme_list:
        return phoneme_list

    filled_phonemes = []
    first_start = phoneme_list[0].get('start_s', 0.0)
    if first_start > gap_threshold:
        filled_phonemes.append({
            'phoneme': 'sil',
            'start_s': 0.0,
            'end_s': first_start,
            'duration_s': first_start
        })

    for i, phoneme in enumerate(phoneme_list):
        filled_phonemes.append(phoneme)
        if i < len(phoneme_list) - 1:
            current_end = phoneme.get('end_s', 0.0)
            next_start = phoneme_list[i + 1].get('start_s', 0.0)
            gap_duration = next_start - current_end
            if gap_duration > gap_threshold:
                filled_phonemes.append({
                    'phoneme': 'sil',
                    'start_s': current_end,
                    'end_s': next_start,
                    'duration_s': gap_duration
                })

    return filled_phonemes

def convert_multiple_files_to_entire_sequence(entire_phoneme_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert multiple files' phoneme data into a single continuous sequence.
    """
    current_time = 0.0
    small_silence_duration = 0.1
    filled_phonemes = []

    for i, phoneme in enumerate(entire_phoneme_sequence):
        if i > 0 and phoneme.get('start_s', 0.0) < entire_phoneme_sequence[i - 1].get('start_s', 0.0):
            silence = {
                'phoneme': 'sil',
                'start_s': current_time,
                'end_s': current_time + small_silence_duration,
                'duration_s': small_silence_duration
            }
            filled_phonemes.append(silence)
            current_time += small_silence_duration

        duration = phoneme.get('end_s', 0.0) - phoneme.get('start_s', 0.0)
        new_phoneme = phoneme.copy()
        new_phoneme['start_s'] = current_time
        new_phoneme['end_s'] = current_time + duration
        new_phoneme['duration_s'] = duration
        filled_phonemes.append(new_phoneme)
        current_time = new_phoneme['end_s']

    return filled_phonemes

def preprocess_phoneme_sequence(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Preprocess all files' phoneme sequences: fill gaps with silence and concatenate.
    """
    entire_phoneme_sequence: List[Dict[str, Any]] = []
    files = data.get('files', {})
    for file_name, file_data in files.items():
        print(f"Processing {file_name}...")
        original_phonemes = file_data.get('viseme_phonemes', [])
        filled_phonemes = fill_gaps_with_silence(original_phonemes)
        gaps_filled = len(filled_phonemes) - len(original_phonemes)
        if gaps_filled > 0:
            print(f"  Added {gaps_filled} silence phonemes to fill gaps")
        entire_phoneme_sequence.extend(filled_phonemes)
    return entire_phoneme_sequence