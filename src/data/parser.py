"""
Functions for parsing a directory of .wav files into phoneme-labeled data.
"""

from pathlib import Path
import torchaudio
from typing import List, Dict, Tuple, Optional
import re
import logging

def extract_label(file_path: Path) -> str:
    """
    Extract the phoneme label from the start of a filename.

    Rules:
    - Label is the first 1 to 4 lowercase letters at the start of the filename
    - Stops at the first digit, space, parenthesis, or other separator

    Args:
        file_path (Path): Path to a .wav file

    Returns:
        str: Parsed phoneme label in lowercase

    Raises:
        ValueError: If the label cannot be extracted
    """
    name = file_path.stem.lower()
    match = re.match(r"[a-z]{1,4}", name)
    if not match:
        raise ValueError(f"Cannot extract label from: {file_path.name}")
    return match.group(0)


def parse_dataset(
    data_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Path], List[int], Dict[str, int], List[int]]:
    """
    Recursively parse a directory of .wav files into paths and integer labels.

    Args:
        data_dir (Path): Root directory containing .wav files
        logger (Logger, optional): Optional logger for messages

    Returns:
        Tuple:
            - List of file paths
            - List of integer labels corresponding to each file
            - Dictionary mapping string labels to integer IDs
            - List of audio file lengths in samples
    """
    
    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_paths: List[Path] = []
    string_labels: List[str] = []
    lengths: List[int] = []

    for wav_file in data_dir.rglob("*.wav"):
        try:
            label = extract_label(wav_file)
            info = torchaudio.info(str(wav_file))
            num_samples = info.num_frames
        except Exception as e:
            log(f"Skipping file: {wav_file.name} — {e}")
            continue

        file_paths.append(wav_file)
        string_labels.append(label)
        lengths.append(num_samples)

    unique_labels = sorted(set(string_labels))
    label_map: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels: List[int] = [label_map[label] for label in string_labels]

    return file_paths, int_labels, label_map, lengths
