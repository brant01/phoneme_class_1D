import pytest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data_utils.parser import parse_dataset
from data_utils.dataset import PhonemeDataset
from experiment.exp_params import ExpParams

def get_test_params(**overrides) -> ExpParams:
    """Helper to create consistent test params with optional overrides"""
    defaults = {
        "data_path": Path("data/raw/New Stimuli 9-8-2024"),
        "target_sr": 16000,
        "pad_strategy": "random",
        "use_mfcc": False,
        "use_time_mask": False,
        "use_freq_mask": False,
        "use_noise": False,
        "device": "cpu",
        "n_augment": 1,
        "max_length": None,
    }
    return ExpParams(**{**defaults, **overrides})


def test_phoneme_dataset_loading():
    params = get_test_params()
    file_paths, labels, _, _ = parse_dataset(params.data_path)
    dataset = PhonemeDataset(file_paths, labels, params=params)

    assert len(dataset) == len(file_paths), "Dataset length does not match number of files"
    assert len(dataset) > 0, "Dataset is empty"

    sample_waveform, sample_label = dataset[0]
    assert isinstance(sample_waveform, torch.Tensor), "Sample waveform is not a tensor"
    assert sample_waveform.ndim in (1, 2), "Sample waveform should be 1D or 2D tensor"
    assert isinstance(sample_label, int), "Sample label is not an integer"
    assert sample_waveform.shape[-1] > 0, "Waveform should have non-zero length"

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    x, y = batch
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape[0] == 4

def test_phoneme_dataset_augmentation_count():
    params = get_test_params(n_augment=5)
    file_paths, labels, _, _ = parse_dataset(params.data_path)
    dataset = PhonemeDataset(file_paths, labels, params=params)

    expected_length = len(file_paths) * params.n_augment
    assert len(dataset) == expected_length, f"Expected {expected_length}, got {len(dataset)}"

def test_padding_variability():
    params = get_test_params(n_augment=2)
    file_paths, labels, _, _ = parse_dataset(params.data_path)
    dataset = PhonemeDataset(file_paths, labels, params=params)

    w1, _ = dataset[0]
    w2, _ = dataset[len(file_paths)]
    assert not torch.equal(w1, w2), "Padding should introduce variability between augmentations"
