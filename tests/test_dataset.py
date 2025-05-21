import pytest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data_utils.parser import parse_dataset
from data_utils.dataset import PhonemeDataset

def test_phoneme_dataset_loading():
    data_dir = Path("data/raw/New Stimuli 9-8-2024")
    file_paths, labels, label_map, lengths = parse_dataset(data_dir)

    dataset = PhonemeDataset(file_paths, labels, target_sr=16000)
    assert len(dataset) == len(file_paths), "Dataset length does not match number of files"
    assert len(dataset) > 0, "Dataset is empty"

    # load a sample
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