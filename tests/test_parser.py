import pytest
from pathlib import Path
from data_utils.parser import parse_dataset

def test_parse_dataset_real_data():
    # Adjust to match your actual test folder
    test_data_dir = Path("data/raw/New Stimuli 9-8-2024")

    expected_num_files = 126
    
    expected_labels = (['ada', 'apa', 'bi', 'bu', 'da', 'de', 'di', 
                        'du', 'ege', 'ete', 'fa', 'fe', 'fi', 'fu', 
                        'ga', 'ge', 'gi', 'gu', 'ibi', 'isi', 'ka', 
                        'ke', 'ki', 'ku', 'pa', 'pe', 'pi', 'pu', 
                        'sa', 'se', 'si', 'su', 'ta', 'te', 'ti', 
                        'tu', 'ubu', 'uku'])
    
    expected_num_labels = len(expected_labels)

    file_paths, int_labels, label_map, lengths = parse_dataset(test_data_dir)

    assert len(file_paths) == expected_num_files, f"Expected {expected_num_files} files, got {len(file_paths)}"
    assert len(label_map) == expected_num_labels, f"Expected {expected_num_labels} labels, got {len(label_map)}"
    for label in expected_labels:
        assert label in label_map, f"Missing label: {label}"
    assert all(isinstance(i, int) and i > 0 for i in lengths), "All lengths should be positive integers"
