from pathlib import Path
from experiment.exp_params import ExpParams
from experiment.experiment import Experiment

if __name__ == "__main__":
    params = ExpParams(
        data_path=Path("data/raw/New Stimuli 9-8-2024"),
        output_dir=Path("output_dir"),
        log_dir=Path("log_dir"),
        target_sr=16000,
        max_length=None,
        n_augment=5,
        use_mfcc=True,
        use_time_mask=True,
        use_freq_mask=True,
        use_noise=True,
        time_mask_p=0.5,
        freq_mask_p=0.5,
        noise_p=0.3,
        noise_std=0.005,
        device="auto",
        batch_size=16,
        epochs=50,
    )

    experiment = Experiment(params=params)
    experiment.train()
