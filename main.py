from pathlib import Path
from experiment.exp_params import ExpParams
from experiment.experiment import Experiment

if __name__ == "__main__":
    params = ExpParams(
        # === Paths ===
        data_path=Path("data/raw/New Stimuli 9-8-2024"),
        output_dir=Path("outputs"),
        log_dir=Path("logs"),
        run_base_dir=Path("runs"),

        # === Audio ===
        target_sr=16000,
        max_length=None,

        # === Dataset ===
        n_augment=2,  # Required for NT-Xent
        pad_strategy="random",

        # === Transforms ===
        use_mfcc=True,
        use_log_mel=False,
        use_wavelet=False,
        use_time_mask=True,
        use_freq_mask=True,
        use_noise=True,
        time_mask_p=0.5,
        time_mask_param=30,
        freq_mask_p=0.5,
        freq_mask_param=10,
        noise_p=0.3,
        noise_std=0.005,

        # === Device ===
        device="auto",

        # === Training ===
        batch_size=16,
        learning_rate=3e-4,
        epochs=50,
        num_workers=2,
        pin_memory=None,  # Will be set automatically
        drop_last=False,

        # === Model ===
        embedding_dim=128,
        use_attention=True,

        # === Contrastive ===
        temperature=0.1,

        # === Run Mode ===
        mode="train",  # Change to "evaluate" when ready
    )

    import torch
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))  # if available


    experiment = Experiment(params)

    match params.mode:
        case "train":
            experiment.train()
        case "evaluate":
            experiment.evaluate()
        case "visualize":
            experiment.visualize()
        case _:
            raise ValueError(f"Unknown mode: {params.mode}")
