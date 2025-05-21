import torch
from experiment.exp_params import ExpParams
from transforms.build_transforms import build_transforms

def test_build_transforms_mfcc():
    params = ExpParams(
        use_mfcc=True,
        use_log_mel=False,
        use_time_mask=False,
        use_freq_mask=False,
        use_noise=False,
    )
    transform = build_transforms(params)

    assert callable(transform), "Transform pipeline should be callable"

    waveform = torch.randn(16000)  # 1 second of mono audio
    out = transform(waveform)

    assert isinstance(out, torch.Tensor), "Output should be a tensor"
    assert out.ndim >= 2, "Output should be at least 2D (features, time)"

def test_build_transforms_with_augmentation():
    params = ExpParams(
        use_mfcc=True,
        use_time_mask=True,
        use_freq_mask=True,
        use_noise=True,
        time_mask_p=1.0,
        freq_mask_p=1.0,
        noise_p=1.0,
    )
    transform = build_transforms(params)

    waveform = torch.randn(16000)
    out = transform(waveform)

    assert out.shape[-1] > 0, "Augmented output should have non-zero length"
