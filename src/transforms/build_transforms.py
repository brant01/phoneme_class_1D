from transforms.compose import Compose
from transforms.mfcc import MFCC
from transforms.log_mel import LogMelSpectrogram
from transforms.masking import RandomTimeMask, RandomFreqMask
from transforms.noise import AddNoise
from experiment.exp_params import ExpParams

def build_transforms(params: ExpParams):
    transforms = []

    if params.use_mfcc:
        transforms.append(MFCC(sample_rate=params.target_sr))
    elif params.use_log_mel:
        transforms.append(LogMelSpectrogram(sample_rate=params.target_sr))

    if params.use_time_mask:
        transforms.append(RandomTimeMask(max_width=params.time_mask_param, p=params.time_mask_p))

    if params.use_freq_mask:
        transforms.append(RandomFreqMask(max_width=params.freq_mask_param, p=params.freq_mask_p))

    if params.use_noise:
        transforms.append(AddNoise(std=params.noise_std, p=params.noise_p))

    return Compose(transforms)
