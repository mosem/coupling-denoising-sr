import os
import math
import subprocess

import numpy as np
import cv2
import sox as sox
from torchvision.utils import make_grid
import torchaudio
import torch

from pesq import pesq
from pystoi import stoi

import logging

logger = logging.getLogger('base')


def tensor2audio(tensor, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    tensor = tensor.unsqueeze(dim=0)
    return tensor


def save_audio(wav, filename, sr):
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def calculate_pesq(out_sig, ref_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    if sr not in [8000, 16000]:
        return pesq_val
    for i in range(len(ref_sig)):
        mode = 'wb' if sr == 16000 else 'nb'
        tmp = pesq(sr, ref_sig[i], out_sig[i], mode)  # from pesq
        pesq_val += tmp
    return pesq_val


# this is because pystoi raises a FutureWarning:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_stoi(out_sig, ref_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=True)
    return stoi_val


def calculate_sisnr(out_sig, ref_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        out_sig: vector (torch.Tensor), enhanced signal [B,T]
        ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
    Returns:
        SISNR
    """
    out_sig = out_sig.numpy()
    ref_sig = ref_sig.numpy()
    assert len(ref_sig) == len(out_sig)
    B, T = ref_sig.shape
    ref_sig = ref_sig - np.mean(ref_sig, axis=1).reshape(B, 1)
    out_sig = out_sig - np.mean(out_sig, axis=1).reshape(B, 1)
    ref_energy = (np.sum(ref_sig ** 2, axis=1) + eps).reshape(B, 1)
    proj = (np.sum(ref_sig * out_sig, axis=1).reshape(B, 1)) * \
           ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2, axis=1) / (np.sum(noise ** 2, axis=1) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr.mean()


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def calculate_lsd(out_sig, ref_sig):
    """
       Compute LSD (log spectral distance)
       Arguments:
           out_sig: vector (torch.Tensor), enhanced signal [B,T]
           ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """

    fft_size = 1024
    shift_size = 120
    win_length = 600
    window = torch.hann_window(win_length)

    X = torch.log(torch.pow(torch.abs(stft(out_sig, fft_size, shift_size, win_length, window)), 2))
    Y = torch.log(torch.pow(torch.abs(stft(ref_sig, fft_size, shift_size, win_length, window)), 2))

    diff = torch.pow(X - Y, 2)
    sum_freq = torch.sqrt(torch.sum(diff, dim=-1) / diff.size(-1))
    value = torch.sum(sum_freq, dim=-1) / sum_freq.size(-1)

    return value if value.shape[-1] > 1 else value.item()


VISQOL_PATH = "/cs/labs/adiyoss/moshemandel/visqol-master; "


# from: https://github.com/eagomez2/upf-smc-speech-enhancement-thesis/blob/main/src/utils/evaluation_process.py
def calculate_visqol(out_sig, ref_sig, filename, sr):
    tmp_reference = f"{filename}_ref.wav"
    tmp_estimation = f"{filename}_est.wav"

    reference_abs_path = os.path.abspath(tmp_reference)
    estimation_abs_path = os.path.abspath(tmp_estimation)

    tfm = sox.Transformer()
    tfm.convert(bitdepth=16)
    ref_sig = np.transpose(ref_sig)
    out_sig = np.transpose(out_sig)
    try:
        tfm.build_file(input_array=ref_sig, sample_rate_in=sr, output_filepath=reference_abs_path)
        tfm.build_file(input_array=out_sig, sample_rate_in=sr, output_filepath=estimation_abs_path)

        visqol_cmd = ("cd " + VISQOL_PATH +
                      "./bazel-bin/visqol "
                      f"--reference_file {reference_abs_path} "
                      f"--degraded_file {estimation_abs_path} "
                      f"--use_speech_mode")

        visqol = subprocess.run(visqol_cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # parse stdout to get the current float value
        visqol = visqol.stdout.decode("utf-8").split("\t")[-1].replace("\n", "")
        visqol = float(visqol)

    except Exception as e:
        logger.info(f'failed to get visqol of {filename}')
        logger.info(str(e))
        visqol = 0

    else:
        # remove files to avoid filling space storage
        os.remove(reference_abs_path)
        os.remove(estimation_abs_path)

    return visqol