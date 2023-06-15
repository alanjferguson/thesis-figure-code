import os
dirpath = os.path.dirname(__file__)

import numpy as np
import pandas as pd

from scipy import signal
from scipy.fftpack import fft

import numba as nb
from numba import njit

################################################################################
# RESAMPLE NOISE
################################################################################
def get_noise(req_Fs):
    noise_df = pd.read_parquet(dirpath + "/noise_data.parquet")
    noise_Fs = 1.0 / noise_df.t.diff().mean()
    if req_Fs is not None:
        (z, t) = signal.resample(x=noise_df.z,
                                 num=int(len(noise_df.z) / (noise_Fs / req_Fs)),
                                 t=noise_df.t)
    else:
        z = noise_df.z.values
        t = noise_df.t.values
    return t, z


################################################################################
# SURROGATE NOISE
################################################################################
@njit(cache=True, parallel=False)
def fftnoise(f):
    Np = (len(f) - 1) // 2
    fft = f.astype(np.complex128)
    phases = np.random.rand(Np).astype(np.complex128) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    fft[1 : Np + 1] *= phases
    fft[-1 : -1 - Np : -1] = np.conj(fft[1 : Np + 1])
    with nb.objmode(noise="float64[:]"):
        noise = np.fft.ifft(fft).real
    return noise


@njit(cache=True, parallel=False)
def _get_surrogate_noise(fft, n_samples):
    surr_noise = fftnoise(fft)
    return surr_noise[:n_samples]


@njit(cache=True, parallel=False)
def _add_noise(s, fft, scale_factor=1.0):
    return s + _get_surrogate_noise(fft*scale_factor, len(s))


@njit(cache=True, parallel=False)
def _pad_and_add_noise(s, PAD_LEN, fft, scale_factor=1.0):
    y = np.zeros(len(s) + 2 * PAD_LEN)
    y[PAD_LEN:-PAD_LEN] = s
    y += _get_surrogate_noise(fft*scale_factor, len(y))
    return y


def lp_filter(s, Fs, Fc=1.0, order=4):
    sos = signal.iirfilter(
        N=order,
        Wn=Fc,
        btype="lowpass",
        ftype="butter",
        fs=Fs,
        output="sos",
    )
    return signal.sosfilt(sos, s)


@njit(cache=True, parallel=False)
def _process_fe_sig(s, PAD_LEN, Fs, fft, noise_Fs):
    # PAD + ADD NOISE
    noisy = _pad_and_add_noise(s, PAD_LEN, Fs, fft, noise_Fs)
    # DETREND
    noisy -= np.mean(noisy[:Fs])
    # FILTER
    with nb.objmode(filt="float64[:]"):
        filt = lp_filter(noisy, Fs)
    # UNPAD
    unpad_filt = unpad(filt, PAD_LEN)
    # Get prominence
    return np.max(unpad_filt) - np.min(unpad_filt)


def process_fe_sig(s, Fs):
    return _process_fe_sig(s, int(Fs), Fs, _noise_FFT, _noise_Fs)

class NoiseModel(object):
    def __init__(self, req_Fs=None):
        t, z = get_noise(req_Fs)
        self._noise_N = len(t)
        self._noise_FFT = fft(z)

    def get_surrogate_noise(self, n_samples=None):
        if n_samples is None:
            n_samples = self._noise_N
        return _get_surrogate_noise(self._noise_FFT, n_samples)
    
    def add_noise(self, s, use_g=False):
        scale_factor = 1.0 / 9.818 if use_g else 1.0
        return _add_noise(s, self._noise_FFT, scale_factor=scale_factor)

    def pad_and_add_noise(self, s, PAD_LEN=1000, use_g=True):
        scale_factor = 1.0 / 9.818 if use_g else 1.0
        return _pad_and_add_noise(s, PAD_LEN, self._noise_FFT, scale_factor=scale_factor)
    
    def unpad(self, s, PAD_LEN=1000):
        return s[PAD_LEN:-PAD_LEN]


        

    
