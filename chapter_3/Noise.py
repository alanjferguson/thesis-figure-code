import numpy as np
import pandas as pd

from scipy import signal
from scipy.fftpack import fft

import numba as nb
from numba import njit

################################################################################
# READ IN NOISE DATA AND SETUP GLOBAL VARS
################################################################################

_noise_df = pd.read_parquet("noise_data.parquet")

_noise_t = _noise_df["t"].to_numpy()
_noise_z = _noise_df["z"].to_numpy()

_noise_Ts = np.mean(np.diff(_noise_t))
_noise_Fs = 1.0 / _noise_Ts

_noise_N = len(_noise_t)

_noise_FFT = fft(_noise_z)

################################################################################
# RESAMPLE NOISE
################################################################################


@njit(cache=True, parallel=False, fastmath=True)
def get_noise(req_Fs=_noise_Fs):
    t = _noise_t
    z = _noise_z
    if req_Fs < _noise_Fs:
        Fs = _noise_Fs
        while Fs / req_Fs > 10:
            t = signal.decimate(t, 10)
            z = signal.decimate(z, 10)
            Fs /= 10
        t = signal.decimate(t, int(Fs / req_Fs))
        z = signal.decimate(z, int(Fs / req_Fs))
    # TODO: handle upsampling case, or maybe just error out?
    return t, z


################################################################################
# SURROGATE NOISE
################################################################################
@njit(cache=True, parallel=False, fastmath=True)
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


@njit(cache=True, parallel=False, fastmath=True)
def _get_surrogate_noise(fft, noise_Fs, req_Fs, n_samples):
    surr_noise = fftnoise(fft)
    if req_Fs < noise_Fs:
        with nb.objmode():
            Fs = noise_Fs
            while Fs / req_Fs > 10:
                surr_noise = signal.decimate(surr_noise, 10)
                Fs /= 10
            surr_noise = signal.decimate(surr_noise, int(Fs / req_Fs))
    # TODO: handle upsampling case, or maybe just error out?
    return surr_noise[:n_samples]


def get_surrogate_noise(req_Fs=_noise_Fs, n_samples=_noise_N):
    return _get_surrogate_noise(_noise_FFT, _noise_Fs, req_Fs, n_samples)


@njit(cache=True, parallel=False, fastmath=True)
def _add_noise(s, fft, noise_Fs, req_Fs):
    return s + _get_surrogate_noise(fft, noise_Fs, req_Fs, len(s))


def add_noise(s, Fs=_noise_Fs):
    return _add_noise(s, _noise_FFT, _noise_Fs, Fs)


@njit(cache=True, parallel=False, fastmath=True)
def _pad_and_add_noise(s, PAD_LEN, Fs, fft, noise_Fs):
    y = np.zeros(len(s) + 2 * PAD_LEN)
    y[PAD_LEN:-PAD_LEN] = s
    y += _get_surrogate_noise(fft, noise_Fs, Fs, len(y))
    return y


def pad_and_add_noise(s, PAD_LEN=1000, Fs=_noise_Fs):
    return _pad_and_add_noise(s, PAD_LEN, Fs, _noise_FFT, _noise_Fs)


@njit(cache=True, parallel=False, fastmath=True)
def unpad(s, PAD_LEN=1000):
    return s[PAD_LEN:-PAD_LEN]


def lp_filter(s, FS):
    FC = 1.0
    ORDER = 4
    b, a = signal.iirfilter(
        N=ORDER,
        Wn=FC,
        btype="lowpass",
        ftype="butter",
        analog=False,
        fs=FS,
        output="ba",
    )
    return signal.lfilter(b, a, s)


@njit(cache=True, parallel=False, fastmath=True)
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
    return _process_fe_sig(s, Fs, Fs, _noise_FFT, _noise_Fs)
