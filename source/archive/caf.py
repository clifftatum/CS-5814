#!/usr/bin/env python3
import numpy as np

from scipy import signal
import numba


@numba.jit(forceobj=True)
def xcor_numba(apple: np.ndarray, banana: np.ndarray) -> np.ndarray:
    '''1D Cross-Correlation'''
    corr = signal.correlate(apple, banana, mode='same', method='fft')
    return np.abs(corr)

def xcor(apple, banana):
    '''1D Cross-Correlation'''
    corr = signal.correlate(apple, banana, mode='same', method='fft')
    return np.abs(corr)

@numba.njit
def apply_fdoa_numba(ray: np.ndarray, fdoa: np.float64, samp_rate: np.float64) -> np.ndarray:
    precache = 2j * np.pi * fdoa / samp_rate
    new_ray = np.empty_like(ray)
    for idx, val in enumerate(ray):
        new_ray[idx] = val * np.exp(precache * idx)
    return new_ray

def apply_fdoa(ray, fdoa, samp_rate):
    precache = 2j * np.pi * fdoa / samp_rate
    new_ray = np.empty_like(ray)
    for idx, val in enumerate(ray):
        new_ray[idx] = val * np.exp(precache * idx)
    return new_ray

@numba.jit(forceobj=True)
def amb_surf_numba(needle: np.ndarray, haystack: np.ndarray, freqs_hz: np.float64, samp_rate: np.float64) -> np.ndarray:
    len_needle = len(needle)
    len_haystack = len(haystack)
    len_freqs = len(freqs_hz)
    assert len_needle == len_haystack
    surf = np.empty((len_freqs, len_needle))
    for fdx, freq_hz in enumerate(freqs_hz):
        # print(fdx)
        shifted = apply_fdoa_numba(needle, freq_hz, samp_rate)
        surf[fdx] = xcor_numba(shifted, haystack)
    return surf


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    samp_rate = 250
    duration_seconds = 10
    duration_samples = int(samp_rate*duration_seconds)


    files = [f for f in os.listdir('../../data/') if os.path.basename(f).startswith("chirp_0_")]
    needle_filename = files[0]
    haystack_filename = files[1]

    data_dir = '../../data'

    print(haystack_filename)
    needle_samples = np.fromfile(os.path.join(data_dir, needle_filename), dtype=np.complex64)
    haystack_samples = np.fromfile(os.path.join(data_dir, haystack_filename), dtype=np.complex64)[0:duration_samples]
    len_needle = len(needle_samples)


    freq_offsets = np.arange(-2.5, 2.5, 0.0001)

    # benchmarks
    rounds = 1
    print('running {} rounds per function'.format(rounds))

    surf = amb_surf_numba(needle_samples, haystack_samples, freq_offsets, samp_rate)


    fmax, tmax = np.unravel_index(surf.argmax(), surf.shape)
    tau_max = len(needle_samples)//2 - tmax
    freq_max = freq_offsets[fmax]


    # plotting
    extents = [
        -len_needle//2, len_needle//2,
        freq_offsets[-1], freq_offsets[0]]
    plt.figure(dpi=150)
    plt.imshow(np.flip(surf, axis=1), aspect='auto', interpolation='nearest', extent=extents,cmap = 'jet')
    # plt.xlim(-250,250)
    # current_xticks = plt.xticks()[0]
    # new_xticks = [str(round(tick*(1/samp_rate),3)) for tick in current_xticks]
    # plt.xticks(current_xticks, new_xticks)

    plt.ylabel('Frequency offset [Hz]')
    plt.xlabel('Time lag [samples]')
    plt.gca().invert_yaxis()
    plt.show()

    print('Time lag: {:f} seconds'.format(tau_max*(1/samp_rate)))
    print('Time lag: {:d} samples'.format(tau_max))
    print('Frequency offset: {:.2f} Hz'.format(freq_max))
