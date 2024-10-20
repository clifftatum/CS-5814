#!/usr/bin/env python3
import numpy as np
import time
from scipy import signal
import numba
# from numba import cuda
# import torch
# import torch.nn.functional as F
# import cupy as cp
# import multiprocessing
# import itertools
# import cupyx.scipy.signal as cp_signal

@numba.jit(forceobj=True)
def xcor_numba(apple: np.ndarray, banana: np.ndarray) -> np.ndarray:
    '''1D Cross-Correlation'''
    corr = signal.correlate(apple, banana, mode='same', method='fft')
    return np.abs(corr)

# def xcor_cupy(apple: np.ndarray, banana: np.ndarray) -> np.ndarray:
#     '''1D Cross-Correlation'''
#     # corr = cp_signal.correlate(apple, banana, mode='same', method='fft')
#     # x = torch.tensor(apple).cuda()
#     # h = torch.tensor(banana).cuda()
#     # corr = F.conv1d(x.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0), padding=(h.size(0)-1)).cpu()
#     x = cp.array(apple)
#     y = cp.array(banana)
#
#     # Perform correlation using cupyx.signal.correlate
#     corr = cp_signal.correlate(x, y, mode='full')
#
#
#     return cp.abs(corr).get()

def xcor(apple, banana):
    '''1D Cross-Correlation'''
    corr = signal.correlate(apple, banana, mode='same', method='fft')
    return np.abs(corr)

# def apply_fdoa_cupy(ray: cp.ndarray, fdoa: np.float64, samp_rate: np.float64) -> np.ndarray:
#     precache = 2j * np.pi * fdoa / samp_rate
#     new_ray = cp.empty_like(ray)
#     for idx, val in enumerate(ray):
#         new_ray[idx] = val * cp.exp(precache * idx)
#     return new_ray

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

# def amb_surf_cupy(needle: np.ndarray, haystack: np.ndarray, freqs_hz: np.float64, samp_rate: np.float64) -> np.ndarray:
#     len_needle = len(needle)
#     len_haystack = len(haystack)
#     len_freqs = len(freqs_hz)
#     assert len_needle == len_haystack
#     surf = cp.empty((len_freqs, len_needle))
#     for fdx, freq_hz in enumerate(freqs_hz):
#         shifted = apply_fdoa_cupy(needle, freq_hz, samp_rate)
#         surf[fdx] = xcor_cupy(shifted, haystack)
#     return surf

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

# def amb_row_worker(args):
#     needle, haystack, fdoa, samp_rate = args
#     shifted = apply_fdoa(needle, fdoa, samp_rate)
#     return xcor(shifted, haystack)

# def amb_row_worker_numba(args):
#     needle, haystack, fdoa, samp_rate = args
#     shifted = apply_fdoa_numba(needle, fdoa, samp_rate)
#     return xcor_numba(shifted, haystack)

# def amb_surf_multiprocessing(needle, haystack, freqs_hz, samp_rate):
#     len_needle = len(needle)
#     len_haystack = len(haystack)
#     len_freqs = len(freqs_hz)
#     assert len_needle == len_haystack
#     # surf = np.empty((len_freqs, len_needle))
#     with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#         args = zip(
#             itertools.repeat(needle),
#             itertools.repeat(haystack),
#             freqs_hz,
#             itertools.repeat(samp_rate)
#         )
#         res = pool.map(amb_row_worker, args)
#     return np.array(res)

# def amb_surf_multiprocessing_numba(needle, haystack, freqs_hz, samp_rate):
#     len_needle = len(needle)
#     len_haystack = len(haystack)
#     len_freqs = len(freqs_hz)
#     assert len_needle == len_haystack
#     # surf = np.empty((len_freqs, len_needle))
#     with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#         args = zip(
#             itertools.repeat(needle),
#             itertools.repeat(haystack),
#             freqs_hz,
#             itertools.repeat(samp_rate)
#         )
#         res = pool.map(amb_row_worker_numba, args)
#     return np.array(res)

# def amb_surf(needle, haystack, freqs_hz, samp_rate):
#     '''
#     Returns the cross ambiguity function surface for a pair of signals.
#
#     Parameters
#     ----------
#     needle : np.ndarray
#         The signal of interest to localize within the haystack.
#     haystack : np.ndarray
#         The broader capture within which to localize the needle.
#     freqs_hz : np.ndarray
#         The frequency offsets to use in computing the CAF.
#     samp_rate : float
#         The sample rate for both the needle and the haystack.
#
#     Returns
#     -------
#     surf : np.ndarray
#         2D array of correlations of the needle in the haystack over frequency x lag.
#     '''
#     len_needle = len(needle)
#     len_haystack = len(haystack)
#     len_freqs = len(freqs_hz)
#     assert len_needle == len_haystack
#     surf = np.empty((len_freqs, len_needle))
#     for fdx, freq_hz in enumerate(freqs_hz):
#         shifted = apply_fdoa(needle, freq_hz, samp_rate)
#         surf[fdx] = xcor(shifted, haystack)
#     return surf
#
#


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    # import sigmf
    # x_sigmf = sigmf.sigmffile.fromfile(r'C:\Users\frank\Documents\GradSchool\CS-5814 - Intro. Deep Learning\Code\CS-5814\wideband sigmf\bpsk_rect_20sps.sigmf')
    # sample_rate = x_sigmf.get_global_field('core:sample_rate')
    # x_samples = x_sigmf.read_samples()
    # needle_samples = x_samples
    # t = np.arange(len(needle_samples)) / sample_rate
    # freq_shift_hz = 5
    # freq_shift_exponential = np.exp(2j * np.pi * freq_shift_hz * t)
    # haystack_samples = needle_samples * freq_shift_exponential
    # haystack_samples = haystack_samples
    # len_needle = len(needle_samples)

    samp_rate = 250
    duration_seconds = 10
    duration_samples = int(samp_rate*duration_seconds)


    files = [f for f in os.listdir('../data/') if os.path.basename(f).startswith("chirp_1_")]
    needle_filename = files[0]
    haystack_filename = files[1]
    # needle_filename = os.path.basename('../data/chirp_1_raw').startswith("chirp_1_raw")
    # haystack_filename = os.path.basename('../data').startswith("chirp_1_")
    data_dir = '../data'
    # needle_filename = 'chirp_1_raw.c64'
    # haystack_filename = 'chirp_1_T+7samp_F-0.79Hz.c64'
    print(haystack_filename)
    needle_samples = np.fromfile(os.path.join(data_dir, needle_filename), dtype=np.complex64)
    haystack_samples = np.fromfile(os.path.join(data_dir, haystack_filename), dtype=np.complex64)[0:duration_samples]
    len_needle = len(needle_samples)


    freq_offsets = np.arange(-2.5, 2.5, 0.0001)

    # benchmarks
    rounds = 1
    print('running {} rounds per function'.format(rounds))
    # [amb_surf, amb_surf_numba, amb_surf_multiprocessing, amb_surf_multiprocessing_numba]
    # for func in [amb_surf_cupy]:#amb_surf_numba
    #     start = time.time()
        # for _ in range(rounds):
    # surf = amb_surf_cupy(cp.asarray(needle_samples), cp.asarray(haystack_samples), freq_offsets, samp_rate)


    # print(torch.cuda.is_available())
    # torch.cuda.init()
    surf = amb_surf_numba(needle_samples, haystack_samples, freq_offsets, samp_rate)

    # elap = (time.time()-start) / rounds
    fmax, tmax = np.unravel_index(surf.argmax(), surf.shape)
    tau_max = len(needle_samples)//2 - tmax
    freq_max = freq_offsets[fmax]
    # print(amb_surf_cupy.__name__, surf.shape, surf.dtype, '->', tau_max, freq_max)
    # print(func.__name__, 'elap {:.9f} s'.format(elap))

    # plotting
    extents = [
        -len_needle//2, len_needle//2,
        2.5, -2.5]
    plt.figure(dpi=150)
    plt.imshow(np.flip(surf, axis=1), aspect='auto', interpolation='nearest', extent=extents,cmap = 'jet')
    plt.xlim(-250,250)
    current_xticks = plt.xticks()[0]
    new_xticks = [str(round(tick*(1/samp_rate),3)) for tick in current_xticks]
    plt.xticks(current_xticks, new_xticks)
    plt.ylabel('Frequency offset [Hz]')
    plt.xlabel('Time offset [seconds]')

    plt.gca().invert_yaxis()
    # plt.plot(tau_max, freq_max, 'x', color='red', alpha=0.75)
    plt.show()

    print('Time lag: {:f} seconds'.format(tau_max*(1/samp_rate)))
    print('Time lag: {:d} samples'.format(tau_max))
    print('Frequency offset: {:.2f} Hz'.format(freq_max))

    # fig = plt.figure(dpi=150)
    # ax = fig.add_subplot(111, projection='3d')
    # x = tau
    # y = freq_offsets
    # X, Y = np.meshgrid(x, y)
    # Z = surf.reshape(X.shape)
    #
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    #
    # ax.set_xlabel('Frequency offset [Hz]')
    # ax.set_ylabel('Lag [samples]')
    # ax.set_zlabel('Correlation')
    # plt.show()