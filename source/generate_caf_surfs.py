
import numpy as np
import scipy.signal as sig
import os
import shutil
import scipy as sp
from scipy import signal
import numba
import matplotlib.pyplot as plt
from datetime import datetime

def apply_offset(signal, dfc, sample_rate):
    '''
    Applies a constant or time-varying frequency offset to the signal.
    Returns a copy of the signal with the frequency offset applied.
    '''
    if type(dfc) in (int, float):
        shift = np.exp(1j*2*np.pi*dfc*np.arange(len(signal))/sample_rate)
    else:
        phi = np.cumsum(2*np.pi*dfc) / sample_rate
        shift = np.exp(1j*(np.arange(len(signal))/sample_rate+phi))
    return shift*signal

def smear(sig,sample_rate, chirp_length=4096, chirp_order=2, relative_bandwidth=1e-2, sweep_range_Hz=10e3, window=np.hanning):
    # Generate some shaped noise
    kernel = sp.signal.firwin(127, cutoff=0.5, fs=sample_rate)
    # chirp = np.random.normal(0, .1, chirp_length) + 1j*np.random.normal(0, .1, chirp_length)
    chirp = sig
    # LPF
    chirp = sp.signal.filtfilt(kernel, 1, chirp)

    # Taper the edges
    if window is not None:
        chirp = window(chirp_length)*chirp
    chirp = chirp.astype(np.complex64)

    # Induce doppler
    shape = np.linspace(-1, 1, chirp_length)**chirp_order
    offset_Hz = shape*sweep_range_Hz
    chirp = apply_offset(chirp, offset_Hz, sample_rate)

    return chirp

def generate_fsk_signal(f0, f1, mode, fsk_length, baud_rate, sample_rate):
    """
    Generates a baseband FSK signal in coherent or non-coherent mode.

    Parameters:
    - f0: Frequency for bit 0 (Hz).
    - f1: Frequency for bit 1 (Hz).
    - mode: 'coherent' for random 0/1 sequence, 'non-coherent' for alternating 0/1 sequence.
    - fsk_length: Length of the output signal (in samples).
    - baud_rate: Baud rate (bits per second).
    - sample_rate: Sampling rate (samples per second).

    Returns:
    - fsk_signal: Generated FSK signal.
    - t: Time vector for the signal.
    - bitstream: The generated bitstream (sequence of 0s and 1s).
    """

    # Calculate number of samples per bit based on baud rate
    samples_per_bit = int(sample_rate / baud_rate)
    num_bits = int(fsk_length / samples_per_bit)
    segment_length = int(num_bits//2)

    # Generate bitstream
    if mode == 'coherent':
        # np.random.seed(0)  # Optional: For reproducibility
        bitstream = np.random.randint(0, 2, num_bits)
    if mode == 'non-coherent':
        bitstream = np.random.randint(0, 2, num_bits)
        start = np.random.randint(0, len(bitstream) - segment_length + 1)
        end = start + segment_length
        ones_zeros_pattern = np.tile([1, 0], segment_length // 2 + 1)[:segment_length]
        bitstream[start:end] = ones_zeros_pattern

    # elif mode == 'non-coherent':
    #     bitstream = np.tile([0, 1], num_bits // 2)
    #     if num_bits % 2 != 0:
    #         bitstream = np.append(bitstream, 0)  # To keep the correct length





    # Time vector for the entire signal
    t = np.arange(fsk_length) / sample_rate

    # Initialize FSK signal
    fsk_signal = np.zeros(fsk_length)

    # Generate FSK signal by modulating with f0 and f1 based on bitstream
    for i, bit in enumerate(bitstream):
        f = f1 if bit == 1 else f0
        t_bit = t[i * samples_per_bit: (i + 1) * samples_per_bit]
        fsk_signal[i * samples_per_bit: (i + 1) * samples_per_bit] = np.sin(2 * np.pi * f * t_bit)

    return fsk_signal, t

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

    # np.random.seed(0)
    data_dir = '../data'
    samp_rate = 250
    duration_seconds = 10
    duration_samples = samp_rate*duration_seconds


    baud_rate = 100  # Baud rate in bits per second
    sample_rate = samp_rate  # Sample rate in samples per second
    f0 = -10  # Frequency for bit 0 in Hz
    f1 = 10  # Frequency for bit 1 in Hz
    # mode = 'non-coherent'
    chirp_length = int(duration_samples)
    chirp_order = 1

    total_span = 25
    relative_bandwidth = 1 #np.linspace(start=3,stop=5,num=total_span)#np.random.uniform(1e-3, 10) # Width of chirp, relative to sample rate
    sweep_range_Hz = np.linspace(start=.1,stop=3,num=total_span)#np.random.uniform(5, 100) # Range of chirp
    time_delays = np.linspace(start=sample_rate*2,stop=0,num=total_span,dtype=int)
    freq_delays = np.linspace(start=-2.4, stop=2.4, num=total_span)

    # oct_26_ind_beg = 0
    # oct_26_ind_end = 20
    # sweep_range_Hz = sweep_range_Hz[oct_26_ind_beg:oct_26_ind_end]
    # time_delays = time_delays[oct_26_ind_beg:oct_26_ind_end]
    # freq_delays = freq_delays[oct_26_ind_beg:oct_26_ind_end]

    total_iters = 4*(total_span**3)
    # oct_26_span= 2*(oct_26_ind_end**3)
    c = 0
    print(datetime.now())
    freq_offsets = np.arange(-2.5, 2.5, 0.0001)

    coherent_fsk,_ = generate_fsk_signal(f0, f1, 'coherent', duration_samples, baud_rate, sample_rate)
    non_coherent_fsk,_ = generate_fsk_signal(f0, f1, 'non-coherent', duration_samples, baud_rate, sample_rate)

    for smear_hz in sweep_range_Hz:
        for dt in time_delays:
            for df in freq_delays:
                for mode in ['coherent', 'non-coherent']:
                    fsk_signal, t = generate_fsk_signal(f0, f1, mode, duration_samples, baud_rate, sample_rate)

                    if mode =='coherent':
                        fsk_signal = coherent_fsk
                    elif mode =='non-coherent':
                        fsk_signal = non_coherent_fsk

                    c+=2
                    signal_wf = smear( sig =fsk_signal,
                                       chirp_length=chirp_length,
                                       chirp_order=float(chirp_order),
                                       relative_bandwidth=float(relative_bandwidth),
                                       sweep_range_Hz=float(smear_hz),
                                       sample_rate=samp_rate)
                    template_chirp = signal_wf.astype(np.complex64)
                    # template_chirp.tofile(os.path.join(data_dir, 'chirp_{:d}_raw.c64'.format(0)))

                    # Time shift
                    lag = dt
                    signal_channel = np.concatenate([np.zeros(lag), template_chirp, np.zeros(96)])

                    # Add noise
                    signal_channel += np.random.normal(0, 1, len(signal_channel)) + 1j * np.random.normal(0, 1,
                                                                              len(signal_channel))

                    # Freq shift
                    signal_channel = apply_offset(signal_channel, float(df), samp_rate)


                    signal_channel = signal_channel.astype(np.complex64)
                    # signal_channel.tofile(os.path.join(data_dir, 'chirp_{:d}_T{:+d}samp_F{:+.2f}Hz.c64'.format(0, lag, df)))

                    surf = amb_surf_numba(template_chirp, signal_channel[0:duration_samples], freq_offsets, samp_rate)

                    fmax, tmax = np.unravel_index(surf.argmax(), surf.shape)
                    tau_max = len(template_chirp) // 2 - tmax
                    freq_max = freq_offsets[fmax]

                    extents = [
                        -len(template_chirp) // 2, len(template_chirp) // 2,
                        freq_offsets[-1], freq_offsets[0]]
                    plt.figure(dpi=150)
                    plt.axis('off')
                    plt.imshow(np.flip(surf, axis=1), aspect='auto', interpolation='nearest', extent=extents,
                               cmap='jet')
                    # Must make this correction(author bug)
                    plt.gca().invert_yaxis()
                    path = r'C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\images'
                    name = mode + '_time_lag_sec{:.5f}_freq_shift_hz{:.5f}_smear_{:.5f}.png'.format(
                        tau_max * (1 / samp_rate),
                        df,
                        smear_hz)
                    plt.savefig(os.path.join(path,name), bbox_inches='tight', pad_inches=0)
                    # plt.show()
                      # Hide axes
                    # plt.ylabel('Frequency offset [Hz]')
                    # plt.xlabel('Time lag [samples]')
                    # This is for the time delay reflection
                    plt.gca().invert_xaxis()
                    name = mode + '_time_lag_sec{:.5f}_freq_shift_hz{:.5f}_smear_{:.5f}.png'.format(
                        -tau_max * (1 / samp_rate),
                        df,
                        smear_hz)
                    plt.savefig(os.path.join(path,name), bbox_inches='tight', pad_inches=0)

                    # plt.show()
                    plt.close()



                    print(str(c)+'/'+ str(total_iters)+ ' '+ name)

    print('Finished: '+ str(datetime.now()))