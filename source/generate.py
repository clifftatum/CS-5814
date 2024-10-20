#!/usr/bin/env python3
'''Make some chirpy things for testing'''
import numpy as np
import scipy.signal as sig
import os
import scipy.interpolate as interp
import scipy as sp
import scipy.signal

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

def generate_chirp(sample_rate, chirp_length=4096, chirp_order=2, relative_bandwidth=1e-2, sweep_range_Hz=10e3, window=np.hanning):
    # Generate some shaped noise
    kernel = sp.signal.firwin(127, cutoff=0.3*relative_bandwidth, fs=sample_rate)
    chirp = np.random.normal(0, 1, chirp_length) + 1j*np.random.normal(0, 1, chirp_length)
    # LPF
    chirp = sp.signal.filtfilt(kernel, 1, chirp)

    # Taper the edges
    if window is not None:
        chirp = window(chirp_length)*chirp
    chirp = chirp.astype(np.complex64)

    # Make it move
    shape = np.linspace(-1, 1, chirp_length)**chirp_order
    # Induce doppler
    offset_Hz = shape*sweep_range_Hz
    chirp = apply_offset(chirp, offset_Hz, sample_rate)

    return chirp


import numpy as np
import matplotlib.pyplot as plt


def generate_qpsk_signal(mode, fsk_length, baud_rate, sample_rate):
    """
    Generates a baseband QPSK signal.

    Parameters:
    - mode: 'coherent' for random bitstream, 'non-coherent' for alternating 0/1 sequence.
    - fsk_length: Length of the output signal (in samples).
    - baud_rate: Baud rate (symbols per second).
    - sample_rate: Sampling rate (samples per second).

    Returns:
    - qpsk_signal: Generated QPSK signal.
    - t: Time vector for the signal.
    - bitstream: The generated bitstream.
    """

    # QPSK encodes 2 bits per symbol, so the symbol rate is half the bit rate
    symbol_rate = baud_rate / 2
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = int(fsk_length / samples_per_symbol)

    # Generate bitstream
    if mode == 'coherent':
        np.random.seed(0)
        bitstream = np.random.randint(0, 2, num_symbols * 2)
    elif mode == 'non-coherent':
        bitstream = np.tile([0, 1], num_symbols)

    # Time vector for the entire signal
    t = np.arange(fsk_length) / sample_rate

    # Initialize QPSK signal
    qpsk_signal = np.zeros(fsk_length)

    # Phase mapping for QPSK (00 -> 0, 01 -> 90, 10 -> 180, 11 -> 270 degrees)
    phase_map = {
        (0, 0): 0,
        (0, 1): np.pi / 2,
        (1, 0): np.pi,
        (1, 1): 3 * np.pi / 2
    }

    # Generate QPSK signal
    for i in range(num_symbols):
        bit_pair = (bitstream[2 * i], bitstream[2 * i + 1])
        phase = phase_map[bit_pair]

        t_symbol = t[i * samples_per_symbol: (i + 1) * samples_per_symbol]
        qpsk_signal[i * samples_per_symbol: (i + 1) * samples_per_symbol] = np.cos(
            2 * np.pi * symbol_rate * t_symbol + phase)

    return qpsk_signal, t
def generate_msk_signal(f0, f1, mode, fsk_length, baud_rate, sample_rate):
    """
    Generates a baseband MSK signal.

    Parameters:
    - f0: Frequency for bit 0 (Hz).
    - f1: Frequency for bit 1 (Hz).
    - mode: 'coherent' for random 0/1 sequence, 'non-coherent' for alternating 0/1 sequence.
    - fsk_length: Length of the output signal (in samples).
    - baud_rate: Baud rate (bits per second).
    - sample_rate: Sampling rate (samples per second).

    Returns:
    - msk_signal: Generated MSK signal.
    - t: Time vector for the signal.
    - bitstream: The generated bitstream (sequence of 0s and 1s).
    """

    # Calculate number of samples per bit based on baud rate
    samples_per_bit = int(sample_rate / baud_rate)
    num_bits = int(fsk_length / samples_per_bit)

    # Generate bitstream
    if mode == 'coherent':
        np.random.seed(0)
        bitstream = np.random.randint(0, 2, num_bits)
    elif mode == 'non-coherent':
        bitstream = np.tile([0, 1], num_bits // 2)
        if num_bits % 2 != 0:
            bitstream = np.append(bitstream, 0)

    # Time vector for the entire signal
    t = np.arange(fsk_length) / sample_rate

    # Calculate the frequency shift for MSK
    delta_f = 1 / (2 * (1 / baud_rate))  # Minimum frequency shift for MSK

    # Initialize MSK signal
    msk_signal = np.zeros(fsk_length)

    # Phase accumulation for continuous phase
    phase = 0

    # Generate MSK signal
    for i, bit in enumerate(bitstream):
        # Choose frequency based on bit value
        f = f1 if bit == 1 else f0
        f = delta_f if bit == 1 else -delta_f

        t_bit = t[i * samples_per_bit: (i + 1) * samples_per_bit]
        msk_signal[i * samples_per_bit: (i + 1) * samples_per_bit] = np.sin(2 * np.pi * f * t_bit + phase)

        # Update phase to ensure continuity
        phase += 2 * np.pi * f * (samples_per_bit / sample_rate)

    return msk_signal, t



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

    # Generate bitstream
    if mode == 'coherent':
        np.random.seed(0)  # Optional: For reproducibility
        bitstream = np.random.randint(0, 2, num_bits)
    elif mode == 'non-coherent':
        bitstream = np.tile([0, 1], num_bits // 2)
        if num_bits % 2 != 0:
            bitstream = np.append(bitstream, 0)  # To keep the correct length

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


if __name__ == '__main__':


    np.random.seed(0)
    data_dir = '../data'
    samp_rate = 250
    duration_seconds = 10
    duration_samples = samp_rate*duration_seconds

    # Parameters
    baud_rate = 50  # Baud rate in bits per second
    sample_rate = samp_rate  # Sample rate in samples per second
    f0 = -10  # Frequency for bit 0 in Hz
    f1 = 10  # Frequency for bit 1 in Hz
    np.random.seed(0)  # For reproducibility
    bitstream = np.random.randint(0, 2, duration_samples//baud_rate)
    bitstream = [(i % 2) for i in range(duration_samples//baud_rate)]
    mode = 'non-coherent'
    fsk_signal, t = generate_fsk_signal(f0, f1, mode, duration_samples, baud_rate, sample_rate)
    # fsk_signal, t = generate_msk_signal(f0, f1, mode, duration_samples, baud_rate, sample_rate)
    # fsk_signal, t = generate_qpsk_signal(mode, duration_samples, baud_rate, sample_rate)



    chirp_length = int(duration_samples)
    # This can stay 1 doesnt really matter much
    chirp_order = 1 #np.random.randint(0,1) # Determines shape of chirp

    # No Higher than 5 I say
    relative_bandwidth = 1#np.random.uniform(1e-3, 10) # Width of chirp, relative to sample rate

    # No Higher than 10 I think
    sweep_range_Hz = 2#np.random.uniform(5, 100) # Range of chirp

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for idx in range(10):
        dfc_range_Hz = 1 # Range of frequency offsets for search capture
        lag = 200 #np.random.randint(7, 15) # Lag (in samples) of SOI in search capture
        chirp = generate_chirp(chirp_length=chirp_length, chirp_order=chirp_order, relative_bandwidth=relative_bandwidth, sweep_range_Hz=sweep_range_Hz, sample_rate=samp_rate)
        chirp = chirp.astype(np.complex64)
        chirp.tofile(os.path.join(data_dir, 'chirp_{:d}_raw.c64'.format(idx)))

        # N = len(chirp)
        # signal_fft = np.fft.fft(chirp)
        # signal_fft = np.fft.fftshift(chirp)  # Shift zero freq to center
        # freq_axis = np.fft.fftfreq(N, d=1 / sample_rate)
        # freq_axis = np.fft.fftshift(freq_axis)  # Shift zero freq to center
        # magnitude_spectrum = np.abs(chirp) / N
        # plt.figure(figsize=(10, 6))
        # plt.plot(freq_axis, 10 * np.log10(magnitude_spectrum))
        # plt.title("Two-Sided Baseband Spectrum of 2-FSK Signal")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Magnitude")
        # plt.grid(True)
        # plt.show()


        # Add a random time lag
        foffset = np.random.uniform(-dfc_range_Hz, dfc_range_Hz)
        chirp_search = np.concatenate([np.zeros(lag), chirp, np.zeros(96)])
        chirp_search = apply_offset(chirp_search, foffset, samp_rate)

        # Add some noise
        chirp_search += np.random.normal(0, 1e-1, len(chirp_search)) + 1j*np.random.normal(0, 1e-1, len(chirp_search))
        chirp_search = chirp_search.astype(np.complex64)
        chirp_search.tofile(os.path.join(data_dir, 'chirp_{:d}_T{:+d}samp_F{:+.2f}Hz.c64'.format(idx, lag, foffset)))