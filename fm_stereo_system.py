"""
FM Stereo Broadcasting System

Complete implementation of FM stereo transmitter and receiver for analyzing
frequency deviation effects on bandwidth and SNR.

Components:
- Transmitter: Stereo multiplexer → Pre-emphasis → FM modulator
- Receiver: FM demodulator → De-emphasis → Stereo decoder

Author: FM Stereo System Project
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


class FMStereoTransmitter:
    """FM Stereo Transmitter with stereo multiplexing, pre-emphasis, and FM modulation."""
    
    def __init__(self, fs_audio=44100, fs_carrier=1e6, fc=100e6, delta_f=75e3):
        """
        Initialize FM Stereo Transmitter.
        
        Args:
            fs_audio: Audio sampling frequency (Hz)
            fs_carrier: Carrier sampling frequency (Hz) - for simulation
            fc: Carrier frequency (Hz) - 100 MHz typical FM broadcast
            delta_f: Maximum frequency deviation (Hz)
        """
        self.fs_audio = fs_audio
        self.fs_carrier = fs_carrier
        self.fc = fc
        self.delta_f = delta_f
        
        # FM Stereo standard frequencies
        self.pilot_freq = 19e3  # 19 kHz pilot
        self.subcarrier_freq = 38e3  # 38 kHz subcarrier (2 * pilot)
        
        # Pre-emphasis time constant (75 μs for US/Japan)
        self.preemphasis_tau = 75e-6
        
    def create_preemphasis_filter(self, fs):
        """
        Create pre-emphasis filter (75 μs time constant).
        
        Pre-emphasis boosts high frequencies to improve SNR.
        H(s) = (1 + s*tau) / (1 + s*tau/10)
        """
        tau = self.preemphasis_tau
        # Design as first-order high-shelf filter
        # Zero at 1/(2*pi*tau) = 2122 Hz
        # Pole at 10/(2*pi*tau) = 21220 Hz
        wz = 1 / tau
        wp = 10 / tau
        
        # Bilinear transform to digital
        b, a = signal.bilinear([tau, 1], [tau/10, 1], fs)
        return b, a
    
    def stereo_multiplex(self, left, right, fs):
        """
        Create FM stereo composite signal.
        
        Composite signal structure:
        - L+R sum (0-15 kHz)
        - 19 kHz pilot tone
        - (L-R) DSB-SC on 38 kHz subcarrier (23-53 kHz)
        
        Args:
            left: Left channel audio
            right: Right channel audio
            fs: Sampling frequency
            
        Returns:
            Composite stereo signal
        """
        n = len(left)
        t = np.arange(n) / fs
        
        # L+R and L-R signals
        sum_signal = (left + right) / 2
        diff_signal = (left - right) / 2
        
        # 19 kHz pilot tone (10% of max deviation)
        pilot = 0.1 * np.sin(2 * np.pi * self.pilot_freq * t)
        
        # L-R on 38 kHz subcarrier (DSB-SC modulation)
        subcarrier = np.sin(2 * np.pi * self.subcarrier_freq * t)
        dsb_sc = diff_signal * subcarrier
        
        # Composite signal
        # Standard levels: L+R = 90%, pilot = 10%, L-R = 90%
        composite = 0.45 * sum_signal + pilot + 0.45 * dsb_sc
        
        return composite
    
    def fm_modulate(self, message, fs):
        """
        FM modulate the message signal.
        
        For simulation purposes, we work at baseband and use the
        instantaneous phase/frequency relationship.
        
        s(t) = A * cos(2*pi*fc*t + 2*pi*kf * integral(m(τ))dτ)
        
        Args:
            message: Message signal (composite stereo signal)
            fs: Sampling frequency
            
        Returns:
            FM modulated signal (complex baseband representation)
        """
        # Normalize message to [-1, 1]
        message_norm = message / (np.max(np.abs(message)) + 1e-10)
        
        # Cumulative integral for phase
        # Phase deviation = 2*pi*delta_f * integral(m(t))dt
        dt = 1 / fs
        phase = 2 * np.pi * self.delta_f * np.cumsum(message_norm) * dt
        
        # Complex baseband FM signal
        fm_signal = np.exp(1j * phase)
        
        return fm_signal, message_norm
    
    def transmit(self, left, right):
        """
        Full transmitter chain: multiplex → pre-emphasis → FM modulation.
        
        Args:
            left: Left channel audio
            right: Right channel audio
            
        Returns:
            fm_signal: Complex FM modulated signal
            composite: Stereo composite signal (for debugging)
        """
        # Resample to higher rate for FM processing (at least 2x composite BW)
        # Composite BW is 53 kHz, so we need at least 106 kHz
        # Use 200 kHz for margin
        fs_composite = 200e3
        
        # Resample audio to composite rate
        resample_ratio = fs_composite / self.fs_audio
        n_resampled = int(len(left) * resample_ratio)
        left_resampled = signal.resample(left, n_resampled)
        right_resampled = signal.resample(right, n_resampled)
        
        # Create stereo composite
        composite = self.stereo_multiplex(left_resampled, right_resampled, fs_composite)
        
        # Apply pre-emphasis
        b, a = self.create_preemphasis_filter(fs_composite)
        composite_preemph = signal.lfilter(b, a, composite)
        
        # FM modulate
        fm_signal, composite_norm = self.fm_modulate(composite_preemph, fs_composite)
        
        return fm_signal, composite_preemph, fs_composite


class FMStereoReceiver:
    """FM Stereo Receiver with FM demodulation, de-emphasis, and stereo decoding."""
    
    def __init__(self, fs_audio=44100, delta_f=75e3):
        """
        Initialize FM Stereo Receiver.
        
        Args:
            fs_audio: Output audio sampling frequency (Hz)
            delta_f: Expected frequency deviation (Hz)
        """
        self.fs_audio = fs_audio
        self.delta_f = delta_f
        
        # FM Stereo standard frequencies
        self.pilot_freq = 19e3
        self.subcarrier_freq = 38e3
        
        # De-emphasis time constant
        self.deemphasis_tau = 75e-6
        
    def create_deemphasis_filter(self, fs):
        """
        Create de-emphasis filter (inverse of pre-emphasis).
        
        H(s) = (1 + s*tau/10) / (1 + s*tau)
        """
        tau = self.deemphasis_tau
        b, a = signal.bilinear([tau/10, 1], [tau, 1], fs)
        return b, a
    
    def fm_demodulate(self, fm_signal, fs):
        """
        FM demodulate using phase differentiation.
        
        The instantaneous frequency is the derivative of the phase.
        
        Args:
            fm_signal: Complex FM signal
            fs: Sampling frequency
            
        Returns:
            Demodulated message signal
        """
        # Get instantaneous phase
        phase = np.unwrap(np.angle(fm_signal))
        
        # Differentiate to get frequency deviation
        # freq = (1/2π) * d(phase)/dt
        freq = np.diff(phase) * fs / (2 * np.pi)
        
        # Normalize by frequency deviation to get message
        message = freq / self.delta_f
        
        # Pad to match input length
        message = np.append(message, message[-1])
        
        return message
    
    def extract_pilot(self, composite, fs):
        """
        Extract and phase-lock to the 19 kHz pilot tone.
        
        Returns:
            Pilot signal for synchronous detection
            Double-frequency signal for L-R demodulation
        """
        # Bandpass filter around 19 kHz
        f_low = 18.5e3
        f_high = 19.5e3
        sos = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
        pilot_filtered = signal.sosfilt(sos, composite)
        
        # Normalize pilot
        pilot_filtered = pilot_filtered / (np.max(np.abs(pilot_filtered)) + 1e-10)
        
        # Generate doubled frequency (38 kHz) for L-R demodulation
        # Using squaring and filtering, or simply generate from recovered pilot
        n = len(composite)
        t = np.arange(n) / fs
        
        # Simple approach: square the pilot and filter
        # Better approach: PLL, but for simulation this works
        pilot_doubled = 2 * pilot_filtered ** 2 - 1  # cos(2θ) = 2cos²(θ) - 1
        
        # Bandpass around 38 kHz
        f_low = 37e3
        f_high = 39e3
        sos38 = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
        subcarrier = signal.sosfilt(sos38, pilot_doubled)
        subcarrier = subcarrier / (np.max(np.abs(subcarrier)) + 1e-10)
        
        return pilot_filtered, subcarrier
    
    def stereo_decode(self, composite, fs):
        """
        Decode stereo composite signal to L and R channels.
        
        Args:
            composite: Demodulated stereo composite signal
            fs: Sampling frequency
            
        Returns:
            left, right: Decoded audio channels
        """
        # Extract L+R (lowpass at 15 kHz)
        sos_lpf = signal.butter(6, 15e3, btype='low', fs=fs, output='sos')
        sum_signal = signal.sosfilt(sos_lpf, composite)
        
        # Extract pilot and generate 38 kHz subcarrier
        pilot, subcarrier = self.extract_pilot(composite, fs)
        
        # Extract L-R DSB-SC signal (bandpass 23-53 kHz)
        sos_bpf = signal.butter(4, [23e3, 53e3], btype='band', fs=fs, output='sos')
        dsb_sc = signal.sosfilt(sos_bpf, composite)
        
        # Synchronous demodulation of L-R
        diff_demod = dsb_sc * subcarrier * 2  # Factor of 2 for DSB-SC
        
        # Lowpass filter to get L-R
        diff_signal = signal.sosfilt(sos_lpf, diff_demod)
        
        # Recover L and R
        # L = (L+R)/2 + (L-R)/2 = L
        # R = (L+R)/2 - (L-R)/2 = R
        left = sum_signal + diff_signal
        right = sum_signal - diff_signal
        
        return left, right
    
    def receive(self, fm_signal, fs_fm):
        """
        Full receiver chain: FM demod → de-emphasis → stereo decode.
        
        Args:
            fm_signal: Complex FM modulated signal
            fs_fm: FM signal sampling frequency
            
        Returns:
            left, right: Recovered audio channels at fs_audio
        """
        # FM demodulate
        composite = self.fm_demodulate(fm_signal, fs_fm)
        
        # Apply de-emphasis
        b, a = self.create_deemphasis_filter(fs_fm)
        composite_deemph = signal.lfilter(b, a, composite)
        
        # Stereo decode
        left, right = self.stereo_decode(composite_deemph, fs_fm)
        
        # Resample to audio rate
        resample_ratio = self.fs_audio / fs_fm
        n_resampled = int(len(left) * resample_ratio)
        left_audio = signal.resample(left, n_resampled)
        right_audio = signal.resample(right, n_resampled)
        
        return left_audio, right_audio


def add_awgn(signal_in, snr_db):
    """
    Add Additive White Gaussian Noise to achieve specified SNR.
    
    Args:
        signal_in: Input signal (complex or real)
        snr_db: Desired SNR in dB
        
    Returns:
        Noisy signal
    """
    # Signal power
    sig_power = np.mean(np.abs(signal_in) ** 2)
    
    # Noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    
    # Generate noise
    if np.iscomplexobj(signal_in):
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal_in)) + 
                                             1j * np.random.randn(len(signal_in)))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(len(signal_in))
    
    return signal_in + noise


def measure_bandwidth_99(signal_in, fs):
    """
    Measure the bandwidth containing 99% of signal power.
    
    Args:
        signal_in: Input signal
        fs: Sampling frequency
        
    Returns:
        bandwidth: Bandwidth in Hz
        f: Frequency array
        psd: Power spectral density
    """
    # Compute PSD using Welch's method
    nperseg = min(len(signal_in), 8192)
    f, psd = signal.welch(signal_in, fs, nperseg=nperseg)
    
    # Total power
    total_power = np.sum(psd)
    
    # Find frequency range containing 99% power
    cumsum_power = np.cumsum(psd) / total_power
    
    # Find lower bound (0.5% from bottom)
    idx_low = np.argmax(cumsum_power >= 0.005)
    
    # Find upper bound (99.5% from bottom = 0.5% from top)
    idx_high = np.argmax(cumsum_power >= 0.995)
    
    bandwidth = f[idx_high] - f[idx_low]
    
    return bandwidth, f, psd


def calculate_snr(original, received):
    """
    Calculate SNR between original and received signals.
    
    Args:
        original: Original signal
        received: Received signal (should be same length)
        
    Returns:
        SNR in dB
    """
    # Ensure same length
    min_len = min(len(original), len(received))
    original = original[:min_len]
    received = received[:min_len]
    
    # Normalize signals
    original = original / (np.max(np.abs(original)) + 1e-10)
    received = received / (np.max(np.abs(received)) + 1e-10)
    
    # Cross-correlate to find best alignment
    correlation = np.correlate(received, original, 'full')
    lag = np.argmax(np.abs(correlation)) - len(original) + 1
    
    # Align signals
    if lag > 0:
        original = original[:-lag] if lag < len(original) else original
        received = received[lag:] if lag < len(received) else received
    elif lag < 0:
        received = received[:lag] if -lag < len(received) else received
        original = original[-lag:] if -lag < len(original) else original
    
    # Ensure same length after alignment
    min_len = min(len(original), len(received))
    original = original[:min_len]
    received = received[:min_len]
    
    # Calculate signal power and noise power
    signal_power = np.mean(original ** 2)
    noise = received - original
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return 100  # Very high SNR (essentially no noise)
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db


def load_audio(left_path, right_path):
    """
    Load left and right audio channels from WAV files.
    
    Args:
        left_path: Path to left channel WAV
        right_path: Path to right channel WAV
        
    Returns:
        left, right: Audio signals
        fs: Sampling frequency
    """
    fs_left, left_data = wavfile.read(left_path)
    fs_right, right_data = wavfile.read(right_path)
    
    # Convert to float
    if left_data.dtype == np.int16:
        left_data = left_data.astype(float) / 32768.0
    if right_data.dtype == np.int16:
        right_data = right_data.astype(float) / 32768.0
    
    # If stereo, take first channel
    if len(left_data.shape) > 1:
        left_data = left_data[:, 0]
    if len(right_data.shape) > 1:
        right_data = right_data[:, 0]
    
    # Use same length
    min_len = min(len(left_data), len(right_data))
    left = left_data[:min_len]
    right = right_data[:min_len]
    
    # Use left file's sample rate (both should be same)
    return left, right, fs_left


def carson_bandwidth(delta_f, f_m=53e3):
    """
    Calculate theoretical FM bandwidth using Carson's rule.
    
    B = 2 * (Δf + f_m)
    
    Args:
        delta_f: Maximum frequency deviation (Hz)
        f_m: Maximum modulating frequency (Hz) - 53 kHz for FM stereo
        
    Returns:
        Bandwidth in Hz
    """
    return 2 * (delta_f + f_m)


if __name__ == "__main__":
    # Test the system with default parameters
    print("FM Stereo System - Module Test")
    print("=" * 50)
    
    # Load audio
    left, right, fs = load_audio("audio/left.wav", "audio/right.wav")
    print(f"Loaded audio: {len(left)/fs:.2f} seconds at {fs} Hz")
    
    # Create transmitter and receiver
    tx = FMStereoTransmitter(fs_audio=fs, delta_f=75e3)
    rx = FMStereoReceiver(fs_audio=fs, delta_f=75e3)
    
    # Transmit
    fm_signal, composite, fs_composite = tx.transmit(left, right)
    print(f"FM signal generated: {len(fm_signal)} samples at {fs_composite/1e3:.0f} kHz")
    
    # Measure bandwidth
    bw, f, psd = measure_bandwidth_99(fm_signal, fs_composite)
    print(f"Measured 99% bandwidth: {bw/1e3:.1f} kHz")
    print(f"Carson's rule bandwidth: {carson_bandwidth(75e3)/1e3:.1f} kHz")
    
    # Receive (no noise)
    left_rx, right_rx = rx.receive(fm_signal, fs_composite)
    print(f"Recovered audio: {len(left_rx)/fs:.2f} seconds")
    
    # Calculate SNR
    snr_left = calculate_snr(left, left_rx)
    snr_right = calculate_snr(right, right_rx)
    print(f"Output SNR: Left = {snr_left:.1f} dB, Right = {snr_right:.1f} dB")
    
    print("\nModule test complete!")
