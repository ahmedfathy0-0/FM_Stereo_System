import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

class FMTransmitter:
    """FM Transmitter with pre-emphasis and FM modulation."""
    
    def __init__(self, fc=100e6, delta_f=75e3):
        """
        Initialize FM Transmitter.
        
        Args:
            fc: Carrier frequency (Hz) - 100 MHz typical FM broadcast (not used in baseband sim)
            delta_f: Maximum frequency deviation (Hz)
        """
        self.fc = fc
        self.delta_f = delta_f
        
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
        
        # Bilinear transform to digital
        b, a = signal.bilinear([tau, 1], [tau/10, 1], fs)
        return b, a
    
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
        if np.max(np.abs(message)) > 0:
            message_norm = message / (np.max(np.abs(message)))
        else:
            message_norm = message
        
        # Cumulative integral for phase
        # Phase deviation = 2*pi*delta_f * integral(m(t))dt
        dt = 1 / fs
        phase = 2 * np.pi * self.delta_f * np.cumsum(message_norm) * dt
        
        # Complex baseband FM signal
        fm_signal = np.exp(1j * phase)
        
        return fm_signal, message_norm
    
    def transmit(self, message, fs):
        """
        Transmitter chain: pre-emphasis → FM modulation.
        
        Args:
            message: Input signal (Composite Signal)
            fs: Sampling frequency
            
        Returns:
            fm_signal: Complex FM modulated signal
            message_preemph: Signal after pre-emphasis (for debugging)
        """
        # Apply pre-emphasis
        b, a = self.create_preemphasis_filter(fs)
        message_preemph = signal.lfilter(b, a, message)
        
        # FM modulate
        fm_signal, _ = self.fm_modulate(message_preemph, fs)
        
        return fm_signal, message_preemph


class FMReceiver:
    """FM Receiver with FM demodulation and de-emphasis."""
    
    def __init__(self, delta_f=75e3):
        """
        Initialize FM Receiver.
        
        Args:
            delta_f: Expected frequency deviation (Hz)
        """
        self.delta_f = delta_f
        
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
    
    def receive(self, fm_signal, fs):
        """
        Receiver chain: FM demod → de-emphasis.
        
        Args:
            fm_signal: Complex FM modulated signal
            fs: FM signal sampling frequency
            
        Returns:
            message: Recovered composite signal
        """
        # FM demodulate
        composite = self.fm_demodulate(fm_signal, fs)
        
        # Apply de-emphasis
        b, a = self.create_deemphasis_filter(fs)
        composite_deemph = signal.lfilter(b, a, composite)
        
        return composite_deemph