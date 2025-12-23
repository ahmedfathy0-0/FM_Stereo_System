import numpy as np
from scipy import signal

class StereoMultiplexer:
    """Handles multiplexing of Left and Right audio channels into a Composite Stereo Signal."""
    
    def __init__(self, output_fs=200000, pilot_freq=19e3):
        """
        Initialize Stereo Multiplexer.
        
        Args:
            output_fs: Sampling frequency for the composite signal. 
                       Must be high enough to support 53kHz bandwidth (>106kHz).
                       Default 200kHz.
            pilot_freq: Frequency of the pilot tone (default 19 kHz).
        """
        self.output_fs = output_fs
        # FM Stereo standard frequencies
        self.pilot_freq = pilot_freq
        self.subcarrier_freq = 2 * pilot_freq

    def multiplex(self, left, right, input_fs):
        """
        Create FM stereo composite signal from Left and Right channels.
        
        Composite signal structure:
        - L+R sum (0-15 kHz)
        - 19 kHz pilot tone
        - (L-R) DSB-SC on 38 kHz subcarrier (23-53 kHz)
        
        Args:
            left: Left channel audio
            right: Right channel audio
            input_fs: Sampling frequency of input audio
            
        Returns:
            composite: Composite stereo signal
            fs: Output sampling frequency (self.output_fs)
        """
        # Resample audio to composite rate
        if input_fs != self.output_fs:
            resample_ratio = self.output_fs / input_fs
            n_resampled = int(len(left) * resample_ratio)
            left_resampled = signal.resample(left, n_resampled)
            right_resampled = signal.resample(right, n_resampled)
        else:
            left_resampled = left
            right_resampled = right
            
        n = len(left_resampled)
        t = np.arange(n) / self.output_fs
        
        # L+R and L-R signals
        sum_signal = (left_resampled + right_resampled) / 2
        diff_signal = (left_resampled - right_resampled) / 2
        
        # Pilot tone (typically 10% injection level)
        pilot = 0.1 * np.sin(2 * np.pi * self.pilot_freq * t)
        
        # L-R on subcarrier (DSB-SC modulation)
        # Use simple sine for subcarrier, assuming phase alignment with pilot 
        # (pilot = sin(wt), subcarrier = sin(2wt))
        # Note: In real systems, pilot phase relationship is critical.
        subcarrier = np.sin(2 * np.pi * self.subcarrier_freq * t)
        dsb_sc = diff_signal * subcarrier
        
        # Composite signal
        # Standard levels: L+R = 90%, pilot = 10%, L-R = 90%
        composite = 0.45 * sum_signal + pilot + 0.45 * dsb_sc
        
        return composite, self.output_fs


class StereoDemultiplexer:
    """Handles Demultiplexing of Composite Stereo Signal into Left and Right channels."""
    
    def __init__(self, pilot_bpf_order=4, pilot_freq=19e3):
        """
        Initialize Stereo Demultiplexer.
        
        Args:
            pilot_bpf_order: Order of the bandpass filter used for pilot extraction.
                             Default is 4.
            pilot_freq: Nominal pilot frequency (default 19 kHz).
        """
        self.pilot_bpf_order = pilot_bpf_order
        self.pilot_freq = pilot_freq
        # Frequencies are derived from pilot, but we extract pilot dynamically
        
    def extract_pilot(self, composite, fs):
        """
        Extract and phase-lock to the 19 kHz pilot tone.
        
        Returns:
            Pilot signal for synchronous detection
            Double-frequency signal for L-R demodulation
        """
        # Bandpass filter around nominal pilot freq
        # Bandwidth +/- 500 Hz
        f_center = self.pilot_freq
        f_low = f_center - 500
        f_high = f_center + 500
        
        sos = signal.butter(self.pilot_bpf_order, [f_low, f_high], btype='band', fs=fs, output='sos')
        pilot_filtered = signal.sosfilt(sos, composite)
        
        # Normalize pilot using a robust max (avoid division by zero)
        max_pilot = np.max(np.abs(pilot_filtered))
        if max_pilot > 0:
            pilot_filtered = pilot_filtered / max_pilot
        
        # Generate doubled frequency (38 kHz) for L-R demodulation
        # Simple approach: square the pilot and filter
        # cos(2θ) = 2cos²(θ) - 1
        # If pilot is sin(wt), pilot^2 = sin^2(wt) = (1 - cos(2wt))/2
        # AC coupling removes DC, leaving -cos(2wt) which is fine up to phase sign
        pilot_doubled = 2 * pilot_filtered ** 2 - 1
        
        # Bandpass around 2 * pilot freq
        f_sub = 2 * f_center
        f_low_sub = f_sub - 1000
        f_high_sub = f_sub + 1000
        sos38 = signal.butter(self.pilot_bpf_order, [f_low_sub, f_high_sub], btype='band', fs=fs, output='sos')
        subcarrier = signal.sosfilt(sos38, pilot_doubled)
        
        max_sub = np.max(np.abs(subcarrier))
        if max_sub > 0:
            subcarrier = subcarrier / max_sub
        
        return pilot_filtered, subcarrier
    
    def demultiplex(self, composite, fs):
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
        
        # Extract L-R DSB-SC signal (bandpass around 38 kHz, width 30k)
        # 38 +/- 15 = 23 to 53
        # Using self.pilot_freq * 2 as center
        f_sub = self.pilot_freq * 2
        f_low = f_sub - 15e3
        f_high = f_sub + 15e3
        
        sos_bpf = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
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


