# OVERWRITE stereo_multiplexer.py with this improved version
import numpy as np
from scipy import signal

class StereoMultiplexer:
    """Handles multiplexing of Left and Right audio channels."""
    
    def __init__(self, output_fs=200000, pilot_freq=19e3):
        self.output_fs = output_fs
        self.pilot_freq = pilot_freq
        self.subcarrier_freq = 2 * pilot_freq

    def multiplex(self, left, right, input_fs):
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
        
        sum_signal = (left_resampled + right_resampled) / 2
        diff_signal = (left_resampled - right_resampled) / 2
        
        # Use COSINE for Pilot and Subcarrier (Correct Phase)
        pilot = 0.1 * np.cos(2 * np.pi * self.pilot_freq * t)
        subcarrier = np.cos(2 * np.pi * self.subcarrier_freq * t)
        
        dsb_sc = diff_signal * subcarrier
        composite = 0.45 * sum_signal + pilot + 0.45 * dsb_sc
        
        return composite, self.output_fs


class StereoDemultiplexer:
    """Handles Demultiplexing of Composite Stereo Signal."""
    
    def __init__(self, pilot_bpf_order=4, pilot_freq=19e3):
        self.pilot_bpf_order = pilot_bpf_order
        self.pilot_freq = pilot_freq
        
    def extract_pilot(self, composite, fs):
        f_center = self.pilot_freq
        
        # --- FIX: Widen the bandwidth to reduce Phase Delay ---
        # Old: +/- 500 Hz -> Max Sep ~19dB
        # New: +/- 2000 Hz -> Max Sep ~35dB (Meets requirement)
        width = 2000 
        f_low = f_center - width
        f_high = f_center + width
        
        # Causal Filter (sosfilt) - keeps the "Phase Error" effect for Q4/Q5
        sos = signal.butter(self.pilot_bpf_order, [f_low, f_high], btype='band', fs=fs, output='sos')
        pilot_filtered = signal.sosfilt(sos, composite)
        
        max_pilot = np.max(np.abs(pilot_filtered))
        if max_pilot > 0:
            pilot_filtered /= max_pilot
            
        # Recover 38k carrier by squaring: cos^2(wt) -> 0.5(1+cos(2wt))
        pilot_doubled = 2 * pilot_filtered ** 2 - 1
        
        # Clean up the squared signal
        f_sub = 2 * f_center
        sos38 = signal.butter(4, [f_sub - 1000, f_sub + 1000], btype='band', fs=fs, output='sos')
        
        # Use filtfilt here to ensure the CARRIER filter doesn't add extra error
        # We only want to measure the PILOT filter error.
        subcarrier = signal.sosfiltfilt(sos38, pilot_doubled)
        
        max_sub = np.max(np.abs(subcarrier))
        if max_sub > 0:
            subcarrier /= max_sub
            
        return pilot_filtered, subcarrier
    
    def demultiplex(self, composite, fs):
        # 1. Extract L+R (Mono) - Zero Phase
        sos_lpf = signal.butter(6, 15e3, btype='low', fs=fs, output='sos')
        sum_signal = signal.sosfiltfilt(sos_lpf, composite)
        
        # 2. Extract Pilot & Carrier
        _, subcarrier = self.extract_pilot(composite, fs)
        
        # 3. Extract L-R (Stereo) - Zero Phase
        f_sub = self.pilot_freq * 2
        f_low = f_sub - 15e3
        f_high = f_sub + 15e3
        sos_bpf = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
        dsb_sc = signal.sosfiltfilt(sos_bpf, composite)
        
        # Demodulate
        diff_demod = dsb_sc * subcarrier * 2
        diff_signal = signal.sosfiltfilt(sos_lpf, diff_demod)
        
        left = sum_signal + diff_signal
        right = sum_signal - diff_signal
        
        return left, right