import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

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
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal_in.shape) + 
                                             1j * np.random.randn(*signal_in.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*signal_in.shape)
    
    return signal_in + noise

def add_awgn_complex(signal_in, snr_db):
    """Adds Complex Gaussian Noise to a signal for a given SNR."""
    # Signal Power (Assume Amplitude 1 -> Power 1 for complex exp)
    P_sig = 1.0 
    
    # SNR_dB = 10 * log10(P_sig / P_noise)
    # P_noise = P_sig / 10^(SNR/10)
    P_noise = P_sig / (10 ** (snr_db / 10))
    
    # Generate Noise (Complex)
    # Variance per dimension is P_noise / 2
    noise_std = np.sqrt(P_noise / 2)
    noise = noise_std * (np.random.randn(*signal_in.shape) + 1j * np.random.randn(*signal_in.shape))
    
    return (signal_in + noise)

def measure_99_bandwidth(signal_in, fs):
    """Calculates the 99% Power Bandwidth of a complex signal."""
    # Use Welch's method for smooth PSD
    f, Pxx = signal.welch(signal_in, fs, nperseg=2048, return_onesided=False)
    
    # Shift to center 0 Hz
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    total_power = np.sum(Pxx)
    target_power = 0.99 * total_power
    
    # Cumulative sum from center outwards is hard, 
    # simpler to do cumulative sum from left edge
    cum_power = np.cumsum(Pxx)
    
    # Find indices where cum power crosses 0.5% and 99.5%
    # (Leaving 0.5% on each tail = 1% total excluded)
    lower_bound_power = 0.005 * total_power
    upper_bound_power = 0.995 * total_power
    
    idx_min = np.searchsorted(cum_power, lower_bound_power)
    idx_max = np.searchsorted(cum_power, upper_bound_power)
    
    bw = f[idx_max] - f[idx_min]
    return bw


def calculate_snr(clean_ref, noisy_sig):
    """
    Calculates SNR by comparing Noisy Output against Clean Output.
    SNR = Power(Clean) / Power(Noisy - Clean)
    """
    # Ensure lengths match
    min_len = min(len(clean_ref), len(noisy_sig))
    clean = clean_ref[:min_len]
    noisy = noisy_sig[:min_len]
    
    # Error signal (Noise)
    noise_component = noisy - clean
    
    p_signal = np.mean(clean**2)
    p_noise = np.mean(noise_component**2)
    
    if p_noise == 0: return 100.0
    return 10 * np.log10(p_signal / p_noise)

def load_audio(path1):
    # Mode 1: Single file (Stereo or Mono)
    fs, data = wavfile.read(path1)
    
    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(float) / 32768.0
        
    if len(data.shape) == 1:
        # Mono file - duplicate to both channels
        left = data
        right = data
    else:
        # Stereo file
        left = data[:, 0]
        right = data[:, 1]
        
    return left, right, fs

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


def measure_thd(signal_in, fs, f_fund=1000):
    """
    Measure Total Harmonic Distortion (THD) of a signal.
    THD = sqrt(sum(V_harmonics^2)) / V_fundamental
    
    Args:
        signal_in: Input signal
        fs: Sampling frequency
        f_fund: Fundamental frequency (Hz)
        
    Returns:
        THD in percent (%)
    """
    # Remove DC
    signal_in = signal_in - np.mean(signal_in)
    
    # Windowing to reduce leakage
    window = np.blackman(len(signal_in))
    y = signal_in * window
    
    # FFT
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/fs)
    
    # Find fundamental peak index (search near f_fund)
    idx_fund_approx = np.argmin(np.abs(freqs - f_fund))
    search_range = 10  # Search +/- 10 bins
    start = max(0, idx_fund_approx - search_range)
    end = min(len(Y), idx_fund_approx + search_range)
    
    if end <= start:
        return 0.0
        
    idx_fund = start + np.argmax(np.abs(Y[start:end]))
    f_actual = freqs[idx_fund]
    
    power_fund = np.abs(Y[idx_fund])**2
    
    # Sum power of first 10 harmonics
    power_harmonics = 0
    harmonics_count = 0
    
    for h in range(2, 11): # 2nd to 10th harmonic
        f_harm = h * f_actual
        
        # Stop if above Nyquist
        if f_harm >= fs/2:
            break
            
        # Find peak for this harmonic
        idx_harm = np.argmin(np.abs(freqs - f_harm))
        
        # Look around expected index
        h_start = max(0, idx_harm - 5)
        h_end = min(len(Y), idx_harm + 5)
        
        if h_end > h_start:
            # Add peak power of harmonic
            power_harmonics += np.max(np.abs(Y[h_start:h_end]))**2
            harmonics_count += 1
            
    if power_fund == 0:
        return 0.0
        
    thd = np.sqrt(power_harmonics / power_fund) * 100
    return thd
