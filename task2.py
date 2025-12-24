import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

# Import your existing classes
from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from fm_stereo_system import FMTransmitter, FMReceiver
from common import add_awgn_complex, load_audio, calculate_snr

def measure_thd(signal_in, fs, freq_target=1000):
    """
    Calculates Total Harmonic Distortion (THD) from a signal.
    THD = sqrt(sum(Harmonics^2)) / Fundamental
    """
    # Power Spectral Density
    f, Pxx = signal.periodogram(signal_in, fs, window='hann')
    
    # 1. Measure Fundamental Power (Target Freq +/- 100Hz)
    bin_width = 100 
    idx_low = np.argmin(np.abs(f - (freq_target - bin_width)))
    idx_high = np.argmin(np.abs(f - (freq_target + bin_width)))
    power_fundamental = np.sum(Pxx[idx_low:idx_high])
    
    if power_fundamental <= 0:
        return 0.0

    # 2. Measure Harmonics Power (2nd to 5th harmonic)
    power_harmonics = 0
    for h in range(2, 6): # H2 to H5
        f_h = h * freq_target
        if f_h < fs/2:
            idx_h = np.argmin(np.abs(f - f_h))
            # Narrow window for harmonic
            idx_h_low = max(0, idx_h - 2)
            idx_h_high = min(len(f), idx_h + 3)
            power_harmonics += np.sum(Pxx[idx_h_low:idx_h_high])
            
    thd = np.sqrt(power_harmonics / power_fundamental) * 100 # In Percent
    return thd

def run_task2_audio():
    print("--- Task 2: Noise Immunity Analysis (L<->R, THD, SNR) ---")
    
    # ==========================================
    # 1. SETUP & AUDIO LOADING
    # ==========================================
    filename = "audio/stereo.wav"  # Ensure this path is correct
    left_src, right_src, fs_audio = load_audio(filename)

    silence = np.zeros_like(left_src)
    
    # Create 1kHz Tone for THD Test
    t_tone = np.arange(fs_audio) / fs_audio
    tone_src = 0.8 * np.cos(2 * np.pi * 1000 * t_tone)
    
    # ==========================================
    # 2. SYSTEM INITIALIZATION
    # ==========================================
    fs_sim = 600000
    mux = StereoMultiplexer(fs_sim)
    tx = FMTransmitter()
    rx = FMReceiver()
    demux = StereoDemultiplexer()
    
    # ==========================================
    # 3. PRE-CALCULATE CLEAN REFERENCES (Infinite SNR)
    # ==========================================
    print("Generating Clean References...")
    
    # Ref 1: Audio L->R (Reference for SNR calculation)
    comp_ref, fs_mux = mux.multiplex(left_src, silence, fs_audio)
    fm_ref_lr, _ = tx.transmit(comp_ref, fs_mux)
    rec_ref = rx.receive(fm_ref_lr, fs_mux)
    l_ref_raw, _ = demux.demultiplex(rec_ref, fs_mux)
    l_ref = signal.resample(l_ref_raw, fs_audio)
    
    # Ref 2: Audio R->L (Pre-calc FM signal only)
    comp_ref_rl, _ = mux.multiplex(silence, right_src, fs_audio)
    fm_ref_rl, _ = tx.transmit(comp_ref_rl, fs_mux)
    
    # Ref 3: Tone (Pre-calc FM signal only)
    comp_ref_tone, _ = mux.multiplex(tone_src, silence, fs_audio)
    fm_ref_tone, _ = tx.transmit(comp_ref_tone, fs_mux)

    # Cut transients for metrics
    cut = int(0.1 * fs_audio)
    l_ref = l_ref[cut:]
    
    # ==========================================
    # 4. MEASUREMENT LOOP
    # ==========================================
    snr_levels = [5, 10, 15, 20, 25]
    
    results = {
        "snr_in": snr_levels,
        "snr_out": [],
        "sep_lr": [],
        "sep_rl": [],
        "thd": []
    }
    
    print(f"\n{'In SNR':<8} | {'Out SNR':<8} | {'Sep L->R':<8} | {'Sep R->L':<8} | {'THD (%)':<8}")
    print("-" * 55)
    
    for snr in snr_levels:
        # --- PASS 1: Audio L->R (Measure Output SNR & Sep L->R) ---
        fm_noisy = add_awgn_complex(fm_ref_lr, snr)
        rec = rx.receive(fm_noisy, fs_mux)
        l_raw, r_raw = demux.demultiplex(rec, fs_mux)
        
        l_out = signal.resample(l_raw, fs_audio)[cut:]
        r_out = signal.resample(r_raw, fs_audio)[cut:]
        
        # i. Measure Output SNR (Time Domain: Signal - Reference)
        val_snr_out = calculate_snr(l_ref, l_out)
        
        # ii. Measure Separation L->R (Active L / Crosstalk R)
        rms_l = np.sqrt(np.mean(l_out**2))
        rms_r = np.sqrt(np.mean(r_out**2))
        val_sep_lr = 20 * np.log10(rms_l / rms_r) if rms_r > 1e-9 else 100

        # --- PASS 2: Audio R->L (Measure Sep R->L) ---
        fm_noisy_rl = add_awgn_complex(fm_ref_rl, snr)
        rec_rl = rx.receive(fm_noisy_rl, fs_mux)
        l_raw_rl, r_raw_rl = demux.demultiplex(rec_rl, fs_mux)
        
        l_out_rl = signal.resample(l_raw_rl, fs_audio)[cut:]
        r_out_rl = signal.resample(r_raw_rl, fs_audio)[cut:]
        
        # ii. Measure Separation R->L (Active R / Crosstalk L)
        rms_r_active = np.sqrt(np.mean(r_out_rl**2))
        rms_l_crosstalk = np.sqrt(np.mean(l_out_rl**2))
        val_sep_rl = 20 * np.log10(rms_r_active / rms_l_crosstalk) if rms_l_crosstalk > 1e-9 else 100
        
        # --- PASS 3: Tone Test (Measure THD) ---
        fm_noisy_tone = add_awgn_complex(fm_ref_tone, snr)
        rec_tone = rx.receive(fm_noisy_tone, fs_mux)
        l_raw_tone, _ = demux.demultiplex(rec_tone, fs_mux)
        l_out_tone = signal.resample(l_raw_tone, fs_audio)[cut:]
        
        # iii. Measure THD
        val_thd = measure_thd(l_out_tone, fs_audio, freq_target=1000)
        
        # Store Data
        results["snr_out"].append(val_snr_out)
        results["sep_lr"].append(val_sep_lr)
        results["sep_rl"].append(val_sep_rl)
        results["thd"].append(val_thd)
        
        print(f"{snr:<8} | {val_snr_out:<8.2f} | {val_sep_lr:<8.2f} | {val_sep_rl:<8.2f} | {val_thd:<8.2f}")

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    plt.figure(figsize=(15, 5))
    
    # Plot A: SNR
    plt.subplot(1, 3, 1)
    plt.plot(results["snr_in"], results["snr_out"], 'b-o', linewidth=2)
    plt.title('a) Output SNR vs Input SNR')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('Output SNR (dB)')
    plt.grid(True)
    
    # Plot B: Separation
    plt.subplot(1, 3, 2)
    plt.plot(results["snr_in"], results["sep_lr"], 'g-o', label='L->R', linewidth=2)
    plt.plot(results["snr_in"], results["sep_rl"], 'r--s', label='R->L', linewidth=2)
    plt.title('b) Channel Separation vs Input SNR')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('Separation (dB)')
    plt.legend()
    plt.grid(True)
    
    # Plot C: THD
    plt.subplot(1, 3, 3)
    plt.plot(results["snr_in"], results["thd"], 'k-x', linewidth=2)
    plt.title('c) THD vs Input SNR')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('THD (%)')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("outputs/task2", exist_ok=True)
    plt.savefig('outputs/task2/graphs_results.png')
    plt.show()

    os.makedirs("outputs/task2", exist_ok=True)

    with open("outputs/task2/results.txt", "w") as f:
        f.write("Task 2: Noise Immunity Analysis\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"{'In SNR':<8} | {'Out SNR':<8} | {'Sep L->R':<10} | {'Sep R->L':<10} | {'THD (%)':<8}\n")
        f.write("-" * 65 + "\n")

        for i in range(len(results["snr_in"])):
            f.write(
                f"{results['snr_in'][i]:<8} | "
                f"{results['snr_out'][i]:<8.2f} | "
                f"{results['sep_lr'][i]:<10.2f} | "
                f"{results['sep_rl'][i]:<10.2f} | "
                f"{results['thd'][i]:<8.2f}\n"
            )

if __name__ == "__main__":
    run_task2_audio()