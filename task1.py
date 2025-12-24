import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from fm_stereo_system import FMTransmitter, FMReceiver
from common import add_awgn_complex, load_audio, calculate_snr, measure_99_bandwidth
import os;
# ==========================================
# 3. Task 1 Execution
# ==========================================

def run_task_1():
    print("--- Running Task 1: Frequency Deviation Effects ---")
    
    # 1. Setup
    filename = "audio/test1.wav"  # Ensure you have a WAV file here
    left_src, right_src, fs_audio = load_audio(filename)
    
    # Use 5 seconds of audio
    N = int(5.0 * fs_audio)
    left_src = left_src[:N]
    right_src = right_src[:N]
    
    # System Sample Rate (Must be high enough for 100kHz dev + 53kHz signal)
    # BW ~ 300kHz. Fs=800kHz is safe.
    fs_sim = 800000 
    
    # 2. Parameters
    deviations = [50000, 75000, 100000]
    fm_signal_bw = 53000 # Baseband bandwidth (Mono + Pilot + Stereo)
    input_snr_test = 25.0 # dB
    
    results_theo_bw = []
    results_meas_bw = []
    results_out_snr = []
    
    # 3. Main Loop
    mux = StereoMultiplexer(fs_sim)
    demux = StereoDemultiplexer()
    
    # Multiplex once (Baseband is same for all)
    composite, fs_mux = mux.multiplex(left_src, right_src, fs_audio)
    
    print(f"\n{'Delta F (kHz)':<15} | {'Theo BW (kHz)':<15} | {'Meas BW (kHz)':<15} | {'Out SNR (dB)':<15}")
    print("-" * 65)
    
    for delta_f in deviations:
        # A. Configure System
        tx = FMTransmitter(delta_f=delta_f)
        rx = FMReceiver(delta_f=delta_f)
        
        # B. Transmit (Clean)
        fm_clean, _ = tx.transmit(composite, fs_mux)
        
        # C. Measure BW
        bw_measured = measure_99_bandwidth(fm_clean, fs_mux)
        
        # D. Calculate Theoretical BW (Carson's Rule)
        # BW = 2 * (Delta_f + f_m)
        bw_theoretical = 2 * (delta_f + fm_signal_bw)
        
        # E. Measure SNR at 25dB Input
        # 1. Get Clean Reference Output (No Noise)
        rec_clean = rx.receive(fm_clean, fs_mux)
        l_ref_raw, _ = demux.demultiplex(rec_clean, fs_mux)
        
        # 2. Get Noisy Output (25dB Noise)
        fm_noisy = add_awgn_complex(fm_clean, input_snr_test)
        rec_noisy = rx.receive(fm_noisy, fs_mux)
        l_noisy_raw, _ = demux.demultiplex(rec_noisy, fs_mux)
        
        # 3. Resample and Align
        l_ref = signal.resample(l_ref_raw, N)
        l_noisy = signal.resample(l_noisy_raw, N)
        
        # Remove transients
        cut = int(0.1 * N)
        snr_out = calculate_snr(l_ref[cut:], l_noisy[cut:])
        
        # Store
        results_theo_bw.append(bw_theoretical / 1000) # kHz
        results_meas_bw.append(bw_measured / 1000)     # kHz
        results_out_snr.append(snr_out)
        
        print(f"{delta_f/1000:<15.0f} | {bw_theoretical/1000:<15.1f} | {bw_measured/1000:<15.1f} | {snr_out:<15.2f}")

    # ==========================================
    # 4. Plots
    # ==========================================
    
    dev_khz = [d/1000 for d in deviations]
    
    plt.figure(figsize=(10, 5))
    
    # Plot: Delta F vs SNR
    plt.plot(dev_khz, results_out_snr, 'b-o', linewidth=2)
    plt.title('Frequency Deviation vs Output SNR (Input SNR = 25dB)')
    plt.xlabel('Frequency Deviation Δf (kHz)')
    plt.ylabel('Output SNR (dB)')
    plt.grid(True)
    plt.xticks(dev_khz)
    
    plt.tight_layout()

    os.makedirs("outputs/task1", exist_ok=True)
    plt.savefig('outputs/task1/task1_deviation_vs_snr.png')
    plt.close()

    with open("outputs/task1/task1_results.txt", "w") as f:
        f.write("Delta F (kHz)   | Theo BW (kHz)   | Meas BW (kHz)   | Out SNR (dB)\n")
        f.write("-----------------------------------------------------------------\n")
        for i in range(len(deviations)):
            f.write(f"{dev_khz[i]:<15.0f} | {results_theo_bw[i]:<15.1f} | {results_meas_bw[i]:<15.1f} | {results_out_snr[i]:<15.2f}\n")



if __name__ == "__main__":
    run_task_1()