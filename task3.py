import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from fm_stereo_system import FMTransmitter, FMReceiver
import os

def run_task_3():
    print("--- Running Task 3: Channel Separation Analysis ---")
    
    # 1. Setup Parameters
    # We use a high FS to support the full FM bandwidth (~260kHz)
    # 600 kHz is safe for simulation.
    fs_sim = 600000  
    duration = 1.0   # 1 second duration
    t = np.arange(int(fs_sim * duration)) / fs_sim
    
    # 2. Generate Audio (L = 1kHz Tone, R = Silence)
    freq_audio = 1000
    left_audio = np.cos(2 * np.pi * freq_audio * t)
    right_audio = np.zeros_like(t) # Silence    
    
    print(f"Injecting 1kHz Tone into Left Channel. Right Channel is Silence.")
    
    # 3. Instantiate Components
    # We set multiplexer output to fs_sim to avoid extra resampling steps
    mux = StereoMultiplexer(output_fs=fs_sim)
    tx = FMTransmitter(delta_f=75e3)
    rx = FMReceiver(delta_f=75e3)
    demux = StereoDemultiplexer(pilot_bpf_order=4)
    
    # 4. Processing Chain
    print("Multiplexing...")
    composite, fs_mux = mux.multiplex(left_audio, right_audio, fs_sim)
    
    print("Transmitting (FM Mod)...")
    fm_sig, _ = tx.transmit(composite, fs_mux)
    
    # (Optional: Add noise here if desired, but Task 3 implies clean separation first)
    
    print("Receiving (FM Demod)...")
    rec_composite = rx.receive(fm_sig, fs_mux)
    
    print("Demultiplexing...")
    rec_l, rec_r = demux.demultiplex(rec_composite, fs_mux)
    
    # 5. Analysis
    # Discard first 10% and last 10% to avoid transient filter ringing
    start_idx = int(0.1 * len(rec_l))
    end_idx = int(0.9 * len(rec_l))
    
    l_cut = rec_l[start_idx:end_idx]
    r_cut = rec_r[start_idx:end_idx]
    
    rms_l = np.sqrt(np.mean(l_cut**2))
    rms_r = np.sqrt(np.mean(r_cut**2))
    
    print(f"\nRecovered Left RMS:  {rms_l:.4f}")
    print(f"Recovered Right RMS: {rms_r:.4f}")
    
    # Calculate Separation
    # Separation_dB = 20 * log10( |L_rec| / |R_rec| )
    if rms_r > 0:
        separation = 20 * np.log10(rms_l / rms_r)
    else:
        separation = float('inf')
        
    print(f"Measured Channel Separation: {separation:.2f} dB")
    
    # Interpretation Helper
    if separation < 0:
        print("\n[!] WARNING: Separation is negative.")
        print("    This means the Silent channel is louder than the Active channel.")
        print("    Likely cause: Channels are swapped due to Pilot Phase Inversion (Delay).")

    os.makedirs("outputs/task3", exist_ok=True) 
    with open("outputs/task3/results.txt", "w") as f:
        f.write(f"Measured Channel Separation: {separation:.2f} dB\n")

if __name__ == "__main__":
    run_task_3()