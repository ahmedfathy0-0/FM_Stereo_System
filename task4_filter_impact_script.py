import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from fm_stereo_system import FMTransmitter, FMReceiver

OUTPUT_DIR = os.path.join('outputs', 'task4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_AUDIO_PATH = 'audio/stereo.wav'
PILOT_FREQ = 19000
FS_MPX = 200000  
FS_TX = 250000      
ORDERS = [4, 8, 12] 

def plot_filter_responses(orders, fs, save_path):
    plt.figure(figsize=(10, 6))
    
    w_freqs = np.linspace(18000, 20000, 1000)
    
    for order in orders:
        f_center = PILOT_FREQ
        f_low = f_center - 500
        f_high = f_center + 500
        
        sos = signal.butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
        
        w, h = signal.sosfreqz(sos, worN=w_freqs, fs=fs)
        
        plt.plot(w, 20 * np.log10(abs(h)), linewidth=2, label=f'Order {order}')

    plt.title('Pilot Extraction Filter Frequency Responses (Q4b)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both', alpha=0.5)
    plt.legend()
    plt.xlim([18000, 20000])
    plt.ylim([-50, 5])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Filter Response Plot: {save_path}")

print("--- Loading Audio and Creating Test Signal ---")
fs_audio, audio_raw = wav.read(INPUT_AUDIO_PATH)
audio_norm = audio_raw.astype(np.float64) / 32768.0

if len(audio_norm.shape) > 1:
    left_src = audio_norm[:, 0]
else:
    left_src = audio_norm

right_src = np.zeros_like(left_src)
t_audio = np.arange(len(left_src)) / fs_audio

results_log = []
separations = []

for order in ORDERS:
    print(f"\nProcessing Filter Order: {order}...")
                
    mux = StereoMultiplexer(output_fs=FS_MPX, pilot_freq=PILOT_FREQ)
    composite, fs_comp = mux.multiplex(left_src, right_src, fs_audio)
    
    tx = FMTransmitter(delta_f=75000)
    fm_sig, _ = tx.transmit(composite, fs_comp)
        
    rx = FMReceiver(delta_f=75000)
    comp_rx = rx.receive(fm_sig, fs_comp)
    
    demux = StereoDemultiplexer(pilot_bpf_order=order, pilot_freq=PILOT_FREQ)
    l_rec, r_rec = demux.demultiplex(comp_rx, fs_comp)
    
    l_rec = signal.resample(l_rec, len(left_src))
    r_rec = signal.resample(r_rec, len(right_src))

    gain_corr = np.max(np.abs(left_src)) / (np.max(np.abs(l_rec)) + 1e-9)
    l_rec *= gain_corr
    r_rec *= gain_corr

    rms_l = np.sqrt(np.mean(l_rec**2))
    rms_r = np.sqrt(np.mean(r_rec**2))
    
    sep_db = 20 * np.log10(rms_l / rms_r) if rms_r > 1e-9 else 100.0
    separations.append(sep_db)
    
    log_entry = f"Order {order}: Left RMS={rms_l:.4f}, Right RMS={rms_r:.4f}, Separation={sep_db:.2f} dB"
    results_log.append(log_entry)
    print(f"  -> {log_entry}")
    
    l_int = np.int16(np.clip(l_rec, -1, 1) * 32767)
    r_int = np.int16(np.clip(r_rec, -1, 1) * 32767)
  
    n_plot = min(len(t_audio), int(6 * fs_audio))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_audio[:n_plot], l_rec[:n_plot], 'b', label='Recovered Left')
    plt.title(f'Recovered Audio - Order {order} (Left should be signal)')
    plt.grid(alpha=0.3)
    plt.ylabel('Amplitude')
    plt.ylim([-1, 1])
    
    plt.subplot(2, 1, 2)
    plt.plot(t_audio[:n_plot], r_rec[:n_plot], 'r', label='Recovered Right')
    plt.title(f'Recovered Audio - Order {order} (Right should be silence)')
    plt.grid(alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'waveform_order_{order}.png'))
    plt.close()

print("\n--- Generating Summary Plots ---")

plot_filter_responses(ORDERS, FS_MPX, os.path.join(OUTPUT_DIR, 'filter_responses.png'))

plt.figure(figsize=(8, 5))
plt.plot(ORDERS, separations, 'o-', linewidth=2, color='purple')
plt.title('Impact of Pilot Filter Order on Channel Separation')
plt.xlabel('Filter Order')
plt.ylabel('Channel Separation (dB)')
plt.grid(True)
plt.xticks(ORDERS)
for i, txt in enumerate(separations):
    plt.annotate(f"{txt:.1f} dB", (ORDERS[i], separations[i]), 
                 xytext=(0, 10), textcoords='offset points', ha='center')
plt.savefig(os.path.join(OUTPUT_DIR, 'separation_trend.png'))
plt.close()

txt_path = os.path.join(OUTPUT_DIR, 'task4_results.txt')
with open(txt_path, 'w') as f:
    f.write("=== Task 4: Filter Design Impact Analysis ===\n")
    f.write("Condition: Left Channel = Audio, Right Channel = Silence\n")
    f.write("Metric: Separation(dB) = 20 * log10(RMS_Left / RMS_Right)\n\n")
    f.write(f"{'Order':<10} {'Left RMS':<15} {'Right RMS':<15} {'Separation (dB)':<20}\n")
    f.write("-" * 60 + "\n")
    for line in results_log:
        # Parsing the log line back for formatting
        parts = line.split(',')
        order_val = parts[0].split(':')[0].split()[1]
        l_val = parts[0].split('=')[1]
        r_val = parts[1].split('=')[1]
        sep_val = parts[2].split('=')[1].replace(' dB', '')
        f.write(f"{order_val:<10} {l_val:<15} {r_val:<15} {sep_val:<20}\n")
    f.write("-" * 60 + "\n")
    f.write("\nObservation:\n")
    f.write("As filter order increases, the phase delay of the extracted pilot increases.\n")
    f.write("This desynchronizes the 38kHz carrier regeneration, causing leakage (crosstalk)\n")
    f.write("into the Right channel, thereby reducing separation.\n")

print(f"\nProcessing Complete.")
print(f"Results saved to: {txt_path}")
print(f"Images saved to: {OUTPUT_DIR}")