import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from fm_stereo_system import FMTransmitter, FMReceiver

OUTPUT_DIR, INPUT_AUDIO = 'outputs/task4', 'audio/stereo.wav'
PILOT_FREQ, FS_MPX = 19000, 200000
ORDERS = [4, 8, 12]
os.makedirs(OUTPUT_DIR, exist_ok=True)

fs, audio = wav.read(INPUT_AUDIO)
audio = audio.astype(np.float64) / 32768.0
left_src = audio[:, 0] if len(audio.shape) > 1 else audio
right_src = audio[:, 1] if len(audio.shape) > 1 else audio
t = np.arange(len(left_src)) / fs

results, sep_lr, sep_rl = [], [], []

print(f"Processing Filter Orders: {ORDERS}...")

plt.figure(figsize=(10, 6))
w_freqs = np.linspace(18000, 20000, 1000)
for order in ORDERS:
    sos = signal.butter(order, [PILOT_FREQ-500, PILOT_FREQ+500], btype='band', fs=FS_MPX, output='sos')
    w, h = signal.sosfreqz(sos, worN=w_freqs, fs=FS_MPX)
    plt.plot(w, 20 * np.log10(abs(h)), linewidth=2, label=f'Order {order}')
plt.title('Pilot Filter Frequency Responses'); plt.xlabel('Frequency (Hz)'); plt.ylabel('dB')
plt.grid(True); plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, 'filter_responses.png')); plt.close()

for order in ORDERS:
    for mode, src, leak in [('L_to_R', 'left', 'right'), ('R_to_L', 'right', 'left')]:
        l_in = left_src if src == 'left' else np.zeros_like(left_src)
        r_in = right_src if src == 'right' else np.zeros_like(right_src) 

        mux = StereoMultiplexer(output_fs=FS_MPX, pilot_freq=PILOT_FREQ)
        comp, fs_c = mux.multiplex(l_in, r_in, fs)
        tx = FMTransmitter(delta_f=75000)
        fm, _ = tx.transmit(comp, fs_c)
        rx = FMReceiver(delta_f=75000)
        rec = rx.receive(fm, fs_c)
        demux = StereoDemultiplexer(pilot_bpf_order=order, pilot_freq=PILOT_FREQ)
        l_out, r_out = demux.demultiplex(rec, fs_c)

        l_out = signal.resample(l_out, len(left_src))
        r_out = signal.resample(r_out, len(left_src))
        
        src_sig_max = np.max(np.abs(left_src)) # Original source max
        rec_sig = l_out if src == 'left' else r_out
        rec_leak = r_out if src == 'left' else l_out
        
        gain_corr = src_sig_max / (np.max(np.abs(rec_sig)) + 1e-9)
        l_out *= gain_corr
        r_out *= gain_corr

        # Metrics
        rms_sig = np.sqrt(np.mean(rec_sig**2))
        rms_leak = np.sqrt(np.mean(rec_leak**2))
        
        sep = 20 * np.log10(rms_sig / rms_leak) if rms_leak > 1e-9 else 100.0
        
        if mode == 'L_to_R': sep_lr.append(sep)
        else: sep_rl.append(sep)
        
        results.append(f"{order:<8} {mode:<10} {rms_sig:<12.4f} {rms_leak:<12.4f} {sep:<10.2f}")
        print(f"Order {order} {mode}: {sep:.2f} dB")

        n = min(len(t), int(6 * fs)) # printing the first six seconds 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(t[:n], rec_sig[:n], 'b'); ax1.set_title(f'Order {order} {mode}: Signal ({src.upper()})'); ax1.set_ylim([-1, 1]); ax1.grid(alpha=0.3)
        ax2.plot(t[:n], rec_leak[:n], 'r'); ax2.set_title(f'Order {order} {mode}: Leakage ({leak.upper()})'); ax2.set_ylim([-1, 1]); ax2.grid(alpha=0.3)
        plt.xlabel('Time (s)'); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f'waveform_order_{order}_{mode}.png')); plt.close()

with open(os.path.join(OUTPUT_DIR, 'task4_results.txt'), 'w') as f:
    f.write(f"{'Order':<8} {'Mode':<10} {'Sig RMS':<12} {'Leak RMS':<12} {'Sep (dB)':<10}\n" + "-"*55 + "\n" + "\n".join(results))

plt.figure(figsize=(8, 5))
plt.plot(ORDERS, sep_lr, 'o-', label='L->R'); plt.plot(ORDERS, sep_rl, 's--', label='R->L')
plt.title('Separation vs Filter Order'); plt.xlabel('Order'); plt.ylabel('dB'); plt.grid(True); plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'separation_trend.png')); plt.close()

print(f"Done. Outputs in {OUTPUT_DIR}")