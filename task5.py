import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from fm_stereo_system import FMTransmitter, FMReceiver
from common import add_awgn

OUTPUT_DIR = 'outputs/task5'
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_AUDIO, NOMINAL_PILOT, FS_MPX = 'audio/stereo.wav', 19000, 200000
OFFSETS = np.linspace(-500, 500, 21)

def compute_spectrum(data, fs):
    n = len(data)
    fft_data = np.fft.fft(data * np.hanning(n))
    mag_db = 20 * np.log10(np.abs(fft_data[:n//2]) + 1e-9)
    return np.fft.fftfreq(n, 1/fs)[:n//2], mag_db - np.max(mag_db)

if __name__ == "__main__":
    fs_audio, audio = wav.read(INPUT_AUDIO)
    audio = audio.astype(np.float64) / 32768.0
    left_src = audio[:, 0] if len(audio.shape) > 1 else audio
    right_src = np.zeros_like(left_src)
    
    res, bad_audio = [], None
    print("Running Robustness Test...")
    
    for offset in OFFSETS:
        mux = StereoMultiplexer(output_fs=FS_MPX, pilot_freq=NOMINAL_PILOT + offset)
        comp, fs_c = mux.multiplex(left_src, right_src, fs_audio)
        tx = FMTransmitter(delta_f=75000)
        rec = FMReceiver(delta_f=75000).receive(add_awgn(tx.transmit(comp, fs_c)[0], 60), fs_c)
        l_rec, r_rec = StereoDemultiplexer(pilot_bpf_order=4, pilot_freq=NOMINAL_PILOT).demultiplex(rec, fs_c)
        
        l_rec = signal.resample(l_rec, len(left_src))
        r_rec = signal.resample(r_rec, len(right_src))
        gain = np.max(np.abs(left_src)) / (np.max(np.abs(l_rec)) + 1e-9)
        l_rec *= gain; r_rec *= gain
        
        rms_l = np.sqrt(np.mean(l_rec[int(len(l_rec)*0.1):-int(len(l_rec)*0.1)]**2))
        rms_r = np.sqrt(np.mean(r_rec[int(len(r_rec)*0.1):-int(len(r_rec)*0.1)]**2))
        sep = 20 * np.log10(rms_l / rms_r) if rms_r > 1e-9 else 100.0
        res.append(sep)
        print(f"Offset {offset:>+4.0f} Hz: {sep:.2f} dB")
        if offset == 500: bad_audio = (l_rec, r_rec)

    pass_idxs = np.where(np.array(res) >= 20.0)[0]
    msg = f"Tolerance (>20dB): {OFFSETS[pass_idxs[0]]:.0f} to +{OFFSETS[pass_idxs[-1]]:.0f} Hz" if len(pass_idxs) else "Fail"
    print(f"\nRESULT: {msg}")

    plt.figure(figsize=(10, 6))
    plt.plot(OFFSETS, res, 'o-'); plt.axhline(20, c='r', ls='--')
    plt.title('Separation vs Pilot Frequency Error'); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'robustness_curve.png')); plt.close()

    if bad_audio:
        f_l, s_l = compute_spectrum(bad_audio[0], fs_audio)
        f_r, s_r = compute_spectrum(bad_audio[1], fs_audio)
        plt.figure(figsize=(12, 6))
        plt.plot(f_l, s_l, 'b', alpha=0.7); plt.plot(f_r, s_r, 'r', alpha=0.7)
        plt.title('Spectrum @ +500 Hz Error'); plt.xlim([0, 15000]); plt.ylim([-80, 5]); plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'spectrum_plus500Hz.png')); plt.close()

    with open(os.path.join(OUTPUT_DIR, 'task5_results.txt'), 'w') as f:
        f.write(f"{msg}\n\nOffset (Hz)     Separation (dB)\n" + "-"*35 + "\n")
        max_width1, max_width2 = len("Offset (Hz)"), len("Separation (dB)")
        for o, s in zip(OFFSETS, res): f.write(f"{o:<{15}.0f} {s:<{20}.2f}\n")