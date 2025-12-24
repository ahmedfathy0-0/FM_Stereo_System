"""
Visualization of Channel Separation for Different Filter Orders
Shows the LEAKAGE clearly in the Right channel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

from fm_stereo_system import FMTransmitter, FMReceiver
from stereo_multiplexer import StereoMultiplexer, StereoDemultiplexer
from common import add_awgn


def visualize_separation_for_all_orders():
    # Parameters
    fs_audio = 44100
    fs_composite = 200000
    delta_f = 75e3
    duration = 0.5
    f_tone = 1000
    snr_db = 40
    
    orders = [4, 8, 12]
    
    t_audio = np.arange(int(fs_audio * duration)) / fs_audio
    
    # ORIGINAL SIGNALS
    left_original = np.sin(2 * np.pi * f_tone * t_audio)
    right_original = np.zeros_like(t_audio)
    
    print("="*60)
    print("Processing FM Stereo System...")
    print("Left = 1 kHz tone, Right = SILENCE")
    print("="*60)
    
    results = {}
    
    for order in orders:
        print(f"\nFilter order = {order}...")
        
        mux = StereoMultiplexer(output_fs=fs_composite)
        composite, _ = mux.multiplex(left_original, right_original, fs_audio)
        
        tx = FMTransmitter(delta_f=delta_f)
        fm_signal, _ = tx.transmit(composite, fs_composite)
        fm_noisy = add_awgn(fm_signal, snr_db=snr_db)
        
        rx = FMReceiver(delta_f=delta_f)
        composite_rx = rx.receive(fm_noisy, fs_composite)
        
        demux = StereoDemultiplexer(pilot_bpf_order=order)
        left_rx, right_rx = demux.demultiplex(composite_rx, fs_composite)
        
        t_recovered = np.arange(len(left_rx)) / fs_composite
        
        cut = int(len(left_rx) * 0.1)
        rms_l = np.sqrt(np.mean(left_rx[cut:-cut]**2))
        rms_r = np.sqrt(np.mean(right_rx[cut:-cut]**2))
        if rms_r < 1e-9:
            rms_r = 1e-9
        separation_db = 20 * np.log10(rms_l / rms_r)
        
        results[order] = {
            'left_rx': left_rx,
            'right_rx': right_rx,
            't_recovered': t_recovered,
            'separation_db': separation_db
        }
        print(f"  Separation: {separation_db:.2f} dB")
    
    # ===========================================
    # CLEAR COMPARISON PLOT - BEFORE vs AFTER
    # ===========================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('LEAKAGE VISUALIZATION: Right Channel Should Be Silent But Has Signal!', 
                 fontsize=14, fontweight='bold', color='red')
    
    for i, order in enumerate(orders):
        res = results[order]
        
        # Get a portion of the signal (avoid transients)
        start_idx = int(0.1 * len(res['left_rx']))
        n_samples = int(0.01 * fs_composite)  # 10 ms of data
        
        t_plot = np.arange(n_samples) / fs_composite * 1000  # in ms
        left_plot = res['left_rx'][start_idx:start_idx+n_samples]
        right_plot = res['right_rx'][start_idx:start_idx+n_samples]
        
        # LEFT CHANNEL PLOT
        ax = axes[i, 0]
        ax.plot(t_plot, left_plot, 'b-', linewidth=1)
        ax.set_title(f'Order {order}: LEFT (Recovered Tone)', fontweight='bold', color='blue')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # RIGHT CHANNEL PLOT - THIS SHOWS THE LEAKAGE!
        ax = axes[i, 1]
        ax.plot(t_plot, right_plot, 'r-', linewidth=1)
        ax.set_title(f'Order {order}: RIGHT = LEAKAGE (Sep = {res["separation_db"]:.1f} dB)', 
                     fontweight='bold', color='red')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Add text box explaining the leakage
        rms_right = np.sqrt(np.mean(right_plot**2))
        ax.text(0.02, 0.95, f'RMS = {rms_right:.4f}\n(Should be ~0!)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Important: Use independent y-axis scaling for RIGHT to show the leakage clearly!
        # The right channel has smaller amplitude, so auto-scale will zoom in on it
    
    plt.tight_layout()
    plt.savefig('outputs/leakage_visualization.png', dpi=150)
    plt.show()
    
    # ===========================================
    # OVERLAY COMPARISON - Before vs After RIGHT channel
    # ===========================================
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
    fig2.suptitle('Before vs After: RIGHT Channel Comparison\n(Black = Original Silence, Red = Recovered with Leakage)', 
                  fontsize=13, fontweight='bold')
    
    # Original right channel (should be zero)
    t_orig = np.arange(1000) / fs_audio * 1000  # First 1000 samples
    right_orig_plot = right_original[:1000]
    
    for i, order in enumerate(orders):
        res = results[order]
        
        start_idx = int(0.1 * len(res['right_rx']))
        n_samples = 1000
        t_plot = np.arange(n_samples) / fs_composite * 1000
        right_plot = res['right_rx'][start_idx:start_idx+n_samples]
        
        ax = axes2[i]
        
        # Plot original (should be flat at zero)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Original RIGHT (Silence)', alpha=0.7)
        
        # Plot recovered (shows leakage)
        ax.plot(t_plot, right_plot, 'r-', linewidth=1, label='Recovered RIGHT (with leakage)')
        
        ax.set_title(f'Filter Order {order}: Separation = {res["separation_db"]:.2f} dB', fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Annotate the leakage
        ax.annotate('← This should be ZERO!\n    But there is signal (leakage)', 
                    xy=(t_plot[len(t_plot)//4], right_plot[len(right_plot)//4]),
                    xytext=(t_plot[len(t_plot)//2], np.max(np.abs(right_plot))*0.8),
                    arrowprops=dict(arrowstyle='->', color='darkred'),
                    fontsize=10, color='darkred', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/before_after_right_channel.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("Check the plots:")
    print("- The RIGHT channel SHOULD be zero (silence)")
    print("- But you can SEE a waveform = THIS IS THE LEAKAGE!")
    print("- The leakage amount varies with filter order")


if __name__ == "__main__":
    visualize_separation_for_all_orders()
