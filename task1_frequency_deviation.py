"""
Task 1: Frequency Deviation Effects Analysis

This script analyzes how different frequency deviations (50, 75, 100 kHz)
affect FM stereo system performance.

Measurements:
1. FM signal bandwidth (99% power)
2. Theoretical bandwidth (Carson's rule)
3. Output SNR with input SNR = 25 dB

Outputs:
- Bandwidth comparison table
- SNR vs. frequency deviation plot
- Analysis and recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

from fm_stereo_system import (
    FMStereoTransmitter, FMStereoReceiver,
    load_audio, add_awgn, measure_bandwidth_99,
    calculate_snr, carson_bandwidth
)


def run_frequency_deviation_analysis(left, right, fs, delta_f, input_snr_db=25):
    """
    Run complete FM stereo analysis for a specific frequency deviation.
    
    Args:
        left, right: Audio channels
        fs: Audio sampling frequency
        delta_f: Frequency deviation (Hz)
        input_snr_db: Input SNR in dB
        
    Returns:
        Dictionary with all measurement results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Δf = {delta_f/1e3:.0f} kHz")
    print('='*60)
    
    # Create transmitter and receiver with specified deviation
    tx = FMStereoTransmitter(fs_audio=fs, delta_f=delta_f)
    rx = FMStereoReceiver(fs_audio=fs, delta_f=delta_f)
    
    # Transmit
    fm_signal, composite, fs_composite = tx.transmit(left, right)
    print(f"FM signal: {len(fm_signal)} samples at {fs_composite/1e3:.0f} kHz")
    
    # Measure bandwidth (99% power)
    measured_bw, freq, psd = measure_bandwidth_99(fm_signal, fs_composite)
    print(f"Measured bandwidth (99% power): {measured_bw/1e3:.1f} kHz")
    
    # Calculate theoretical bandwidth (Carson's rule)
    theoretical_bw = carson_bandwidth(delta_f, f_m=53e3)
    print(f"Theoretical bandwidth (Carson): {theoretical_bw/1e3:.1f} kHz")
    
    # Add noise to achieve input SNR
    fm_noisy = add_awgn(fm_signal, input_snr_db)
    
    # Verify input SNR
    noise = fm_noisy - fm_signal
    actual_input_snr = 10 * np.log10(np.mean(np.abs(fm_signal)**2) / np.mean(np.abs(noise)**2))
    print(f"Actual input SNR: {actual_input_snr:.1f} dB")
    
    # Receive noisy signal
    left_rx, right_rx = rx.receive(fm_noisy, fs_composite)
    
    # Calculate output SNR
    output_snr_left = calculate_snr(left, left_rx)
    output_snr_right = calculate_snr(right, right_rx)
    output_snr_avg = (output_snr_left + output_snr_right) / 2
    
    print(f"Output SNR: Left = {output_snr_left:.1f} dB, Right = {output_snr_right:.1f} dB")
    print(f"Average output SNR: {output_snr_avg:.1f} dB")
    
    # Also measure clean output SNR (no noise, for reference)
    left_rx_clean, right_rx_clean = rx.receive(fm_signal, fs_composite)
    clean_snr_left = calculate_snr(left, left_rx_clean)
    clean_snr_right = calculate_snr(right, right_rx_clean)
    clean_snr_avg = (clean_snr_left + clean_snr_right) / 2
    print(f"Clean output SNR (no noise): {clean_snr_avg:.1f} dB")
    
    # Calculate modulation index (beta)
    f_m = 53e3  # Maximum modulating frequency
    beta = delta_f / f_m
    print(f"Modulation index (β): {beta:.2f}")
    
    return {
        'delta_f': delta_f,
        'measured_bw': measured_bw,
        'theoretical_bw': theoretical_bw,
        'input_snr': actual_input_snr,
        'output_snr_left': output_snr_left,
        'output_snr_right': output_snr_right,
        'output_snr_avg': output_snr_avg,
        'clean_snr_avg': clean_snr_avg,
        'beta': beta,
        'freq': freq,
        'psd': psd
    }


def create_bandwidth_table(results):
    """
    Create comparison table of theoretical vs measured bandwidth.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Table as string
    """
    table = []
    table.append("=" * 80)
    table.append("TASK 1: BANDWIDTH COMPARISON TABLE - Theoretical vs Measured")
    table.append("=" * 80)
    table.append("")
    table.append(f"{'Δf (kHz)':<12} {'β':<8} {'Theoretical BW':<18} {'Measured BW':<18} {'Difference':<12}")
    table.append(f"{'':12} {'':8} {'(Carson Rule)':<18} {'(99% power)':<18} {'(%)':<12}")
    table.append("-" * 80)
    
    for r in results:
        delta_f_khz = r['delta_f'] / 1e3
        theo_bw_khz = r['theoretical_bw'] / 1e3
        meas_bw_khz = r['measured_bw'] / 1e3
        diff_pct = ((meas_bw_khz - theo_bw_khz) / theo_bw_khz) * 100
        
        table.append(f"{delta_f_khz:<12.0f} {r['beta']:<8.2f} {theo_bw_khz:<18.1f} {meas_bw_khz:<18.1f} {diff_pct:<12.1f}")
    
    table.append("-" * 80)
    table.append("")
    table.append("Notes:")
    table.append("- Carson's Rule: B = 2(Δf + fm), where fm = 53 kHz (max composite frequency)")
    table.append("- Measured bandwidth contains 99% of total signal power")
    table.append("- β = Δf/fm is the modulation index")
    table.append("")
    
    return "\n".join(table)


def create_snr_table(results):
    """
    Create SNR measurement table.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Table as string
    """
    table = []
    table.append("=" * 80)
    table.append("TASK 1: SNR MEASUREMENTS (Input SNR = 25 dB)")
    table.append("=" * 80)
    table.append("")
    table.append(f"{'Δf (kHz)':<12} {'Output SNR Left':<18} {'Output SNR Right':<18} {'Average':<12}")
    table.append(f"{'':12} {'(dB)':<18} {'(dB)':<18} {'(dB)':<12}")
    table.append("-" * 80)
    
    for r in results:
        delta_f_khz = r['delta_f'] / 1e3
        table.append(f"{delta_f_khz:<12.0f} {r['output_snr_left']:<18.1f} {r['output_snr_right']:<18.1f} {r['output_snr_avg']:<12.1f}")
    
    table.append("-" * 80)
    table.append("")
    
    return "\n".join(table)


def plot_snr_vs_deviation(results, output_path):
    """
    Plot output SNR vs frequency deviation.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    delta_f_values = [r['delta_f'] / 1e3 for r in results]
    snr_values = [r['output_snr_avg'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # Main plot
    plt.plot(delta_f_values, snr_values, 'bo-', linewidth=2, markersize=10, label='Average Output SNR')
    
    # Add individual L/R points
    snr_left = [r['output_snr_left'] for r in results]
    snr_right = [r['output_snr_right'] for r in results]
    plt.plot(delta_f_values, snr_left, 'g^--', linewidth=1, markersize=8, alpha=0.7, label='Left Channel')
    plt.plot(delta_f_values, snr_right, 'rs--', linewidth=1, markersize=8, alpha=0.7, label='Right Channel')
    
    # Input SNR reference line
    plt.axhline(y=25, color='r', linestyle=':', linewidth=1.5, label='Input SNR (25 dB)')
    
    plt.xlabel('Frequency Deviation Δf (kHz)', fontsize=12)
    plt.ylabel('Output SNR (dB)', fontsize=12)
    plt.title('FM Stereo System: SNR vs Frequency Deviation\n(Input SNR = 25 dB)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(delta_f_values, snr_values)):
        plt.annotate(f'{y:.1f} dB', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SNR plot to: {output_path}")


def plot_bandwidth_comparison(results, output_path):
    """
    Plot bandwidth comparison (theoretical vs measured).
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    delta_f_values = [r['delta_f'] / 1e3 for r in results]
    theo_bw = [r['theoretical_bw'] / 1e3 for r in results]
    meas_bw = [r['measured_bw'] / 1e3 for r in results]
    
    x = np.arange(len(delta_f_values))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, theo_bw, width, label='Theoretical (Carson)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, meas_bw, width, label='Measured (99% power)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Frequency Deviation Δf (kHz)', fontsize=12)
    ax.set_ylabel('Bandwidth (kHz)', fontsize=12)
    ax.set_title('FM Signal Bandwidth: Theoretical vs Measured', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.0f}' for d in delta_f_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved bandwidth comparison plot to: {output_path}")


def write_analysis(results, output_path):
    """
    Write complete analysis including trade-off discussion.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save analysis
    """
    with open(output_path, 'w') as f:
        # Write bandwidth table
        f.write(create_bandwidth_table(results))
        f.write("\n")
        
        # Write SNR table
        f.write(create_snr_table(results))
        f.write("\n")
        
        # Write analysis
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS: BANDWIDTH vs SNR TRADE-OFF\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OBSERVATIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Analyze bandwidth trend
        f.write("\n1. Bandwidth Analysis:\n")
        for r in results:
            delta_f_khz = r['delta_f'] / 1e3
            theo_bw_khz = r['theoretical_bw'] / 1e3
            meas_bw_khz = r['measured_bw'] / 1e3
            f.write(f"   - Δf = {delta_f_khz:.0f} kHz: ")
            f.write(f"Theoretical = {theo_bw_khz:.0f} kHz, ")
            f.write(f"Measured = {meas_bw_khz:.0f} kHz\n")
        
        # Analyze SNR trend
        f.write("\n2. SNR Analysis:\n")
        for r in results:
            delta_f_khz = r['delta_f'] / 1e3
            f.write(f"   - Δf = {delta_f_khz:.0f} kHz: ")
            f.write(f"Output SNR = {r['output_snr_avg']:.1f} dB\n")
        
        f.write("\n3. Trade-off Discussion:\n")
        f.write("   The fundamental trade-off in FM broadcasting is between:\n")
        f.write("   - BANDWIDTH: Higher Δf requires more spectrum\n")
        f.write("   - SNR IMPROVEMENT: FM provides SNR gain proportional to β²\n\n")
        
        f.write("   From our measurements:\n")
        
        # Find best SNR
        best_snr_idx = np.argmax([r['output_snr_avg'] for r in results])
        best_result = results[best_snr_idx]
        
        f.write(f"   - Highest output SNR: {best_result['output_snr_avg']:.1f} dB ")
        f.write(f"at Δf = {best_result['delta_f']/1e3:.0f} kHz\n")
        f.write(f"   - This uses bandwidth of {best_result['measured_bw']/1e3:.0f} kHz\n\n")
        
        f.write("4. FM Improvement Factor:\n")
        f.write("   FM provides SNR improvement over baseband by factor of 3β²\n")
        f.write("   where β = Δf/fm (modulation index)\n\n")
        for r in results:
            improvement = 10 * np.log10(3 * r['beta']**2)
            f.write(f"   - β = {r['beta']:.2f}: Theoretical improvement = {improvement:.1f} dB\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("For FM stereo broadcasting, Δf = 75 kHz is the optimal choice because:\n\n")
        f.write("1. It is the standard frequency deviation for FM broadcasting (FCC specification)\n\n")
        f.write("2. It provides a good balance between:\n")
        f.write("   - Spectrum efficiency (256 kHz bandwidth per Carson's rule)\n")
        f.write("   - Audio quality (good SNR improvement)\n")
        f.write("   - Compatibility with standard FM receivers\n\n")
        f.write("3. The 200 kHz channel spacing used in FM broadcasting is designed for\n")
        f.write("   this deviation, allowing adjacent channel operation.\n\n")
        f.write("4. Higher deviations (100 kHz) provide slightly better SNR but:\n")
        f.write("   - Exceed standard specifications\n")
        f.write("   - May cause adjacent channel interference\n")
        f.write("   - Are not compatible with standard receivers\n\n")
        f.write("5. Lower deviations (50 kHz) waste the available channel bandwidth\n")
        f.write("   without providing any benefit.\n\n")
    
    print(f"Saved analysis to: {output_path}")


def main():
    """Main function to run Task 1 analysis."""
    print("=" * 80)
    print("TASK 1: FREQUENCY DEVIATION EFFECTS ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Load audio files
    print("\nLoading audio files...")
    left, right, fs = load_audio("audio/left.wav", "audio/right.wav")
    print(f"Loaded: {len(left)/fs:.2f} seconds of stereo audio at {fs} Hz")
    
    # Frequency deviations to test
    deviations = [50e3, 75e3, 100e3]  # 50, 75, 100 kHz
    input_snr = 25  # dB
    
    # Run analysis for each deviation
    results = []
    for delta_f in deviations:
        result = run_frequency_deviation_analysis(left, right, fs, delta_f, input_snr)
        results.append(result)
    
    # Create outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # a) Bandwidth comparison table
    print("\n" + create_bandwidth_table(results))
    
    # b) SNR vs deviation plot
    plot_snr_vs_deviation(results, "outputs/task1_snr_vs_deviation.png")
    
    # Additional: Bandwidth comparison plot
    plot_bandwidth_comparison(results, "outputs/task1_bandwidth_comparison.png")
    
    # c) Write complete analysis with trade-off discussion
    write_analysis(results, "outputs/task1_bandwidth_table.txt")
    
    print("\n" + "=" * 80)
    print("TASK 1 COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - outputs/task1_bandwidth_table.txt (comparison table + analysis)")
    print("  - outputs/task1_snr_vs_deviation.png (SNR plot)")
    print("  - outputs/task1_bandwidth_comparison.png (bandwidth comparison)")


if __name__ == "__main__":
    main()
