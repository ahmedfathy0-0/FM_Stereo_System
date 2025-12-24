# Task 5: System Robustness Analysis

## Objective

Assess the robustness of the FM Stereo Receiver against pilot tone frequency errors ($\pm 500$ Hz), which commonly occur due to oscillator drift in real hardware.

we shifted the pilot tone frequency in the transmitter while keeping the receiver tuned to the nominal 19 kHz.

## Results

### a) Channel Separation vs. Pilot Frequency Error

We swept the pilot frequency offset from $-500$ Hz to $+500$ Hz and measured the resulting channel separation.

![Separation vs Frequency](../outputs/task5/robustness_curve.png)

| Offset (Hz) | Separation (dB) |
| :---------: | :-------------: |
|    -500     |      7.14       |
|    -400     |      14.25      |
|    -300     |    **20.00**    |
|    -200     |      24.71      |
|    -100     |    **25.03**    |
|      0      |      22.33      |
|     +50     |    **20.69**    |
|    +100     |      18.93      |
|    +200     |      15.03      |
|    +300     |      9.81       |
|    +400     |      5.03       |
|    +500     |      0.35       |

**Observation:**
Separation peaks around -100 Hz offset rather than 0 Hz. This suggests that the receiver's filters introduce a baseline phase delay that is accidentally "compensated" by a slightly lower frequency pilot (which experiences less delay or matches the system's delay characteristic better at that point).

### b) Spectrum Analysis at +500 Hz Error

The following plot shows the spectrum of the recovered Left and Right channels when the pilot frequency error is $+500$ Hz.

![Spectrum at +500 Hz](../outputs/task5/spectrum_plus500Hz.png)

**Observation:**
The "Silence" channel (Red) has almost the same energy level as the "Signal" channel (Blue).

- **Separation is effectively 0 dB (~0.35 dB measured).**
- This occurs because at +500 Hz, the pilot is significantly phase-shifted by the bandpass filter.
- The regenerated 38 kHz carrier is therefore phase-shifted by roughly double that amount.
- A phase error of $\approx 45^\circ$ causes the L-R signal to demodulate into the wrong quadrant, causing massive crosstalk (L leaking into R).

### c) Tolerance Range

We define "tolerance" as the range of frequency offsets where Channel Separation remains **above 20 dB**.

**Result:**
The system can tolerate pilot frequency errors in the range:
**[-300 Hz, +50 Hz]**

Outside this range, the phase error in the extracted pilot becomes too large for high-fidelity stereo separation.
