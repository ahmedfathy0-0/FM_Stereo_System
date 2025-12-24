# Task 3: Channel Separation Analysis

## Objective

Quantify the baseline channel separation of the FM Stereo Receiver and identify the components limiting its performance.

## Measurement

We injected a **1 kHz tone** into the Left channel (Right = Silence) and measured the recovered RMS levels.

- **Recovered Channel Separation**: **23.31 dB**

## Analysis

### b) Limiting Component

The primary limiting factor is the **Pilot Extraction Filter**.

- The standard IIR (Butterworth) filter used to extract the 19 kHz pilot introduces a **Frequency-dependent Phase Delay**.
- Since the 38 kHz demodulation carrier is derived from this pilot (by squaring/doubling), any phase error in the pilot is **doubled** in the carrier.
- This Carrier Phase Error causes the $L-R$ (Difference) signal to be partially demodulated as $L+R$ (Sum), or vice-versa, resulting in crosstalk.

### c) Proposed Improvement

**Modification:** Use **Zero-Phase Filtering** or **Delay Compensation**.

1.  **Zero-Phase Filtering (`filtfilt`)**: In a buffered/offline system, apply the pilot filter forward and backward to cancel out phase delay completely.
2.  **Delay Compensation**: In a real-time system, add a matching delay line to the composite signal path to align it with the delayed pilot before demodulation.

_Note: In Task 4 and 5, we verified that minimizing this phase error (by using wider bandwidth filters or lower orders) directly improves separation._
