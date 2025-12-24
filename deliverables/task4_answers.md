# Task 4: Filter Design Impact Analysis

## Objective

Analyze how the order of the Pilot Bandpass Filter (BPF) affects the performance of the FM stereo receiver, specifically focusing on **Channel Separation** and signal recovery quality.

## 1. Pilot Extraction Filter Response

We designed Butterworth bandpass filters centered at 19 kHz with a bandwidth of 1 kHz ($19 \pm 0.5$ kHz) for varying orders ($N \in \{4, 8, 12\}$).

![Filter Frequency Responses](../outputs/task4/filter_responses.png)

**Observation:**

- Higher order filters provide sharper roll-off and better isolation of the pilot tone from adjacent noise/signals.
- However, higher order IIR filters (like the Butterworth used here) introduce larger **phase delays** (group delay), particularly near the cutoff edges.

## 2. Channel Separation Analysis

We measured channel separation in **both directions** to ensure symmetry:

1. **L $\to$ R Leakage**: Transmitting Left-only, measuring noise in Right.
2. **R $\to$ L Leakage**: Transmitting Right-only, measuring noise in Left.

### Results

| Filter Order |   Mode    | Signal RMS | Leakage RMS | Separation (dB) |
| :----------: | :-------: | :--------: | :---------: | :-------------: |
|    **4**     | L $\to$ R |   0.1974   |   0.0149    |  **22.47 dB**   |
|    **4**     | R $\to$ L |   0.2056   |   0.0157    |  **22.35 dB**   |
|              |           |            |             |                 |
|    **8**     | L $\to$ R |   0.1973   |   0.0230    |    18.65 dB     |
|    **8**     | R $\to$ L |   0.2055   |   0.0241    |    18.61 dB     |
|              |           |            |             |                 |
|    **12**    | L $\to$ R |   0.1971   |   0.0379    |    14.33 dB     |
|    **12**    | R $\to$ L |   0.2054   |   0.0395    |    14.31 dB     |

![Separation Trend](../outputs/task4/separation_trend.png)

### Key Finding: Inverse Relationship

The results show a clear trend: **As the filter order increases, Channel Separation decreases.** Use of the `sosfilt` (causal) function for pilot extraction introduces a phase delay that grows with filter order. This desynchronizes the regenerated 38 kHz carrier from the payload, causing leakage.

## 3. Time-Domain Signal Recovery

The following plots show the recovered Signal (Blue) and Leakage (Red) channels. Ideally, the Red line should be flat (silence).

### Order 4 (Best Separation)

![Waveform Order 4 L->R](../outputs/task4/waveform_order_4_L_to_R.png)

### Order 8

![Waveform Order 8 L->R](../outputs/task4/waveform_order_8_L_to_R.png)

### Order 12 (Worst Separation)

![Waveform Order 12 L->R](../outputs/task4/waveform_order_12_L_to_R.png)
_Note how the amplitude of the "noise" on the silent channel (Red) increases as the filter order goes up._

## 4. Conclusion

While higher-order filters are generally desirable for rejecting noise and interferers, in the context of **coherent FM demodulation**, they introduce detrimental phase shifts. Without delay compensation (or using zero-phase filtering like `filtfilt`), **lower-order filters (Order 4)** generally yield better stereo separation because they maintain tighter phase alignment between the pilot and the multiplexed signal.
