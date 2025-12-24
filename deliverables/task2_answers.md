# Task 2: Noise Immunity Analysis

## Objective

Assess the system's performance (SNR, Channel Separation, THD) under varying channel noise conditions (Input SNR: 5 dB to 25 dB) with a fixed frequency deviation $\Delta f = 75$ kHz.

## Results Table

| Input SNR (dB) | Output SNR (dB) | Sep L$\to$R (dB) | Sep R$\to$L (dB) | THD (%) |
| :------------: | :-------------: | :--------------: | :--------------: | :-----: |
|     **5**      |      5.05       |       4.03       |       4.66       |  1.79   |
|     **10**     |      23.47      |      18.73       |      18.62       |  0.23   |
|     **15**     |      28.90      |      21.89       |      23.09       |  0.13   |
|     **20**     |      33.97      |      23.01       |      23.12       |  0.06   |
|     **25**     |      38.84      |      23.31       |      23.36       |  0.05   |

## Plots

### a) Output SNR vs Input SNR & b) Separation vs Input SNR

![Task 2 Results](../outputs/task2/graphs_results.png)

_(Note: The graph includes THD as well, showing inverse correlation with SNR)_

## Analysis

### c) Threshold SNR Effect

**Threshold SNR:** Approximately **10 dB**.

**Observation:**

- Below 10 dB Input SNR (e.g., at 5 dB), the Output SNR matches the Input SNR (~5 dB), indicating the FM system has "crashed" or lost its coding gain (FM Threshold Effect). THD spikes to 1.79%.
- Above 10 dB (e.g., at 15 dB), the Output SNR jumps significantly (to ~29 dB), showing the expected "FM Improvement Factor."
- Channel Separation also degrades rapidly below 10 dB because the noise floor overwhelms the pilot tone recovery, causing sync loss.

**Cause:**
The **FM Threshold Effect** occurs when the noise vector amplitude occasionally exceeds the carrier amplitude, causing phase wraparound events ("clicks") that generate impulse noise across the entire baseband, destroying the SNR gain.
