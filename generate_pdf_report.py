
import os
import sys

# Try to import fpdf. If not found, exit gracefully with instruction.
try:
    from fpdf import FPDF
except ImportError:
    print("Error: The 'fpdf' library is not installed.")
    print("Please run the following command to install it:")
    print("    pip install fpdf")
    sys.exit(1)

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'FM Stereo System Report', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, txt):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, txt)
        self.ln()
        
    def add_image_centered(self, image_path, w=150):
        if os.path.exists(image_path):
            self.image(image_path, x=30, w=w)
            self.ln(5)
        else:
            print(f"Warning: Image not found {image_path}")

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

# --- Title Page ---
pdf.set_font('Arial', 'B', 24)
pdf.cell(0, 40, 'FM Stereo Receiver System', 0, 1, 'C')
pdf.set_font('Arial', '', 16)
pdf.cell(0, 10, 'Final Project Report', 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Students:', 0, 1, 'C')
pdf.set_font('Arial', '', 14)
pdf.cell(0, 10, 'Ahmed Fathy 9230162', 0, 1, 'C')
pdf.cell(0, 10, 'Ziad Montaser 9231142', 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, 'Date: 2025-12-25', 0, 1, 'C')

# --- Content ---
# Since we cannot easily parse markdown perfectly without a heavy library, 
# we will construct the report structure manually to ensure high quality PDF output.

pdf.add_page()

# Task 1
pdf.chapter_title('Task 1: Frequency Deviation Effects')
pdf.chapter_body("""Objective: Analyze the trade-off between Frequency Deviation, Bandwidth, and SNR.

Results: 
- 75 kHz deviation offers optimal trade-off (SNR ~31dB, BW ~80kHz).
- Comparison Table:
  50 kHz: SNR 27.42 dB (BW 76.6 kHz)
  75 kHz: SNR 30.90 dB (BW 80.5 kHz)
  100 kHz: SNR 33.39 dB (BW 112.5 kHz)
""")
pdf.add_image_centered('outputs/task1/task1_deviation_vs_snr.png')

# Task 2
pdf.add_page()
pdf.chapter_title('Task 2: Noise Immunity Analysis')
pdf.chapter_body("""Objective: Assess SNR and Separation under varying noise conditions (5-25 dB Input SNR).

Key Findings:
- FM Threshold Effect observed at ~10 dB Input SNR.
- Below 10 dB, performance crashes (Output SNR ~= Input SNR).
- Above 10 dB, significant coding gain is observed.
""")
pdf.add_image_centered('outputs/task2/graphs_results.png', w=180)

# Task 3
pdf.ln(5)
pdf.chapter_title('Task 3: Channel Separation Analysis')
pdf.chapter_body("""Measurement:
- Recovered Separation: 23.31 dB (Baseline).

Limiting Factor:
- Pilot Extraction Filter Phase Delay. The IIR filter introduces delay that misalignment the 38kHz carrier.
""")

# Task 4
pdf.add_page()
pdf.chapter_title('Task 4: Filter Design Impact')
pdf.chapter_body("""Objective: Analyze impact of Pilot Filter Order on Separation.

Results:
- Order 4: ~22.47 dB Separation (Best)
- Order 12: ~14.33 dB Separation (Worst)
- Conclusion: Higher order filters increase phase delay, degrading stereo separation.
""")
pdf.add_image_centered('outputs/task4/separation_trend.png')
pdf.ln(5)
pdf.chapter_body("Waveform Comparison (Order 4 vs 12):")
pdf.add_image_centered('outputs/task4/waveform_order_4_L_to_R.png', w=120)
pdf.add_image_centered('outputs/task4/waveform_order_12_L_to_R.png', w=120)

# Task 5
pdf.add_page()
pdf.chapter_title('Task 5: System Robustness')
pdf.chapter_body("""Objective: Test tolerance to pilot frequency errors (+/- 500 Hz).

Results:
- Tolerance Range (>20dB separation): -300 Hz to +50 Hz.
- At +500 Hz, separation is lost (~0 dB).
""")
pdf.add_image_centered('outputs/task5/robustness_curve.png')
pdf.ln(5)
pdf.chapter_body("Spectrum at +500 Hz Error (Showing Leakage):")
pdf.add_image_centered('outputs/task5/spectrum_plus500Hz.png')


# Appendix
pdf.add_page()
pdf.chapter_title('Appendix: Source Code')
pdf.chapter_body("The complete source code is attached in 'all_project_code.py'.")

# Output
pdf_filename = "deliverables/project_document.pdf"
pdf.output(pdf_filename, 'F')
print(f"Success: PDF generated at {pdf_filename}")
