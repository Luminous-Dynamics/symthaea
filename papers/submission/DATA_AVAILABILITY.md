# Data Availability Statement

## Paper 01: The Master Equation of Consciousness

---

## Primary Datasets

### Sleep Stage Validation
**Dataset**: Sleep-EDF Database Expanded
**Source**: PhysioNet (https://physionet.org/content/sleep-edfx/1.0.0/)
**License**: Open Data Commons Attribution License (ODC-BY)
**Citation**:
> Kemp B, Zwinderman AH, Tuk B, Kamphuisen HA, Oberyé JJ. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Trans Biomed Eng. 2000;47(9):1185-94.

**Access**: Freely available with PhysioNet Credentialed Access

### Anesthesia Validation
**Dataset**: Synthetic data generated using published parameters
**Basis**: Purdon et al. 2013, Mashour & Avidan 2015
**Generation Code**: `analysis/compute_components.py`

---

## Derived Data

All derived data products are available at:
**Repository**: [GitHub/Zenodo URL - to be created upon acceptance]
**DOI**: [To be assigned upon publication]

### Included Data Products

1. **Component values by sleep stage** (Table S1)
   - Format: CSV
   - Size: < 1 KB
   - Contents: Mean ± SD for Φ, B, W, A, R, C across Wake, N1, N2, N3, REM

2. **Component values by anesthesia depth** (Table S2)
   - Format: CSV
   - Size: < 1 KB
   - Contents: Mean ± SD across Awake, Sedation, Light, Moderate, Deep

3. **Model comparison statistics** (Table S3)
   - Format: CSV
   - Size: < 1 KB
   - Contents: AIC, BIC, r values for min/product/geometric/weighted

---

## Code Availability

**Repository**: [GitHub URL - to be created]
**Language**: Python 3.11+
**License**: MIT

### Main Modules
- `compute_components.py`: Core component computation
- `sleep_edf_analysis.py`: Sleep-EDF validation pipeline
- `generate_figures.py`: Figure generation scripts

### Dependencies
- numpy ≥1.24
- scipy ≥1.11
- mne ≥1.6 (optional, for EDF reading)
- matplotlib ≥3.8 (optional, for visualization)

---

## Reproducibility

To reproduce all analyses:

```bash
# Clone repository
git clone [repository-url]
cd five-component-consciousness

# Install dependencies
pip install -r requirements.txt

# Run sleep stage analysis (synthetic)
python compute_components.py

# Run sleep stage analysis (real data, requires MNE)
pip install mne
python sleep_edf_analysis.py

# Generate figures
pip install matplotlib
python generate_figures.py
```

All random seeds are fixed for reproducibility.

---

## Contact for Data Requests

For questions about data access:
[corresponding.author@institution.edu]

For code issues:
[GitHub Issues URL]

---

## Ethical Statement

No new human subjects data was collected for this study. All validation used:
1. Publicly available datasets (PhysioNet)
2. Synthetic data generated from published parameters

---

## Funding and Support

[To be completed with actual funding sources]

---

*Last updated: [Date]*
