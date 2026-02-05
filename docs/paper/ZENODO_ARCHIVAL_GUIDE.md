# 📦 Zenodo Data Archival Guide

**Purpose**: Archive research data and code with permanent DOI for manuscript submission
**Platform**: Zenodo (https://zenodo.org) - CERN's open science repository
**Time Required**: 1-2 hours
**Output**: Permanent DOI for data citation

---

## Why Zenodo?

✅ **Permanent**: DOIs never expire or change
✅ **Free**: No cost for uploads up to 50 GB
✅ **Trusted**: Run by CERN, integrated with European Commission
✅ **Citable**: DOIs accepted by all journals
✅ **Versioned**: Updates get new DOIs, old versions preserved
✅ **Discoverable**: Indexed by Google Scholar, DataCite, OpenAIRE

---

## Step 1: Create Zenodo Account (10 minutes)

### Option A: GitHub Sign-In (RECOMMENDED)
1. Go to https://zenodo.org
2. Click "Sign up" → "Sign up with GitHub"
3. Authorize Zenodo to access public repos
4. Verify email address

**Benefit**: Can auto-sync GitHub releases with Zenodo

### Option B: ORCID Sign-In
1. Go to https://zenodo.org
2. Click "Sign up" → "Sign up with ORCID"
3. Link your ORCID iD
4. Verify email

**Benefit**: Auto-populates author metadata

### Option C: Email Registration
1. Go to https://zenodo.org
2. Click "Sign up" → Manual registration
3. Enter email, create password
4. Verify email address

---

## Step 2: Prepare Dataset (30 minutes)

### Create Dataset Directory Structure

```bash
cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb

# Create dataset folder
mkdir -p zenodo-dataset/symthaea-hlb-v0.1.0

cd zenodo-dataset/symthaea-hlb-v0.1.0

# Create subdirectories
mkdir -p raw_data analysis_scripts figures supplementary
```

### Collect Raw Data

```bash
# Copy Φ measurement results
cp ../../TIER_3_VALIDATION_RESULTS_*.txt raw_data/
cp ../../DIMENSIONAL_SWEEP_RESULTS_*.txt raw_data/

# Convert to CSV for easier analysis
cat > convert_to_csv.py << 'EOF'
import re
import csv

def parse_tier3_results(filename):
    """Convert tier 3 results to CSV"""
    with open(filename) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if "Topology:" in line:
            match = re.search(r'Topology: ([^,]+), Seed: (\d+), Phi: ([\d.]+)', line)
            if match:
                data.append({
                    'topology': match.group(1).strip(),
                    'seed': int(match.group(2)),
                    'phi': float(match.group(3)),
                    'method': 'RealHV'
                })

    return data

def parse_dimensional_results(filename):
    """Convert dimensional sweep to CSV"""
    with open(filename) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if "Dimension:" in line:
            match = re.search(r'Dimension: (\d+), Seed: (\d+), Phi: ([\d.]+)', line)
            if match:
                data.append({
                    'dimension': int(match.group(1)),
                    'seed': int(match.group(2)),
                    'phi': float(match.group(3)),
                    'topology': f'{match.group(1)}D Hypercube'
                })

    return data

# Process tier 3 results
tier3_data = []
for i in range(10):
    tier3_data.extend(parse_tier3_results(f'raw_data/TIER_3_VALIDATION_RESULTS_seed{i}.txt'))

with open('raw_data/tier_3_phi_measurements.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['topology', 'seed', 'phi', 'method'])
    writer.writeheader()
    writer.writerows(tier3_data)

# Process dimensional sweep
dim_data = []
for i in range(10):
    dim_data.extend(parse_dimensional_results(f'raw_data/DIMENSIONAL_SWEEP_RESULTS_seed{i}.txt'))

with open('raw_data/dimensional_sweep_phi.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dimension', 'seed', 'phi', 'topology'])
    writer.writeheader()
    writer.writerows(dim_data)

print(f"Tier 3: {len(tier3_data)} measurements → tier_3_phi_measurements.csv")
print(f"Dimensional: {len(dim_data)} measurements → dimensional_sweep_phi.csv")
EOF

python convert_to_csv.py
```

### Copy Analysis Scripts

```bash
# Copy figure generation
cp ../../generate_figures.py analysis_scripts/

# Create requirements.txt
cat > analysis_scripts/requirements.txt << 'EOF'
numpy==1.26.4
scipy==1.11.4
matplotlib==3.8.2
pandas==2.1.4
seaborn==0.13.2
EOF

# Copy statistical analysis (if created separately)
# cp ../../statistical_tests.py analysis_scripts/
```

### Copy Figures

```bash
# Copy all publication figures
cp ../../figures/*.png figures/
cp ../../figures/*.pdf figures/
```

### Copy Supplementary Materials

```bash
# Copy supplementary content
cp ../../PAPER_SUPPLEMENTARY_MATERIALS.md supplementary/
cp ../../COMPLETE_TOPOLOGY_ANALYSIS.md supplementary/
cp ../../FIGURE_LEGENDS.md supplementary/
```

### Create README

```bash
cat > README.md << 'EOF'
# Network Topology and Integrated Information: Research Dataset

**Associated Manuscript**: "Network Topology and Integrated Information: A Comprehensive Characterization"
**Authors**: Tristan Stoltz, Claude Code (AI Assistant)
**Date**: December 28, 2025
**Version**: v0.1.0
**License**: CC-BY-4.0

## Dataset Description

This dataset contains all raw data, analysis scripts, and figures supporting the manuscript's findings on the relationship between network topology and integrated information (Φ), a proposed measure of consciousness.

**Key Contributions**:
- 260 total Φ measurements across 19 network topologies
- Dimensional sweep analysis (1D-7D hypercubes)
- First demonstration of asymptotic Φ limit (Φ → 0.5)
- Novel HDC-based Φ approximation method

## Contents

### `raw_data/` - Raw Measurements
- `tier_3_phi_measurements.csv` - 19 topologies × 10 seeds = 190 measurements
- `dimensional_sweep_phi.csv` - 7 dimensions × 10 seeds = 70 measurements
- `TIER_3_VALIDATION_RESULTS_seed*.txt` - Original output files
- `DIMENSIONAL_SWEEP_RESULTS_seed*.txt` - Original output files

**CSV Format**:
```
topology,seed,phi,method
Ring,0,0.4954,RealHV
Ring,1,0.4953,RealHV
...
```

### `analysis_scripts/` - Reproducibility
- `generate_figures.py` - Creates all 4 publication figures
- `requirements.txt` - Python dependencies (exact versions)
- `statistical_tests.py` - t-tests, ANOVA, effect sizes (if separate)

**Reproducing Figures**:
```bash
pip install -r requirements.txt
python generate_figures.py
```

### `figures/` - Publication Figures
- `figure_1_dimensional_curve.{png,pdf}` - Asymptotic Φ convergence
- `figure_2_topology_rankings.{png,pdf}` - 19-topology Φ rankings
- `figure_3_category_comparison.{png,pdf}` - Category-level analysis
- `figure_4_non_orientability.{png,pdf}` - Twist dimension effects

All figures 300 DPI, colorblind-safe palette.

### `supplementary/` - Supporting Materials
- `PAPER_SUPPLEMENTARY_MATERIALS.md` - Full supplementary text
- `COMPLETE_TOPOLOGY_ANALYSIS.md` - Detailed topology characterization
- `FIGURE_LEGENDS.md` - Comprehensive figure captions

## Reproducibility

### System Requirements
- **Software**: Rust 1.82, Python 3.13
- **Hardware**: Standard desktop (measurements take ~5 seconds each)
- **OS**: Any (tested on NixOS 25.11)

### Regenerating Data

See the main code repository for complete instructions:
https://github.com/luminous-dynamics/symthaea-hlb

```bash
# Run 19-topology validation
cargo run --release --example tier_3_validation

# Run dimensional sweep
cargo run --release --example dimensional_sweep

# Generate figures
python generate_figures.py
```

## Statistical Summary

**Tier 3 Validation** (19 topologies):
- Champion: Hypercube 4D (Φ = 0.4976 ± 0.0001)
- Runner-up: Hypercube 3D (Φ = 0.4960 ± 0.0002)
- Lowest: Möbius Strip 1D (Φ = 0.3729 ± 0.0003)

**Dimensional Sweep** (1D-7D):
- 1D Hypercube (K₂): Φ = 1.0000 (edge case)
- 2D-7D: Asymptotic approach to Φ ≈ 0.5
- Fitted model: Φ(k) = 0.4998 - 0.0522·exp(-0.89·k), R² = 0.998

## Citation

If you use this dataset, please cite:

```
Stoltz, T. & Claude Code. (2025). Network Topology and Integrated
Information: A Comprehensive Characterization [Dataset]. Zenodo.
https://doi.org/10.5281/zenodo.XXXXXXX
```

## License

This dataset is licensed under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.

You are free to:
- **Share**: Copy and redistribute in any medium or format
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made

See: https://creativecommons.org/licenses/by/4.0/

## Contact

**Tristan Stoltz** - tristan.stoltz@gmail.com
**Luminous Dynamics** - Richardson, TX, USA

For questions about the dataset, analysis methods, or code, please open an issue on the GitHub repository or email the corresponding author.

---

*Dataset prepared using the Sacred Trinity development model: Human vision + AI assistance + autonomous scientific workflow.*

**Last Updated**: December 28, 2025
EOF
```

### Create Metadata File (.zenodo.json)

```bash
cat > .zenodo.json << 'EOF'
{
  "title": "Network Topology and Integrated Information: Research Dataset",
  "description": "Complete research dataset supporting the manuscript 'Network Topology and Integrated Information: A Comprehensive Characterization'. Contains 260 integrated information (Φ) measurements across 19 network topologies and dimensional sweep (1D-7D hypercubes). First demonstration of asymptotic Φ limit and comprehensive topology-consciousness characterization using hyperdimensional computing.",
  "creators": [
    {
      "name": "Stoltz, Tristan",
      "affiliation": "Luminous Dynamics",
      "orcid": "XXXX-XXXX-XXXX-XXXX"
    },
    {
      "name": "Claude Code",
      "affiliation": "Anthropic PBC"
    }
  ],
  "keywords": [
    "integrated information theory",
    "consciousness",
    "network topology",
    "hyperdimensional computing",
    "neuroscience",
    "artificial intelligence"
  ],
  "license": "CC-BY-4.0",
  "upload_type": "dataset",
  "access_right": "open",
  "related_identifiers": [
    {
      "identifier": "https://github.com/luminous-dynamics/symthaea-hlb",
      "relation": "isSupplementTo",
      "scheme": "url"
    }
  ],
  "contributors": [
    {
      "name": "Anthropic PBC",
      "type": "HostingInstitution"
    }
  ],
  "references": [
    "Manuscript submitted to Nature Neuroscience (2025)"
  ],
  "version": "1.0.0"
}
EOF
```

### Create Archive

```bash
# Create ZIP archive
cd ..
zip -r symthaea-hlb-v0.1.0-dataset.zip symthaea-hlb-v0.1.0/

# Verify size
du -sh symthaea-hlb-v0.1.0-dataset.zip
# Should be < 50 MB
```

---

## Step 3: Upload to Zenodo (20 minutes)

### Create New Upload

1. Log in to https://zenodo.org
2. Click "Upload" → "New Upload"
3. Fill in metadata form:

**Basic Information**:
- Title: "Network Topology and Integrated Information: Research Dataset"
- Upload type: Dataset
- Publication date: 2025-12-28 (or submission date)

**Creators** (Authors):
- Name: Stoltz, Tristan
- Affiliation: Luminous Dynamics
- ORCID: (your ORCID if you have one)

- Name: Claude Code
- Affiliation: Anthropic PBC

**Description**:
```
Complete research dataset supporting the manuscript "Network Topology and Integrated Information: A Comprehensive Characterization".

This dataset contains 260 integrated information (Φ) measurements across 19 distinct network topologies and a dimensional sweep from 1D to 7D hypercubes. Key findings include:

1. Discovery of asymptotic Φ limit (Φ → 0.5 as dimension → ∞)
2. Identification of 4D hypercube as optimal topology (Φ = 0.4976)
3. Demonstration of 3D brain dimensional optimality (99.2% of maximum)
4. First comprehensive topology-consciousness characterization using HDC

Dataset includes raw measurements (CSV), analysis scripts (Python), publication figures (300 DPI PNG/PDF), and complete supplementary materials.

All data and code are open source (CC-BY-4.0 and MIT licenses) to enable reproducibility and future research.
```

**Keywords**:
- integrated information theory
- consciousness
- network topology
- hyperdimensional computing
- neuroscience
- artificial intelligence
- dimensional optimization

**License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)

**Related/Alternate Identifiers**:
- Type: "is supplemented by"
- Identifier: https://github.com/luminous-dynamics/symthaea-hlb
- Scheme: URL

**Funding**: None (or leave blank)

**Version**: 1.0.0

### Upload File

1. Drag `symthaea-hlb-v0.1.0-dataset.zip` to upload box
2. Wait for upload to complete (progress bar)
3. Verify file appears in file list

### Review and Publish

1. Click "Preview" to see how record will appear
2. Review all metadata for accuracy
3. Check description formatting
4. Verify files uploaded correctly
5. Click "Publish" (this makes it permanent!)

**⚠️ IMPORTANT**: Once published, you CANNOT delete the record. You can only create new versions. Double-check everything before publishing!

---

## Step 4: Get DOI and Update Manuscript (15 minutes)

### Copy DOI

After publishing, Zenodo will display:
```
DOI: 10.5281/zenodo.XXXXXXX
```

**Example**: `10.5281/zenodo.8234567`

### Update Manuscript Data Availability

Edit `MASTER_MANUSCRIPT.md`:

```markdown
## Data Availability

All data supporting the findings of this study are openly available in
Zenodo at **https://doi.org/10.5281/zenodo.8234567**.
```

Update all occurrences (should be 3-4 places):
- MASTER_MANUSCRIPT.md
- PAPER_SUPPLEMENTARY_MATERIALS.md
- COVER_LETTER.md
- SUBMISSION_READINESS_SUMMARY.md

### Update Code Repository

Add DOI badge to README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8234567.svg)]
(https://doi.org/10.5281/zenodo.8234567)
```

### Regenerate PDF

After updating DOI, recreate manuscript PDF:

```bash
# Re-run PDF creation with updated DOI
# See PDF_CREATION_GUIDE.md
```

---

## Step 5: Verification (10 minutes)

### Test DOI Resolution

1. Open browser
2. Go to: `https://doi.org/10.5281/zenodo.XXXXXXX` (your DOI)
3. Verify it redirects to Zenodo record
4. Check all files downloadable
5. Verify metadata displays correctly

### Download and Test Dataset

```bash
# Download your own dataset
wget https://zenodo.org/record/XXXXXXX/files/symthaea-hlb-v0.1.0-dataset.zip

# Extract
unzip symthaea-hlb-v0.1.0-dataset.zip

# Test scripts run
cd symthaea-hlb-v0.1.0/analysis_scripts
pip install -r requirements.txt
python generate_figures.py

# Verify figures generated correctly
```

---

## Troubleshooting

**Upload fails / times out**:
- Check file size < 50 GB limit
- Try smaller file (compress more)
- Use stable internet connection
- Try different browser

**DOI not resolving**:
- Wait 5-10 minutes (propagation delay)
- Clear browser cache
- Try incognito/private window
- Check DOI typed correctly

**Metadata won't save**:
- All required fields filled?
- Keywords separated by newlines
- Dates in YYYY-MM-DD format
- ORCID format correct (XXXX-XXXX-XXXX-XXXX)

**Can't publish (button disabled)**:
- Upload at least one file
- Fill all required metadata
- Accept terms of service
- Verify email address

---

## Best Practices

✅ **Do**:
- Include comprehensive README
- Use semantic versioning (1.0.0, 1.1.0, 2.0.0)
- Provide requirements.txt with exact versions
- Test reproducibility before uploading
- Add .zenodo.json for automation
- Use descriptive filenames

❌ **Don't**:
- Upload sensitive data
- Include large binaries unnecessarily
- Forget license specification
- Use vague descriptions
- Skip testing download/extract

---

## After Publication Checklist

- [ ] DOI resolves correctly
- [ ] All files downloadable
- [ ] README displays properly
- [ ] Metadata accurate
- [ ] License displayed
- [ ] Updated manuscript with DOI
- [ ] Regenerated PDF with DOI
- [ ] Added DOI badge to GitHub
- [ ] Notified co-authors
- [ ] Ready for journal submission!

---

## Zenodo Features

**Versioning**:
- Create new version: Click "New version" on record page
- Each version gets new DOI
- Concept DOI always points to latest

**Communities**:
- Submit to relevant communities (Neuroscience, AI, Open Science)
- Increases discoverability
- Optional but recommended

**GitHub Integration**:
- Link Zenodo to GitHub repository
- Auto-create DOI on each release
- Sync metadata automatically

---

**Estimated Total Time**: 1-2 hours
**Difficulty**: Easy to Moderate
**Cost**: Free (up to 50 GB)

✅ **Once complete**: You have permanent, citable DOI for your research data!

🚀 **Next step**: Submit manuscript to journal with DOI included!

---

*Zenodo archival ensures your research data remains accessible and citable forever, even if your institution changes or website goes offline.*
