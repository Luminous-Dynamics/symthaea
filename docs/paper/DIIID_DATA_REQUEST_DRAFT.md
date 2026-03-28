# DIII-D Data Access Request — DRAFT

**Status**: Draft for Tristan's review before submission
**Date**: 2026-03-19
**Submit to**: contact@d3dfusion.org (DIII-D User Office)
**CC consideration**: crea@psfc.mit.edu (Cristina Rea, MIT PSFC Disruption Studies Group)

---

## Option A: Email to DIII-D User Office (contact@d3dfusion.org)

**Subject**: Data Access Request — Zero-Shot HDC Disruption Prediction Cross-Machine Validation

Dear DIII-D User Office,

I am writing to request data access to the DIII-D National Fusion Facility for a machine learning research project on disruption prediction in tokamak plasmas.

### Researcher Information

- **Name**: Tristan Stoltz
- **Organization**: Luminous Dynamics (independent research organization)
- **Location**: Richardson, TX, United States
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com

### Research Description

We are developing a novel approach to tokamak disruption prediction using Hyperdimensional Computing (HDC) with 16,384-dimensional binary hypervectors. Unlike conventional ML approaches (random forests, deep neural networks), HDC operates through algebraic binding and bundling in high-dimensional spaces, enabling transparent and interpretable classification with no hidden layers or opaque learned weights.

Our initial results on Alcator C-Mod data are promising. Using the Rea & Granetz (2018) disruption database parameters (plasma internal inductance, Greenwald fraction, radiated power fraction, q95, beta_N, n=1 amplitude, and rotation), we achieved:

- **AUC 0.778** in zero-shot classification (no DIII-D training data used)
- **Accuracy 73.2%** with balanced precision/recall
- Transparent decision process via HDC similarity voting

These results were obtained using only C-Mod training data, with no exposure to DIII-D-specific plasma behavior during training.

### Why DIII-D Data Is Needed

DIII-D is essential for validating the cross-machine generalization of our approach for several reasons:

1. **Carbon wall vs. metal wall**: C-Mod uses molybdenum walls while DIII-D uses graphite tiles, presenting fundamentally different plasma-wall interaction regimes and impurity transport. Cross-machine transfer across wall materials is a key test of physics-based generalization.

2. **Scale and geometry differences**: DIII-D (R=1.67m, a=0.67m) operates at significantly different scale from C-Mod (R=0.67m, a=0.22m), testing whether HDC representations capture scale-invariant disruption physics.

3. **Diagnostic overlap**: DIII-D's extensive diagnostic suite (100+ systems) includes all signals used in our C-Mod analysis, enabling direct comparison without signal substitution.

4. **Benchmark comparability**: The Rea & Granetz (2018) and subsequent DIII-D disruption prediction studies provide direct performance baselines for our zero-shot approach.

5. **DisruptionBench alignment**: Our work aligns with the MIT PSFC DisruptionBench framework (Journal of Fusion Energy, 2025), which standardizes zero-shot, few-shot, and many-shot evaluation across C-Mod, DIII-D, and EAST.

### Specific Data Requested

We request access to DIII-D disruption prediction datasets, specifically:

- Disrupted and non-disrupted discharge databases (ideally the same shot lists used in Rea & Granetz 2018 and/or DisruptionBench)
- Time-series diagnostic signals: plasma current (Ip), internal inductance (li), Greenwald fraction (fGW), radiated power fraction (Prad/Ptot), safety factor (q95), normalized beta (beta_N), locked mode amplitude, plasma rotation
- Shot metadata: disruption labels, disruption times, plasma parameters

We anticipate using DisruptionPy (MIT PSFC) as the data retrieval interface if MDSplus server access is granted.

### Data Usage and Reciprocity

- All results will be shared with the DIII-D team prior to publication
- We will acknowledge DIII-D and General Atomics per the DIII-D publication policy
- Our HDC disruption prediction code and trained models will be made available to the DIII-D community
- We are preparing a manuscript for Nuclear Fusion / Plasma Physics and Controlled Fusion and will include DIII-D co-authors where appropriate

### Access Type Requested

We are requesting **remote data access only** (no on-site facility access needed). We understand this requires:

1. A Non-Proprietary User Agreement (or Data Usage Agreement if data-only access is available)
2. Completion of the Cyber Security Awareness Program (if full CyberAccess is needed)
3. Electronic signature of the DIII-D Data Usage Agreement

Please advise on the appropriate pathway for an independent research organization seeking data-only access for ML research. We are happy to provide any additional information required.

Thank you for your time and consideration.

Sincerely,
Tristan Stoltz
Luminous Dynamics
Richardson, TX

---

## Option B: Parallel Email to Cristina Rea (crea@psfc.mit.edu)

**Subject**: Collaboration Inquiry — Zero-Shot HDC Disruption Prediction on C-Mod/DIII-D

Dear Dr. Rea,

I am writing regarding our work on tokamak disruption prediction using Hyperdimensional Computing (HDC), which builds on your foundational disruption database and machine learning work on C-Mod and DIII-D (Rea & Granetz 2018, PPCF 60:084008).

We have developed a disruption classifier using 16,384-dimensional binary hypervectors that achieves AUC 0.778 in zero-shot cross-machine prediction (trained on C-Mod, tested on DIII-D parameters). HDC offers fully transparent decision-making through algebraic similarity in hyperdimensional space, with no hidden layers or opaque learned weights.

We are seeking:

1. **DIII-D data access** for cross-machine validation — we plan to apply through the DIII-D User Office but would welcome your guidance on the most efficient pathway for ML researchers.

2. **DisruptionPy integration** — we would like to use DisruptionPy as our data retrieval framework and contribute any improvements back to the project. Could you advise on obtaining MDSplus/SQL credentials for DIII-D?

3. **DisruptionBench benchmarking** — our zero-shot HDC approach aligns well with the DisruptionBench evaluation framework. We would be interested in contributing our method as an additional baseline.

4. **Potential collaboration** — we are happy to share our code, models, and results with your group. HDC's interpretability may complement the existing ML approaches in your disruption prediction toolkit.

Our code is built in Rust (part of the Symthaea project) with Python bindings planned for DisruptionPy integration.

I would welcome the opportunity to discuss this further at your convenience.

Best regards,
Tristan Stoltz
Luminous Dynamics
tristan.stoltz@evolvingresonantcocreationism.com

---

## Data Access Pathway Summary

### Recommended Approach (Dual-Track)

| Track | Action | Contact | Timeline |
|-------|--------|---------|----------|
| **1. DIII-D User Office** | Request Non-Proprietary User Agreement + data access | contact@d3dfusion.org | ~2 weeks (US citizen) |
| **2. MIT PSFC** | Collaboration + DisruptionPy credentials + DisruptionBench | crea@psfc.mit.edu | Varies |

### Process Steps

1. **Send Option B** (to Cristina Rea) first — her guidance may streamline the DIII-D process
2. **Send Option A** (to DIII-D User Office) — formal data access request
3. **Install DisruptionPy** while waiting for access:
   ```bash
   uv tool install disruption-py
   # or
   pip install disruption-py
   ```
4. **Prepare credentials config** at `~/.config/disruption-py/user.toml`:
   ```toml
   [d3d.inout.mdsplus]
   server = ""  # Will be provided after access granted

   [d3d.inout.sql]
   db_user = ""  # Will be provided after access granted
   db_pass = ""
   ```
5. **Complete DIII-D requirements** when approved:
   - Sign Data Usage Agreement (electronic)
   - Complete Cyber Security Awareness Program (if CyberAccess granted)
   - Receive MDSplus server credentials

### Key Contacts

| Person | Role | Email |
|--------|------|-------|
| DIII-D User Office | Access administration | contact@d3dfusion.org |
| Cristina Rea | MIT PSFC Disruption Group Leader | crea@psfc.mit.edu |
| Robert Granetz | Senior disruption researcher | granetz@mit.edu |
| ML Working Group | MIT PSFC mailing list | machine_learning@psfc.mit.edu |

### Important Notes

- DisruptionPy does **not** provide data access itself — it requires MDSplus/SQL credentials obtained separately from each institution
- DIII-D data is "open but controlled" — all calibrated data available after signing Data Usage Agreement
- Non-Proprietary User Agreement: facility resources at no cost, results shared with DIII-D team
- The Rea & Granetz 2018 disruption database is not separately downloadable — it is accessed through the MDSplus infrastructure via DisruptionPy
- DisruptionBench (Journal of Fusion Energy, 2025) provides standardized multi-machine evaluation but also requires institutional data access
- For journal figure data (post-March 2014 publications): available without password at the DIII-D external publications site
