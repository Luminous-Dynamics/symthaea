# Cross-Machine Tokamak Disruption Prediction: Data Access Guide

**Purpose**: Actionable guide for accessing multi-machine disruption prediction datasets to validate zero-shot HDC disruption prediction across tokamaks.

**Researcher**: Tristan Stoltz, Luminous Dynamics
**Date**: 2026-03-19

---

## Table of Contents

1. [DIII-D (General Atomics)](#1-diii-d-general-atomics-san-diego)
2. [JET (Joint European Torus)](#2-jet-joint-european-torus-culham-uk)
3. [ITPA Global Disruption Database](#3-itpa-global-disruption-database)
4. [DisruptionBench & DisruptionPy](#4-disruptionbench--disruptionpy-mit-psfc)
5. [Other Open Datasets & Tools](#5-other-open-datasets--tools)
6. [Summary Table](#6-summary-table)
7. [Recommended Action Plan](#7-recommended-action-plan)
8. [Key References](#8-key-references)

---

## 1. DIII-D (General Atomics, San Diego)

### Overview

DIII-D is the largest magnetic fusion facility in the US, operated by General Atomics for the DOE. It has over 100 individual diagnostic systems and has been a primary source of disruption prediction ML research since Rea & Granetz (2018).

### Data Access

- **Policy**: DIII-D data is "open but controlled." All calibrated and derived data is available to users who sign a Data Usage Agreement.
- **Data-only access**: You do NOT need physical facility access. Users who only need data (no computing resources) can complete the Data Usage Policy form only.
- **Full access**: If you need access to computing resources, data servers, or the DIII-D website, complete the Cyber Access process.

**Application steps**:

1. Review DIII-D capabilities at [d3dfusion.org](https://d3dfusion.org/)
2. Contact relevant division leaders via [d3dfusion.org/become-a-user](https://d3dfusion.org/become-a-user/) to discuss proposed work
3. Complete the Data Usage Agreement and/or Cyber Access form at [diii-d.gat.com/ssl_form/cyberaccess](https://diii-d.gat.com/ssl_form/cyberaccess/)
4. For new research projects, submit a Project Information Form; for grants, a Record of Discussion (ROD) is required

**Timeline**:
- US citizens: ~2 weeks for access
- Foreign nationals: 2+ months (federal approval required)

### Data Format & Infrastructure

| Aspect | Detail |
|--------|--------|
| **Storage** | MDSplus, PTDATA, video camera archives |
| **Access method** | MDSplus servers (remote), SQL relational DB mirror |
| **Format** | MDSplus tree structure; can be extracted to HDF5/CSV via DisruptionPy |
| **Diagnostics** | 100+ systems: Thomson scattering, ECE, MSE, magnetics, SXR, bolometry, interferometry, Mirnov coils, charge exchange, polarimetry |

### Dataset Scale

| Dataset | Shots |
|---------|-------|
| Rea & Granetz 2018 study | Large sets of disrupted + non-disrupted discharges on DIII-D |
| 2022-2024 ML database | 2,094 disruption + 4,858 non-disruption shots, 16 diagnostic signals |
| 2019-2022 carbon divertor campaigns | 668 disruption + 113 non-disruption shots |

### Wall Material

DIII-D has historically used **graphite** (carbon) plasma-facing components. Some campaigns have used SiC-coated walls. This is distinct from JET's ITER-Like Wall (beryllium/tungsten). The carbon vs. metal wall distinction is important for cross-machine transfer learning since wall material significantly affects impurity transport and disruption characteristics.

### Key Contacts

- DIII-D User Office: [d3dfusion.org/become-a-user](https://d3dfusion.org/become-a-user/)
- General Atomics Fusion: [ga.com/magnetic-fusion/diii-d](https://www.ga.com/magnetic-fusion/diii-d)

---

## 2. JET (Joint European Torus, Culham, UK)

### Overview

JET was the world's largest tokamak (operated 1984-2023, now decommissioned). It produced the only D-T fusion experiments and installed an ITER-Like Wall (ILW) with beryllium limiters and tungsten divertor in 2009-2011. JET data is managed by EUROfusion and hosted at the UKAEA Culham Centre.

### Data Access

- **Eligibility**: Must work at or be affiliated with a **EUROfusion member institution**.
- **Non-EUROfusion researchers**: Require additional approval from the INCO Responsible Officer. This adds time to the approval process.
- **Application portal**: [users.jetdata.eu](https://users.jetdata.eu/)

**Application steps**:

1. Validate your work email address at the JET Data Centre account request portal
2. Complete the details form provided during validation
3. Await multi-stage approval: JDC Librarian -> Scientific Coordinator (your org) -> INCO Responsible Officer (non-beneficiaries) -> TE-PSO -> EUROfusion Head of Plasma Science Division

**Timeline**:
- EUROfusion members: ~1 week (5 working days after all approvals)
- Non-EUROfusion: Several weeks to months (additional INCO approval step)

### Data Format & Infrastructure

| Aspect | Detail |
|--------|--------|
| **Raw data** | JET Pulse Files (JPF) |
| **Processed data** | Processed Pulse Files (PPF) |
| **Access layer** | Simple Access Layer (SAL) — unified API for JPF/PPF |
| **Remote access** | NoMachine remote desktop to Heimdall analysis cluster |
| **Format** | Proprietary JET format; extractable to standard formats |

### Key Datasets

| Dataset | Description |
|---------|-------------|
| **de Vries et al. (2011)** | Survey of 2,309 disruptions over a decade of JET operations. Classified by root cause: NTM locking (most common), human error (second), density limit, etc. Published in *Nuclear Fusion* 51, 053018. |
| **JET-ILW disruption data** | Disruptions with ITER-Like Wall (Be/W), distinct from earlier carbon-wall operation. Studied in de Vries (2014) "Disruption causes during first operations with JET ITER-like wall." |
| **JET-ILW confinement database** | EUROfusion global confinement database for JET-ILW. |
| **D-T campaign data** | JET conducted D-T campaigns (DTE1 1997, DTE2 2021-2022). Access likely restricted; contact EUROfusion. |

### Wall Material Distinction

JET is uniquely valuable because it operated with **both** carbon walls (pre-2011) and the **ITER-Like Wall** (Be/W, post-2011). This provides a natural experiment for studying wall-material effects on disruption characteristics — directly relevant to ITER predictions.

### Key Contacts

- JET Data Centre: [users.jetdata.eu](https://users.jetdata.eu/)
- EUROfusion: [euro-fusion.org](https://euro-fusion.org/)
- UKAEA Scientific Publications: [scientific-publications.ukaea.uk](https://scientific-publications.ukaea.uk/)

---

## 3. ITPA Global Disruption Database

### Overview

The International Disruption Database (IDDB), developed under the ITPA MHD Topical Group, is the most comprehensive multi-machine disruption database available. It aims to find commonalities between disruption characteristics across a wide variety of tokamaks for extrapolation to ITER.

### Machines Included (9 tokamaks)

**Conventional aspect ratio**:
- ADITYA (India)
- Alcator C-Mod (MIT, USA) — decommissioned
- ASDEX Upgrade (IPP Garching, Germany)
- DIII-D (General Atomics, USA)
- JET (Culham, UK) — decommissioned
- JT-60U (QST, Japan) — decommissioned, replaced by JT-60SA
- TCV (EPFL, Switzerland)

**Spherical tokamaks**:
- MAST (UKAEA, UK) — upgraded to MAST-U
- NSTX (PPPL, USA) — upgraded to NSTX-U

### Data Format & Structure

| Aspect | Detail |
|--------|--------|
| **Active database** | MDSplus, maintained by General Atomics (restricted access) |
| **Public frozen version** | Available on Harvard Dataverse |
| **SQL mirror** | Daily backups of MDSplus and SQL datasets from each device |
| **Parameters** | ~50 scalar variables per shot |
| **Content** | Device configuration, pre-disruption Ip/shape/magnetics/kinetics, current decay rate/waveform, halo currents, impurity injection data |

### Parameter Categories

The ~50 scalar variables are organized into:
- **Base variables**: device ID, shot number, Ip, BT, q95, elongation, triangularity, ne, Te, beta, li, Wmhd
- **Halo current variables**: halo current fraction, toroidal peaking factor, duration
- **Impurity injection variables**: species, quantity, timing (for mitigated disruptions)
- **Disruption characteristics**: thermal quench time, current quench time, radiated energy fraction

### Access

**Public (frozen) version**:
- **Harvard Dataverse**: [doi:10.7910/DVN/NXDX6U](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NXDX6U)
- Freely downloadable, no approval needed
- Suitable for initial analysis and cross-machine studies

**Active (live) version**:
- Maintained on MDSplus at General Atomics
- Restricted access; contact ITPA MHD Topical Group
- Contains more recent data than frozen version

**Timeline**: Immediate for Harvard Dataverse frozen version.

### Key Publications

- Granetz, R.S. et al. (2016). "The ITPA disruption database." *Nuclear Fusion* 56, 026013. DOI: [10.1088/0029-5515/56/2/026013](https://doi.org/10.1088/0029-5515/56/2/026013)
- Wesley, J.C. et al. (2006). "Disruption characterization and database activities for ITER." IAEA FEC.
- EUROfusion pre-print: [EFDP14025](https://scipub.euro-fusion.org/wp-content/uploads/2014/11/EFDP14025.pdf)
- OSTI record: [osti.gov/pages/biblio/1371725](https://www.osti.gov/pages/biblio/1371725)

---

## 4. DisruptionBench & DisruptionPy (MIT PSFC)

### DisruptionBench

A standardized benchmarking platform for ML-driven disruption prediction, published in *Journal of Fusion Energy* (2025).

| Aspect | Detail |
|--------|--------|
| **Machines** | Alcator C-Mod, DIII-D, EAST |
| **Tasks** | 9 tasks: zero-shot, few-shot, many-shot transfer learning |
| **Models evaluated** | Random Forest, Hybrid Deep Learner (HDL), GPT-2-based transformer, Continuous Convolutional Neural Network (CCNN) |
| **Best performance** | CCNN achieves AUC 0.974 on C-Mod |
| **Publication** | [Springer: s10894-025-00495-2](https://link.springer.com/article/10.1007/s10894-025-00495-2) |

DisruptionBench is directly relevant to our zero-shot HDC validation — it defines the exact benchmark tasks and baselines we need to beat.

### DisruptionPy

Open-source Python framework for building ML-ready disruption datasets from raw tokamak data.

| Aspect | Detail |
|--------|--------|
| **GitHub** | [github.com/MIT-PSFC/disruption-py](https://github.com/MIT-PSFC/disruption-py) |
| **PyPI** | `pip install disruption-py` |
| **Zenodo DOI** | [10.5281/zenodo.13935223](https://zenodo.org/records/18612738) |
| **Supported machines** | Alcator C-Mod (MDSplus+SQL), DIII-D (MDSplus+SQL), EAST (MDSplus+SQL), HBT-EP (MDSplus), MAST (S3) |
| **Output formats** | Multiple (CSV, HDF5, etc.) |
| **Requirements** | Python 3.x, MDSplus library, SQL access (machine-dependent) |

**Important caveat**: DisruptionPy retrieves data from MDSplus servers but does NOT provide access to those servers. You must obtain authorization from each institution independently.

### MIT Disruptions Group

- **Website**: [disruptions.mit.edu](https://disruptions.mit.edu/)
- **Lead**: Cristina Rea (MIT PSFC)
- **Team**: [disruptions.mit.edu/team](https://disruptions.mit.edu/team/)

---

## 5. Other Open Datasets & Tools

### FRNN (Fusion Recurrent Neural Network) — Princeton

| Aspect | Detail |
|--------|--------|
| **GitHub** | [github.com/PPPLDeepLearning/plasma-python](https://github.com/PPPLDeepLearning/plasma-python) |
| **Paper** | Kates-Harbeck, Svyatkovskiy & Tang (2019). "Predicting disruptive instabilities in controlled fusion plasmas through deep learning." *Nature* 568, 526-531. |
| **Data** | DIII-D + JET (cross-machine transfer demonstrated) |
| **Performance** | 95% true positive, <3% false alarm |
| **Note** | Code is open-source; data requires institutional access |

### J-TEXT Cross-Tokamak Transfer

| Aspect | Detail |
|--------|--------|
| **Institution** | Huazhong University of Science and Technology (China) |
| **Key result** | Model trained on J-TEXT transferred to EAST with only 20 discharges, matching performance of models trained on 1,896 EAST discharges |
| **Papers** | Zheng et al. (2023). *Communications Physics* 6, 181. [nature.com/articles/s42005-023-01296-9](https://www.nature.com/articles/s42005-023-01296-9) |
| **Data** | Not publicly available; contact Huazhong UST |

### EAST Tokamak (ASIPP, China)

| Aspect | Detail |
|--------|--------|
| **Institution** | Institute of Plasma Physics, Chinese Academy of Sciences |
| **Data access** | Via MDSplus; authorization required from ASIPP |
| **Supported by** | DisruptionPy (MDSplus+SQL) |
| **Key feature** | Superconducting, steady-state capable; closer to ITER parameters |

### KSTAR (Korea)

| Aspect | Detail |
|--------|--------|
| **Institution** | Korea Institute of Fusion Energy (KFE) |
| **Research** | Disruption prediction via random forest and multimodal deep learning |
| **Data access** | Contact KFE; not publicly available |

### JET ML Disruption Code (Croonen)

| Aspect | Detail |
|--------|--------|
| **GitHub** | [github.com/JoostCroonen/ML_Tokamak_Disruption_Prediction](https://github.com/JoostCroonen/ML_Tokamak_Disruption_Prediction) |
| **Data** | Trained on JET data |
| **Note** | Code open-source; data requires JET access |

### Kaggle / Competitions

No active Kaggle competitions for tokamak disruption prediction were found as of March 2026. The closest open ML challenge is **DisruptionBench** (see Section 4).

---

## 6. Summary Table

| Dataset | Machines | Size | Format | Access | Timeline | Cost |
|---------|----------|------|--------|--------|----------|------|
| **ITPA IDDB (Harvard)** | 9 tokamaks | ~50 vars/shot, thousands of shots | MDSplus/SQL/CSV | Public, free download | Immediate | Free |
| **DIII-D** | DIII-D | 2,094+ disruptive, 4,858+ non-disruptive (recent) | MDSplus | Data Usage Agreement | 2 weeks (US), 2+ months (foreign) | Free |
| **JET** | JET | 2,309+ disruptions (de Vries survey alone) | JPF/PPF/SAL | EUROfusion affiliation + approval | 1 week (EU), months (non-EU) | Free |
| **DisruptionBench** | C-Mod, DIII-D, EAST | Multi-machine benchmark | Via DisruptionPy | Requires MDSplus access per machine | Per-machine | Free |
| **FRNN code** | DIII-D, JET | Code only | Python/HDF5 | GitHub (code); data needs institutional access | Immediate (code) | Free |
| **DisruptionPy** | C-Mod, DIII-D, EAST, HBT-EP, MAST | Framework (no data) | Python package | PyPI | Immediate | Free |

---

## 7. Recommended Action Plan

### Phase 1: Immediate (This Week)

1. **Download ITPA IDDB from Harvard Dataverse**
   - URL: [doi:10.7910/DVN/NXDX6U](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NXDX6U)
   - This is the fastest path to multi-machine disruption data (9 tokamaks, ~50 parameters)
   - Begin HDC encoding experiments on the ~50 scalar parameters

2. **Install DisruptionPy**
   ```bash
   pip install disruption-py
   ```
   - Clone repo: `git clone https://github.com/MIT-PSFC/disruption-py.git`
   - Review built-in signal retrieval routines for the parameter list

3. **Clone FRNN codebase**
   ```bash
   git clone https://github.com/PPPLDeepLearning/plasma-python.git
   ```
   - Review data loading pipeline for format understanding

### Phase 2: Short-term (1-2 Weeks)

4. **Apply for DIII-D data access**
   - Go to [d3dfusion.org/become-a-user](https://d3dfusion.org/become-a-user/)
   - Contact division leaders to discuss HDC disruption prediction research
   - Complete Data Usage Agreement at [diii-d.gat.com/ssl_form/cyberaccess](https://diii-d.gat.com/ssl_form/cyberaccess/)
   - As a US-based researcher, expect ~2 weeks for access

5. **Read the DisruptionBench paper**
   - [DOI: 10.1007/s10894-025-00495-2](https://link.springer.com/article/10.1007/s10894-025-00495-2)
   - Understand the 9 benchmark tasks (zero-shot, few-shot, many-shot)
   - These define the exact evaluation protocol for our HDC approach

### Phase 3: Medium-term (1-2 Months)

6. **Apply for JET data access**
   - Register at [users.jetdata.eu](https://users.jetdata.eu/)
   - As a non-EUROfusion researcher, the INCO approval will take longer
   - Consider establishing a collaboration with a EUROfusion-affiliated researcher to accelerate access

7. **Contact MIT PSFC Disruptions Group**
   - Email Cristina Rea's group: [disruptions.mit.edu/team](https://disruptions.mit.edu/team/)
   - Discuss potential collaboration on HDC-based disruption prediction
   - They maintain the definitive C-Mod and DIII-D disruption databases

### Phase 4: Long-term (2-3 Months)

8. **Establish cross-machine validation pipeline**
   - Train HDC encoder on DIII-D data
   - Zero-shot test on C-Mod (via DisruptionBench protocol)
   - Zero-shot test on EAST
   - Compare against CCNN baseline (AUC 0.974)

9. **Pursue JET carbon-wall vs ILW comparison**
   - Once JET access is granted, test whether HDC encoding captures the wall-material-dependent disruption signatures
   - This is a unique validation opportunity: same machine, different wall materials

---

## 8. Key References

### Foundational Papers

| Paper | Relevance |
|-------|-----------|
| Rea, C. & Granetz, R.S. (2018). "Disruption prediction investigations using ML tools on DIII-D and Alcator C-Mod." *Plasma Phys. Control. Fusion* 60, 084004. | First systematic ML comparison across two machines; Random Forest baseline |
| Kates-Harbeck, J. et al. (2019). "Predicting disruptive instabilities via deep learning." *Nature* 568, 526-531. | FRNN deep learning; DIII-D-to-JET transfer |
| Montes, K.J. et al. (2019). "Machine learning for disruption warnings on C-Mod, DIII-D, and EAST." *Nuclear Fusion* 59, 096015. | Three-machine ML warnings |
| de Vries, P.C. et al. (2011). "Survey of disruption causes at JET." *Nuclear Fusion* 51, 053018. | Comprehensive JET disruption taxonomy (2,309 disruptions) |
| Granetz, R.S. et al. (2016). "The ITPA disruption database." *Nuclear Fusion* 56, 026013. | IDDB design and content |

### Transfer Learning & Cross-Machine

| Paper | Relevance |
|-------|-----------|
| Zheng, W. et al. (2023). "Disruption prediction for future tokamaks using parameter-based transfer learning." *Commun. Phys.* 6, 181. | J-TEXT to EAST with 20 shots |
| Chayapathy, D. et al. (2024). "Time Series Viewmakers for Robust Disruption Prediction." arXiv:2410.11065. | Domain-invariant representations |
| DisruptionBench (2025). "DisruptionBench and Complementary New Models." *J. Fusion Energy* 44, 26. | Standardized multi-machine benchmark |

### JET-ILW Specific

| Paper | Relevance |
|-------|-----------|
| de Vries, P.C. et al. (2014). "Disruption causes during first operations with JET ITER-like wall." EUROfusion. | Wall-material effect on disruptions |
| Maslov, M. et al. "The EUROfusion JET-ILW global confinement database." | JET-ILW confinement scaling |

---

## Appendix: Data Format Notes

### MDSplus

MDSplus is the standard data system for fusion experiments worldwide. Data is stored in hierarchical "tree" structures. Key points:
- Each "shot" (discharge) has a unique integer ID
- Data is organized in named "nodes" within trees (e.g., `\ip` for plasma current)
- Time-series data is stored as signal objects (value array + time array)
- Python access via `MDSplus` package: `pip install mdsplus`
- Remote access via thin-client connections to institutional servers

### Common Diagnostic Signals for Disruption Prediction

| Signal | Description | Typical MDSplus path |
|--------|-------------|---------------------|
| Ip | Plasma current | `\ip` |
| ne_bar | Line-averaged electron density | `\denv_01` (varies) |
| q95 | Safety factor at 95% flux | `\q95` |
| beta_N | Normalized beta | `\betan` |
| li | Internal inductance | `\li` |
| Wmhd | Stored energy | `\wmhd` |
| n=1 amplitude | Locked mode indicator | `\n1rms` |
| Prad | Radiated power | `\pradtot` |
| dIp/dt | Current derivative (disruption signature) | Derived |
| Greenwald fraction | ne/nGW (density limit) | Derived |

*Note: MDSplus paths vary by machine. DisruptionPy abstracts these differences.*
