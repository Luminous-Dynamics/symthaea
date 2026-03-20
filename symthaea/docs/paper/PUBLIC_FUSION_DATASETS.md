# Public Tokamak / Fusion / Plasma Disruption Datasets

Comprehensive catalog of publicly downloadable datasets relevant to tokamak disruption prediction, plasma physics, and fusion energy research. Compiled 2026-03-19.

**Legend**: Accessibility is rated as OPEN (no credentials, direct download), TOOL (requires open-source tool/API but no credentials), GATED (requires free registration or institutional access), or CODE-ONLY (code available but data requires separate access).

---

## 1. Fully Open Datasets (Direct Download)

### 1.1 FAIR-MAST (MAST Tokamak, UKAEA)

| Field | Details |
|-------|---------|
| **URL** | https://mastapp.site/ |
| **GitHub** | https://github.com/ukaea/fair-mast |
| **Machine** | MAST (Mega Ampere Spherical Tokamak), UKAEA, UK |
| **Format** | Parquet (shot metadata: 11,573 rows x 189 columns), Zarr (signal data via S3) |
| **Size** | ~TB-scale (full diagnostic archive) |
| **Diagnostics** | 39 signals across heterogeneous modalities (magnetics, Thomson scattering, interferometry, spectroscopy, etc.) |
| **Disruption labels** | Not explicitly labeled for disruption, but shot metadata includes plasma current quench indicators |
| **License** | CC BY-SA 4.0 |
| **API** | GraphQL endpoint at `/graphql`; S3-backed object storage (no auth required for public data) |
| **Citation** | FAIR-MAST: A fusion device data management system, SoftwareX (2024) |
| **Notes** | The only openly available dataset of real tokamak diagnostics with no authentication barriers. Foundation for TokaMark benchmark (14 tasks) and TokaMind foundation model. |

### 1.2 LHD (Large Helical Device, NIFS Japan) on AWS

| Field | Details |
|-------|---------|
| **URL** | https://registry.opendata.aws/nifs-lhd/ |
| **Data browser** | https://nifs-lhd.s3.amazonaws.com/README.nifs-lhd.html |
| **Machine** | LHD (Large Helical Device), NIFS, Toki, Japan (stellarator/heliotron) |
| **Format** | `.dat` and `.zip` archived files |
| **Size** | ~2 petabytes total (all diagnostics since March 1998) |
| **Diagnostics** | 40+ million data items covering all LHD plasma diagnostics |
| **Disruption labels** | No explicit disruption labels (stellarators don't disrupt like tokamaks, but instability events exist) |
| **License** | Open under AWS Open Data Sponsorship Program terms of use |
| **Citation** | NIFS LHD experiment team |
| **Notes** | Largest open fusion dataset in existence. 25 years of experimental data. Stellarator geometry (not tokamak), but relevant for plasma instability and confinement studies. Can be accessed directly from AWS cloud for high-performance analysis. |

### 1.3 ITPA Disruption Database (Multi-Machine) on Harvard Dataverse

| Field | Details |
|-------|---------|
| **URL** | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NXDX6U |
| **Machine** | Multi-machine: Alcator C-Mod, ASDEX Upgrade, DIII-D, JET, JT-60U, MAST, NSTX, TCV (9 devices) |
| **Format** | Tabular (Harvard Dataverse format) |
| **Size** | 3,875 discharges |
| **Diagnostics** | Disruption-relevant scalar parameters: plasma current, stored energy, density, q95, etc. |
| **Disruption labels** | Yes -- this is a disruption-specific database |
| **License** | Harvard Dataverse terms |
| **Citation** | Granetz et al., "The ITPA disruption database," Nuclear Fusion (2016) |
| **Notes** | Public "frozen" version of the active IDDB (which is on restricted MDSplus at iddb.gat.com:9000). Primary data is scalar (not time-series). Key resource for cross-machine disruption statistics and scaling laws. |

### 1.4 GOLEM Tokamak (Czech Technical University)

| Field | Details |
|-------|---------|
| **URL** | http://golem.fjfi.cvut.cz/utils/data/ |
| **Machine** | GOLEM tokamak, CTU Prague, Czech Republic |
| **Format** | ASCII, CSV, NumPy `.npz`, Excel |
| **Size** | 20,000+ shots (small tokamak, modest per-shot data) |
| **Diagnostics** | Plasma current, loop voltage, Mirnov coils, bolometry, visible light, H-alpha |
| **Disruption labels** | Not formally labeled, but disruptions are identifiable from plasma current traces |
| **License** | Open (educational/research use) |
| **API** | Direct URL: `http://golem.fjfi.cvut.cz/utils/data/[shot]/[diagnostic].csv` |
| **Citation** | GOLEM tokamak team, CTU Prague |
| **Notes** | Self-described as "the only fully open-source tokamak." Small device (R=0.4m, Ip~5kA) but excellent for algorithm development and educational purposes. Python library `pygolem_lite` available. |

### 1.5 Semi-Supervised ML Detector for Tokamak Physics Events (DIII-D)

| Field | Details |
|-------|---------|
| **URL** | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2LGWJR |
| **Machine** | DIII-D |
| **Format** | Tabular (Harvard Dataverse) |
| **Diagnostics** | Labeled physics events: H-L back transitions, locked modes, core radiative collapses |
| **Disruption labels** | Yes -- event labels including disruption-precursor events |
| **License** | Harvard Dataverse terms |
| **Citation** | DOI: 10.7910/DVN/2LGWJR |
| **Notes** | Hundreds of DIII-D discharges with manually identified physics events. Excellent for anomaly detection benchmarking. |

### 1.6 Turbulent Plasma Dynamics via Deep Learning (Synthetic + Alcator C-Mod)

| Field | Details |
|-------|---------|
| **URL** | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNMLN2 |
| **Machine** | Synthetic plasma (two-fluid model) validated against Alcator C-Mod |
| **Format** | Tabular (Harvard Dataverse) |
| **Diagnostics** | Electron density, temperature, turbulent fields, electrostatic potential |
| **Disruption labels** | No (turbulence focus, not disruption) |
| **License** | Harvard Dataverse terms |
| **Citation** | Mathews et al., Phys. Rev. E 104, 025205 (2021) |
| **Notes** | Physics-informed neural network training data. Useful for plasma turbulence and edge physics studies. |

### 1.7 Tokamak Disruption Database (Chinese, SciDB)

| Field | Details |
|-------|---------|
| **URL** | https://www.scidb.cn/en/detail?dataSetId=28c19590bc87416b8a4a63d35b1a5908 |
| **Machine** | Chinese tokamaks (likely EAST/J-TEXT/HL-2A) |
| **Format** | Unknown (check SciDB page) |
| **Disruption labels** | Yes |
| **License** | SciDB/CAS terms |
| **Citation** | See SciDB entry |
| **Notes** | Hosted on Chinese Academy of Sciences' ScienceDB platform. May require browsing the Chinese-language interface. |

### 1.8 High Aspect Ratio Fusion Device Design Dataset (Stellarator Simulation)

| Field | Details |
|-------|---------|
| **URL** | https://zenodo.org/records/13623959 |
| **GitHub** | https://github.com/pedrocurvo/MLStellaratorDesign |
| **Machine** | Simulated stellarator configurations |
| **Format** | Data files (check Zenodo) |
| **Disruption labels** | No (design optimization, not disruption) |
| **License** | Open (Zenodo) |
| **Citation** | Curvo et al., J. Plasma Physics (2025) |
| **Notes** | Inverse design dataset: input parameters mapped to confinement properties. Relevant for stellarator optimization via ML. |

### 1.9 Disruptions & Tokamak Economics (Modeling Data)

| Field | Details |
|-------|---------|
| **URL** | https://github.com/andrew-maris/disruptions-tokamak-economics |
| **Zenodo DOI** | 10.5281/zenodo.8044908 |
| **Machine** | Modeling (not from a specific tokamak) |
| **Format** | Jupyter notebooks, Python data |
| **Disruption labels** | N/A (economic modeling) |
| **License** | Open source |
| **Citation** | Maris et al., Fusion Science and Technology 80(5), 2024 |
| **Notes** | LCOE model quantifying economic impact of disruptions on tokamak power plants. Useful for contextualizing disruption costs. |

### 1.10 DOE Data Explorer: Disruption Prediction Dataset (DIII-D ECEi)

| Field | Details |
|-------|---------|
| **URL** | https://www.osti.gov/dataexplorer/biblio/dataset/1661171 |
| **DOI** | 10.11578/1661171 |
| **Machine** | DIII-D |
| **Format** | Associated with deep CNN paper |
| **Diagnostics** | Electron Cyclotron Emission imaging (ECEi) -- raw, high temporal resolution |
| **Disruption labels** | Yes |
| **License** | DOE public data |
| **Citation** | Churchill, R.M. et al. (2020) |
| **Notes** | Raw ECEi data used for disruption prediction with dilated CNNs. F1-score ~91% on individual time-slices. |

### 1.11 DOE Data Explorer: Additional Tokamak Datasets

Multiple public datasets available at https://www.osti.gov/dataexplorer/ including:
- **Local Helicity Injection** (OSTI 1419641): Tokamak plasma initiation data
- **SPARC predictions** (OSTI 1881499): Core plasma performance predictions
- **Gas Puff Imaging** (OSTI 1562076): Edge plasma turbulence diagnostics
- **TRANSP optimization** (OSTI 1562098): Tokamak scenario development

---

## 2. Tool-Accessible Datasets (Open Tools, No Credentials for Code)

### 2.1 DisruptionBench (MIT PSFC)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/MIT-PSFC/DisruptionBench |
| **Machine** | Alcator C-Mod, DIII-D, EAST (~30,000 discharges total) |
| **Format** | ML-ready time series |
| **Diagnostics** | 0D plasma parameters at multiple sampling frequencies |
| **Disruption labels** | Yes -- binary disruption/non-disruption labels |
| **License** | Open source |
| **Citation** | Spangher et al., J. Fusion Energy 44:26 (2025) |
| **Notes** | First standardized, model-agnostic, machine-agnostic benchmark for disruption prediction. Includes zero-shot, few-shot, and many-shot evaluation protocols. The benchmark data itself may require DisruptionPy to regenerate from MDSplus, but test set compositions are fixed and recorded in the repo. **Check repo for pre-built data files.** |

### 2.2 DisruptionPy (MIT PSFC)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/MIT-PSFC/disruption-py |
| **PyPI** | `pip install disruption-py` |
| **Zenodo** | https://zenodo.org/records/18612738 |
| **Machine** | Alcator C-Mod, DIII-D, EAST |
| **Format** | Generates CSV/HDF5/Parquet from MDSplus queries |
| **Diagnostics** | Configurable: any signal available on the MDSplus servers |
| **Disruption labels** | Yes (built-in disruption time metadata) |
| **License** | Open source |
| **Citation** | MIT PSFC Disruption Studies Group |
| **Notes** | Framework for building ML-ready disruption datasets. The tool is fully open-source, but **accessing the underlying MDSplus servers may require institutional authorization** for DIII-D and EAST. C-Mod data may be more accessible. DOE FES funded (2024-2026). |

### 2.3 VEST Tokamak Data (VAFT Framework)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/VEST-Tokamak/vaft |
| **Install** | `pip install vaft` |
| **Machine** | VEST (Versatile Experiment Spherical Torus), Seoul National University, Korea |
| **Format** | HDF5 via HSDS, ODS via OMAS |
| **Diagnostics** | Standard tokamak diagnostics (R=0.4m, B=0.1T spherical torus) |
| **Disruption labels** | Not specifically |
| **License** | Open |
| **Notes** | Free "reader" account available (endpoint: http://147.46.36.244:5101, username: reader, password: test). Small spherical torus data. |

### 2.4 TokaMark Benchmark (MAST via FAIR-MAST)

| Field | Details |
|-------|---------|
| **Paper** | https://arxiv.org/abs/2602.10132 |
| **Data source** | FAIR-MAST (see 1.1 above) |
| **Machine** | MAST |
| **Format** | Standardized subset of FAIR-MAST with harmonized metadata |
| **Tasks** | 14 downstream tasks in 4 groups (representation learning, temporal reasoning, robustness, generalization) |
| **Disruption labels** | Includes disruption-related classification tasks |
| **License** | CC BY-SA 4.0 (inherits from FAIR-MAST) |
| **Citation** | TokaMark (2026) |
| **Notes** | Code and data to be fully open-sourced. Uses 39 signals from FAIR-MAST. First comprehensive ML benchmark on real fusion data. |

---

## 3. Simulation Codes with Built-In Data / Scenarios

### 3.1 DREAM (Disruption Runaway Electron Analysis Model)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/chalmersplasmatheory/DREAM |
| **Docs** | https://ft.nephy.chalmers.se/dream/ |
| **Type** | Physics simulation (not experimental data) |
| **Capabilities** | Tokamak disruption physics: thermal quench, current quench, runaway electron dynamics |
| **Format** | HDF5 output, Python/C++ |
| **License** | Open source |
| **Citation** | Hoppe et al., Comput. Phys. Commun. 268, 108098 (2021) |
| **Notes** | Can generate synthetic disruption time-series data with controllable physics parameters. Excellent for creating labeled training data. Related: STREAM (startup runaway electron code). |

### 3.2 TORAX (Google DeepMind)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/google-deepmind/torax |
| **Type** | Differentiable tokamak transport simulator in JAX |
| **Capabilities** | Ion/electron heat transport, particle transport, current diffusion |
| **Format** | JAX arrays, Python |
| **License** | Apache 2.0 |
| **Citation** | DeepMind (2024) |
| **Notes** | Not disruption-specific, but can simulate plasma evolution toward instability. Differentiable -- enables gradient-based optimization and neural surrogate training. Verified against RAPTOR code. |

### 3.3 KSTAR Tokamak Simulator (Neural Network)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/jaem-seo/KSTAR_tokamak_simulator |
| **Type** | LSTM-based neural network simulator trained on KSTAR data |
| **Machine** | KSTAR (Korea) |
| **Format** | Python, pre-trained model weights included |
| **License** | Open source |
| **Citation** | Seo et al. |
| **Notes** | Neural surrogate trained on real KSTAR discharges. Can generate synthetic plasma evolution trajectories. Interactive GUI with slider controls. Related: `AI_tokamak_control` (RL-based control) and `KSTAR-data-driven-tokamak-simulator` by ZINZINBIN. |

### 3.4 OpenMHD

| Field | Details |
|-------|---------|
| **URL** | https://sci.nao.ac.jp/MEMBER/zenitani/openmhd-e.html |
| **Type** | MHD simulation code |
| **Capabilities** | Resistive MHD, ideal MHD -- can simulate tearing instabilities and disruption precursors |
| **License** | Open source |
| **Notes** | General MHD code, not tokamak-specific, but can be configured for tokamak-relevant geometries. |

---

## 4. Code Repositories with Data Access Patterns (Data Requires Separate Access)

### 4.1 FRNN / plasma-python (PPPL)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/PPPLDeepLearning/plasma-python |
| **Machine** | DIII-D, JET |
| **Model** | LSTM-based disruption prediction (Fusion Recurrent Neural Network) |
| **Data location** | `/tigress/FRNN` on Princeton clusters; ALCF Theta paths |
| **Disruption labels** | Yes |
| **Citation** | Kates-Harbeck et al., Nature 568, 526-531 (2019) |
| **Notes** | Code is fully open-source. Data is on institutional clusters and not directly downloadable. The Nature 2019 paper is the seminal deep learning disruption prediction work. |

### 4.2 KSTAR Disruption Prediction (Multimodal Deep Learning)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/ZINZINBIN/Disruption-Prediciton-based-on-Multimodal-Deep-Learning |
| **Machine** | KSTAR |
| **Model** | Multimodal DL using IVIS video + 0D parameters |
| **Data** | KSTAR IVIS video data + 0D parameters (requires KFE access) |
| **Disruption labels** | Yes |
| **Citation** | Kim et al., Fusion Engineering and Design (2024) |
| **Notes** | First use of video data for disruption prediction. Code is open. Data requires KSTAR institutional access. Related: `Tokamak-Plasma-Operation-Control-based-on-RL` (RL control). |

### 4.3 DisruptCNN (DIII-D ECEi)

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/rmchurch/disruptcnn |
| **Machine** | DIII-D |
| **Model** | Temporal Convolutional Network for disruption prediction |
| **Data** | ECEi diagnostic data in HDF5 format (not included in repo -- see Issue #6) |
| **Disruption labels** | Yes |
| **Citation** | Churchill et al. |
| **Notes** | Code is open but H5 data files are not included in the repository. |

### 4.4 ADITYA Tokamak Disruption Prediction

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/amanbasu/plasma-disruption |
| **Machine** | ADITYA (India) |
| **Model** | LSTM for disruption prediction |
| **Data** | ADITYA diagnostic data (requires IPR access) |
| **Disruption labels** | Yes (12ms advance prediction) |
| **Notes** | Code open. Data from Institute for Plasma Research, India. |

### 4.5 JET/TCABR Disruption Prediction (ML Techniques)

| Field | Details |
|-------|---------|
| **Paper** | https://arxiv.org/abs/2005.05139 |
| **Code** | https://github.com/JoostCroonen/ML_Tokamak_Disruption_Prediction |
| **Machine** | JET (178 TCABR discharges also referenced) |
| **Model** | SVM, Random Forest, Gradient-Boosted Trees, LSTM |
| **Data** | JET data (requires EUROfusion/CCFE access) |
| **Disruption labels** | Yes |
| **Notes** | Balanced dataset of 89 disruptive + 89 normal discharges from TCABR. Code open, data access varies by machine. |

---

## 5. Platforms and Infrastructure

### 5.1 Fusion Data Platform (FDP)

| Field | Details |
|-------|---------|
| **URL** | https://ga-fdp.github.io/ |
| **GitHub** | https://github.com/Fusion-Data-Platform/fdp |
| **Docs** | https://fdp.readthedocs.io/ |
| **Machines** | DIII-D, MAST (beta); expanding |
| **Notes** | Comprehensive open-access infrastructure for fusion data. Beta release provides multi-machine access via Open Science Data Federation (OSDF). Developed by General Atomics + UCSD. Integrates TokSearch query engine. |

### 5.2 TokSearch (General Atomics)

| Field | Details |
|-------|---------|
| **Paper** | https://doi.org/10.1016/j.fusengdes.2018.02.027 |
| **Notes** | Recently open-sourced search engine for fusion experimental data. Parallelized queries over archived shot data. Being integrated with FDP. Python and Matlab APIs. |

### 5.3 Data Fusion Labeler (dFL)

| Field | Details |
|-------|---------|
| **URL** | https://dfl.sophelio.io/ |
| **Paper** | https://arxiv.org/abs/2511.09725 |
| **Notes** | Desktop tool for harmonizing, labeling, and modeling multi-sensor fusion data. Reduces time-to-analysis >50x. Free trial available. Not a dataset itself, but critical tool for working with fusion data. |

### 5.4 INPTDAT (Plasma Technology Data Platform)

| Field | Details |
|-------|---------|
| **URL** | https://www.inptdat.de/ |
| **GitHub** | https://github.com/plasma-mds/inptdat-platform |
| **Notes** | Interdisciplinary platform for low-temperature plasma data. Metadata schema: Plasma-MDS. Mostly industrial/medical plasma, not tokamak disruptions, but relevant for plasma diagnostics ML. |

### 5.5 Plasma and Fusion Cloud (China)

| Field | Details |
|-------|---------|
| **Paper** | https://doi.org/10.1016/j.fusengdes.2025.114904 |
| **Machines** | EAST, HL-2A, HL-3, J-TEXT |
| **Notes** | Chinese platform for fusion data ecosystem. Under development toward open science. Not yet fully public. |

### 5.6 IAEA CollisionDB

| Field | Details |
|-------|---------|
| **URL** | https://db-amdis.iaea.org/collisiondb/ |
| **Notes** | Open-source database of plasma collision processes. Not disruption data, but useful for plasma modeling and synthetic diagnostics. |

---

## 6. Curated Lists and Meta-Resources

### 6.1 Awesome ML in Plasma Physics

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/kharitonov-ivan/awesome-ML-in-plasma-physics |
| **Notes** | Curated list of ML tools, papers, and datasets for plasma physics. Regularly updated. Includes links to disruption prediction repos, simulators, and data tools. |

### 6.2 Fusion Open Source Projects

| Field | Details |
|-------|---------|
| **GitHub** | https://github.com/kripnerl/fusion-open-source |
| **Notes** | Curated list of open-source fusion projects including PlasmaPy, OMAS, ToFu, Aurora, Bluemira, FreeGS, and more. |

### 6.3 MIT PSFC Disruptions Group

| Field | Details |
|-------|---------|
| **URL** | https://disruptions.mit.edu/ |
| **Dataverse** | https://dataverse.harvard.edu/dataverse/MIT-PSFC |
| **Notes** | Central hub for MIT's disruption research. Links to DisruptionPy, DisruptionBench, and Harvard Dataverse datasets. |

---

## 7. Summary: Best Bets for Immediate Use

Ranked by accessibility and relevance to disruption prediction:

| Priority | Dataset | Machines | Disruption Labels | Access |
|----------|---------|----------|-------------------|--------|
| 1 | **FAIR-MAST** | MAST | Indirect | OPEN -- no auth, S3+GraphQL |
| 2 | **ITPA Disruption DB** (Harvard) | 9 machines | Yes | OPEN -- Harvard Dataverse |
| 3 | **GOLEM** | GOLEM | Identifiable | OPEN -- direct CSV/NPZ URLs |
| 4 | **LHD on AWS** | LHD (stellarator) | No | OPEN -- AWS S3 |
| 5 | **Semi-supervised events** (Harvard) | DIII-D | Yes (events) | OPEN -- Harvard Dataverse |
| 6 | **DOE ECEi dataset** | DIII-D | Yes | OPEN -- DOE Data Explorer |
| 7 | **DisruptionBench** | C-Mod, DIII-D, EAST | Yes | TOOL -- check repo for pre-built data |
| 8 | **DREAM** | Synthetic | Configurable | OPEN -- simulation code |
| 9 | **TORAX** | Synthetic | Configurable | OPEN -- simulation code |
| 10 | **KSTAR Simulator** | KSTAR (neural) | Configurable | OPEN -- pre-trained weights |
| 11 | **SciDB Disruption DB** | Chinese tokamaks | Yes | OPEN -- SciDB platform |
| 12 | **VEST/VAFT** | VEST | No | TOOL -- free reader account |

---

## 8. Machines Without Open Data (Institutional Access Required)

These major machines appear frequently in disruption research but their data requires institutional collaboration:

| Machine | Location | Data Access |
|---------|----------|-------------|
| **DIII-D** | General Atomics, San Diego, USA | MDSplus via institutional access; some data on DOE Data Explorer |
| **JET** | Culham, UK (decommissioned 2023) | EUROfusion/CCFE collaboration |
| **EAST** | Hefei, China | Institutional; some via SciDB |
| **KSTAR** | Daejeon, Korea | KFE collaboration |
| **ASDEX Upgrade** | Garching, Germany | IPP collaboration |
| **HL-3** | Chengdu, China | SWIP collaboration |
| **J-TEXT** | Wuhan, China | Huazhong University collaboration |
| **ADITYA/ADITYA-U** | Gujarat, India | IPR collaboration |
| **TCV** | Lausanne, Switzerland | SPC/EPFL collaboration |
| **Wendelstein 7-X** | Greifswald, Germany | IPP Greifswald collaboration |
| **NSTX-U** | Princeton, USA | PPPL collaboration |

---

## 9. Key Papers and Their Data Availability

| Paper | Year | Data Status |
|-------|------|-------------|
| Kates-Harbeck et al., Nature 568:526 | 2019 | Code: github.com/PPPLDeepLearning/plasma-python. Data on PPPL clusters only. |
| Seo et al., Nature 626:746 | 2024 | Tearing instability avoidance on DIII-D. No public data release found. |
| Vega et al., Nat. Commun. 15:2550 | 2024 | JET disruption prediction. No public data release found. |
| Spangher et al., J. Fusion Energy 44:26 | 2025 | DisruptionBench: github.com/MIT-PSFC/DisruptionBench |
| Seo et al., Nat. Phys. 18:741 | 2022 | AI disruption prediction survey. References FRNN data. |

---

## 10. Emerging / Forthcoming Datasets (2026)

| Dataset | Status | Expected |
|---------|--------|----------|
| **TokaMark benchmark** | Paper published Feb 2026 | Code + data "upon acceptance" |
| **TokaMind weights** | Paper published Feb 2026 | Weights "to be released" |
| **PanoMHD** | Paper March 2026 | Unknown if data will be released |
| **FDP Beta** | In development | Multi-machine access via OSDF |
| **Plasma and Fusion Cloud** (China) | Under development | Aims for open science |

---

*Last updated: 2026-03-19*
