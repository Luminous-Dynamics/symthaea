# Superconductivity discovery and REBCO manufacturing: SOTA / industry comparison (21 Nov 2026)

This is a source-pinned benchmark map, not a claim that Symthaea or Symtropy has reproduced these results. Published metrics are preserved with their label authority, evaluation split, application conditions and maturity status.

## 1. Discovery systems: compare the right estimand

### AutoSuperCon — broad experimental parameter data and composition baselines

Manium et al., arXiv:2611.15627 (11 Nov 2026), extracted eight experimentally reported parameters from more than 47,000 arXiv preprints and historical literature. AutoSuperCon contains 20,354 entries for 6,793 superconductors, covering \(T_c\), \(H_c\), \(H_{c1}\), \(H_{c2}\), \(J_c\), coherence length, penetration depth and gap. Its extraction agreement with NIMS SuperCon is reported as \(R^2=0.96\) for 2,439 overlapping materials, with median absolute deviation 0.25 K / 3.5%.

Composition-only TabPFN-3 models report:
- \(T_c\): \(R^2_{\log}=0.83\), median absolute error 1.6 K on 5,843 held-out materials;
- orientation-averaged \(H_{c2}\): \(R^2_{\log}=0.67\) for composition alone and 0.69 after transfer from predicted \(T_c\), on 1,204 materials.

The important result is the family-transfer failure. Under leave-one-family-out evaluation, the cuprate \(T_c\) score falls to \(R^2_{\log}=-10.2\) (versus 0.78 when cuprates are represented during training); the iron-based family also fails strongly. For \(H_{c2}\), the cuprate leave-one-family-out score is -9.0 and iron-based -8.6. These are not small errors: negative \(R^2_{\log}\) means worse than predicting the family mean. The paper explicitly concludes that these composition models do not extrapolate to families absent from training.

**Benchmark implication for Symthaea:** report random/in-family and family-held-out scores separately. Add material-structure and mechanism-aware features, but never claim out-of-family superiority from random row splits. Family-level recall, calibration, abstention and false-negative rates must accompany MAE / \(R^2\).

Source: https://arxiv.org/abs/2611.15627 (v1 dated 11 Nov 2026); detailed results in the PDF at https://borowiecki.dev/pdf/2611.15627.

### BEE-NET — high-throughput electron–phonon discovery pipeline

BEE-NET predicts the Eliashberg spectral function and DFT/Allen–Dynes transition estimates. Its reported 0.87 K MAE is measured against DFT-based Allen–Dynes labels—not experimental \(T_c\). Its CPD model integrates a coarse phonon density of states; the pipeline applies structural / thermodynamic and harmonic-dynamical filters before expensive \(\alpha^2F(\omega)\) calculations.

The reported campaign screened more than 1.3 million candidate structures to 741 stable candidates with DFT-computed \(T_c>5\) K, including 69 predicted at \(T_c\ge20\) K. It performed 866 expensive \(\alpha^2F(\omega)\) calculations, reports an overall final precision of 86%, and reports experimental synthesis / superconducting confirmation for two materials, Be\(_2\)HfNb\(_2\) and Be\(_2\)Hf\(_2\)Nb. Different BEE-NET variants and reports quote different TNRs (e.g. 99.4% in the paper abstract versus 0.991 for CPD in its repository table); these must be kept attached to the specific variant and test split.

**Benchmark implication:** the valuable comparator is not just the 0.87 K computed-label MAE. It is the staged funnel, cost per high-fidelity calculation, recall at each filter, final precision, and experimental validation. The marginal-phonon false-negative child #13134 directly challenges the hard harmonic filter. Do not relax harmonic stability claims; route marginal/uncertain cases to further convergence, distortion or justified finite-temperature evaluation.

Sources:
- Paper: https://www.nature.com/articles/s41524-026-01964-8
- Code / model variants: https://github.com/henniggroup/BEE-NET

### ORNL physics-enhanced unconventional-superconductor programme

ORNL's project, active from September 2026 to May 2027, plans to combine high-accuracy first-principles simulation, machine learning, automated workflows, synthesis, neutron scattering and experimental feedback. The public project description proposes screening roughly 1.5 million structures to 200–300 candidates, evaluating about 15,000 candidates and ranking a shortlist of about 50 outside conventional cuprate/nickelate spaces. The work focuses on competing magnetic states and strong electron–phonon coupling.

This is an **active project plan**, not a completed performance result. It is nevertheless an important strategic comparator: the field is moving toward physics-informed agentic active learning with neutron and synthesis feedback.

**Our differentiation should be evidence and decision quality:** preserve the full path from source and intended structure to realized phase, uncertainty, specimen, batch, measurement and application. Benchmark against ORNL's planned scale where practical, but do not claim a resource or throughput advantage before matched prospective trials.

Source: https://impact.ornl.gov/en/projects/physics-enhanced-ai-framework-for-discovering-new-classes-of-unco/

## 2. REBCO manufacturing and metrology: industry anchors

### SuperPower — reel-to-reel process and longitudinal quality

SuperPower describes reel-to-reel buffer deposition, MOCVD REBCO growth, Zr addition that creates BZO columnar pinning structures, silver sealing and oxygen annealing, plus post-processing such as slitting and copper stabilization. It reports full-length \(I_c\) characterization using direct transport samples at 3 m intervals or a calibrated non-contact magnetization system (“TapeStar”).

Its public 2G HTS specification lists 2–12 mm widths and piece lengths typically 200–500 m, with some specifications above 900 m. For 4 mm tape at 77 K self-field using a 1 \(\mu\)V/cm transport criterion, it lists 120–160 A for AP tape and 80–120 A for HM tape. These are product-specific manufacturer values, not universal thresholds.

**Benchmark implication:** distinguish actual footprint from sample spacing; direct transport from calibrated proxy; reel-average from lower-tail and maximum contiguous weak-section risk; and inline proxy predictions from independent transport validation.

Sources:
- Process and measurement: https://www.superpower-inc.com/Technology.aspx
- Product specification: https://www.superpower-inc.com/specification.aspx
- Quality systems: https://www.superpower-inc.com/Quality.aspx

### Fujikura — artificial pinning and in-field performance

Fujikura's public FESC-SCH04 product entry specifies a 4 mm artificial-pin tape with 130 A at 77 K self-field and 640 A at 20 K / 5 T. The page explicitly says the 20 K / 5 T value is a reference value and does not guarantee delivered performance. It is therefore a useful product/condition comparator, not a generic REBCO pass line.

Source: https://www.fujikura.co.jp/en/products/superconductors/superconductingwire-reel/superconductingwire-reel.html

### Independent multi-supplier and inter-laboratory comparisons

Himanshu et al., IEEE Transactions on Applied Superconductivity 36(5), 2026, DOI 10.1109/TASC.2026.3672426, compares electrical performance at 77 K and thermal/mechanical behaviour at room temperature across four industrial REBCO manufacturers, and compares two production-year batches of THEVA. This demonstrates why batch and supplier comparisons must be repeated under common measurement methods.

Babouche et al., Superconductor Science and Technology 39 (2026) 085021, DOI 10.1088/1361-6668/ae940e, characterizes three 4 mm commercial tapes from SST, SuperPower and Faraday Factory Japan/SuperOx. It measures full-width tape \(I_c(B,\theta,T)\) at UNIGE from 4.2–40 K and up to 19 T across five orientations, and same-spool samples at HFLSM using microbridges from 5–55 K and up to 24 T with angular scans. The paper discusses differences between labs and specimen formats, including spatial inhomogeneity and alignment. Underlying measurement data are available from the authors on reasonable request; they are not presumed openly downloadable.

Sources:
- Four-supplier / batch comparison: https://doi.org/10.1109/TASC.2026.3672426
- Cross-lab anisotropy study: https://doi.org/10.1088/1361-6668/ae940e

### Manufacturing maturity and cost

A 2026 review of the superconductivity translation gap estimates routine piece lengths and yield above 20 km / 98% for NbTi, above 10 km / 95% for Nb\(_3\)Sn, and 0.3–2 km / 80–90% for REBCO, with REBCO joints still an engineering constraint. Treat these as the review's estimates, not a universal audited industry census; vendor and product definitions differ.

The 2026 REBCO cable roadmap defines manufacturing cost/performance in €/kA·m under application working conditions, with production throughput, length, width and \(J_c\) all contributing. The roadmap also discusses wider templates, double-sided coating and process throughput as potential scale levers.

Sources:
- Translation-gap review: https://www.sciencedirect.com/science/article/pii/S2949822826015558
- Cable manufacturing roadmap: https://snf.ieeecsc.org/files/ieeecsc/2026-01/sust25cableroadmapacman.pdf

## 3. Proposed Luminous benchmark ladder

These are proposed targets, not achieved results. Freeze cohort definitions, measurement conditions, sample sizes and analysis code before opening held-out data.

| Layer | Baseline to beat | Primary benchmark |
|---|---|---|
| Data quality | AutoSuperCon / NIMS agreement and source provenance | Condition-aware extraction agreement; false equilibrium/transient/theory labels; duplicate lineage; uncertainty retained |
| Discovery interpolation | AutoSuperCon in-family \(T_c\) and \(H_{c2}\) models | Matched random/in-family metrics plus calibration and family-stratified recall |
| New-family transfer | AutoSuperCon leave-one-family-out | Better than family-mean and composition-only baselines on held-out complete families; explicit abstention |
| Electron–phonon screening | BEE-NET 0.87 K MAE on Allen–Dynes labels | Matched test split and labels; cost per expensive calculation; precision/recall after each funnel stage |
| Soft-mode recall | Any-imaginary-mode hard rejection | At least 50% lower marginal-mode positive-control FN at matched compute, with zero unsupported stability claims |
| Synthesis realization | Predicted candidate vs synthesized phase | Prospective phase realization, batch repeatability, measured property and independent confirmation |
| REBCO tape | SuperPower/Fujikura product specifications | \(I_c(B,\theta,T)\), \(J_e\), n-value, strain, AC loss, joints, lower-tail and contiguous-defect metrics at identical conditions |
| Long-length metrology | Transport samples + calibrated TapeStar | Proxy bias/interval coverage, missed weak-section probability, localization, drift, blind zones and inspection cost |
| Industrial readiness | NbTi / Nb\(_3\)Sn maturity | Cost per accepted kA·m at target field/temperature/angle, accepted yield, joint resistance, reliability and qualification cost |

### Numerical targets worth preregistering

1. **New-family discovery:** improve family-held-out recall and calibrated utility against AutoSuperCon's negative leave-one-family-out baseline; do not make a fixed \(R^2\) target the only success criterion. A useful candidate is one whose property interval, uncertainty and synthesis pathway are prospectively validated.
2. **Expensive calculations:** at matched compute budget, increase experimentally confirmed useful candidates per CPU/GPU/DFT hour; keep false-discovery and family-stratified recall explicit. Do not compare different candidate prevalences or thresholds without reporting precision–recall curves.
3. **Soft phonons:** ≥50% relative reduction in false negatives for a preregistered marginal-phonon cohort, at equal compute budget; unsupported stability claims = zero.
4. **Proxy metrology:** nominal 90% intervals should reach 85–95% empirical coverage on sufficiently populated held-out reels/labs; set false-negative ceiling from the actual weakest-section length and magnet/cable safety case.
5. **Adaptive inspection:** compare uniform, random, uncertainty-only, one-step VOI and finite-horizon oracle at identical measurement cost and method mix; target lower expected false-accept loss and lower missed contiguous-defect risk.
6. **Industrial conductor:** target lower cost per accepted kA·m at matched \(T,B,\theta\), width, geometry, electric-field criterion, strain and reliability—not merely higher self-field \(I_c\).

## 4. Where we can genuinely outperform

The opportunity is not to out-scale ORNL or reproduce BEE-NET's single MAE number immediately. It is to make the whole scientific and industrial decision chain auditable and better calibrated:

- **Beyond composition-only models:** family-held-out, mechanism-aware representation plus pairing/coherence response separation and calibrated abstention.
- **Beyond hard phonon filtering:** uncertainty- and convergence-aware marginal triage; retain candidates for further work without falsely labelling them stable.
- **Beyond one-shot candidate ranking:** value-of-information allocation of expensive calculations and experiments under fixed budgets.
- **Beyond average tape \(I_c\):** spatially resolved lower-tail and contiguous-weak-section qualification, separating direct transport and calibrated proxy evidence.
- **Beyond nominal material identity:** realized phase, specimen, batch, campaign, laboratory and instrument lineage, with independent cross-lab and batch transfer.
- **Beyond material-only performance:** cost, yield, joint resistance, strain, AC loss, thermal margin and reliability at the intended magnet/cable operating envelope.

The immediate executable work in this repository covers only pieces of that programme: discovery split/evidence auditing, soft-mode synthetic disposition, source-linked positive-control metadata, and a separate budget-aware inspection/condition-profile prototype in Symtropy. No Luminous model has yet reproduced AutoSuperCon or BEE-NET; no physical new-family candidate or REBCO yield improvement is claimed.

## 5. Research governance and data rights

- Treat source-reported values, digitized figures, inferred values, model predictions, synthetic fixtures and physical measurements as different evidence classes.
- Use complete families, source papers, structures, batches, campaigns and reels as holdout units where appropriate; no random per-row split can support a family-generalization or manufacturing-transfer claim.
- Do not copy restricted author/manufacturer raw tables. Pin source identifiers, access terms, data version, hashes and permitted derived artifacts.
- Do not promote proxy estimates to direct transport measurements, harmonic results to anharmonic stability, or a synthetic policy outcome to a physical manufacturing result.
