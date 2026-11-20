# NEOPIC particle-measure field-operator benchmark v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-NEOPIC-REPRO-001 #12952](https://github.com/Luminous-Dynamics/symthaea/issues/12952)  
**Status:** source review and benchmark protocol; no model/data reproduction performed

## 1. Primary source and artifact inventory

S. Muralikrishnan, C. M. John, J. Kelling, M. Bussmann, R. Speck & P. Gibbon, “NEOPIC: A Neural Operator Framework for Particle-based Kinetic Plasma Simulations,” arXiv:2610.40528, submitted 29 October 2026.

The paper's core contribution is a nonuniform Fourier neural operator (NUFNO) that acts on the empirical measure of the particles and evaluates the learned electric field at particle positions. Unlike a deposited-grid operator, no charge deposition onto a mesh is used for the learned operator. The paper analyses representation equivalence, sampling / learning / time-stepping error, noise-induced regularisation and an O(N_p) evaluation cost under its assumptions.

Source links:
- Paper: https://arxiv.org/abs/2610.40528
- Implementation/configurations (as stated by paper): https://github.com/srikrrish/operator_learning
- Paper result tag: neopic-arxiv-v1 → commit `2cf6a15e0ddb6c408a4ef6d162c31faf1deb9431` (annotated tag object `8e872e5fa520843a9c9db4a68d43036a66355c19`)
- NUFFT cost tag: neopic-arxiv-v1-nufft → commit `1222226acab272b8d78ba541cb67c1e877669785` (annotated tag object `0230db29af90d143ccfe8dfd2e0eb28496139ccb`)
- Paper scripts: https://github.com/NEOPIC-framework/neopic-paper-scripts
- Paper scripts tag: neopic-arxiv-v1 → commit `2ee358abed5c782213d6d35c1194568da26ed8f1` (annotated tag object `dea39090f5ad1c9efd8c3435ed0d19ecb058dec4`)
- Separate NEOPIC repository: https://github.com/srikrrish/NEOPIC

Observed licensing/access facts:
- The default-branch LICENSE of srikrrish/operator_learning is MIT (copyright Chelsea Maria John, 2025).
- The default-branch LICENSE of NEOPIC-framework/neopic-paper-scripts is MIT (copyright Sriram Muralikrishnan, 2026).
- The separate srikrrish/NEOPIC repository's default-branch LICENSE is BSD 3-Clause (copyright Sriramkrishnan Muralikrishnan, 2025).
- The paper says implementation/configurations are in operator_learning and that training datasets, trained models and reference HDF5 outputs are available from the corresponding author on reasonable request, not included in Git.

**Admission caveat:** the three named annotated tags are now resolved to immutable target commit SHAs. This pins source identity but does not establish the licences of every tagged snapshot, submodule, dependency, dataset, trained model or reference output. The HDF5 datasets/checkpoints and their redistribution rights remain unadmitted.

## 2. Why this changes our research strategy

NEOPIC makes direct particle-measure operator learning a mandatory serious baseline for neural PIC field operators. It is conceptually different from:
- grid-based FNOs that learn on deposited fields and can receive aliased particle noise;
- LRC-FNO, which uses compressed fields and latent residual refinement;
- a warm-start neural model whose output is accepted only after a classical residual-correction solve.

**Add a variational Fourier particle method as an explicit comparator.** Variational Fourier methods are established prior art, but keep two schemes distinct. The gridless GEMPIF/PIF coupling and grid-based Fourier-GEMPIC coupling use different projections and have different fully discrete properties. The 2024 paper derives semi-discrete Gauss/energy/momentum properties for its variational spectral particle method and constructs fully discrete Hamiltonian-splitting schemes; the Fourier-GEMPIC variant uses a DFT grid, with conservation properties that differ from gridless PIF. Pin the exact projection, shape, filter and integrator. See *On Variational Fourier Particle Methods*, Journal of Scientific Computing (2024), DOI `10.1007/s10915-024-02708-w`. Our new small-N energy-gradient oracle is a known-answer tool for testing candidate learned energies; it is not a claim to invent variational PIF.

The core tournament therefore includes source NEOPIC NUFNO, deposited-grid FNO, LRC-FNO, gridless variational GEMPIF/PIF, grid-based Fourier-GEMPIC, standard CIC-PIC, PIF/NUFFT, Barnes–Hut and direct-sum small-N checks.

Keep adjacent solver lanes separate unless the governing equation, geometry, boundary conditions and diagnostic definitions match: local Gauss-preserving electrostatic Vlasov–Ampère PIC; semi-implicit energy-conserving electrostatic PIC in its high-density regime; and energy-conserving full electromagnetic PIC (e.g. EPIC-GOD) only in a full Vlasov–Maxwell cohort. These are stronger physical references for their own scopes, not interchangeable scores against every NEOPIC mini-app.

The benchmark needs three separate questions:
1. Is the particle-measure neural field operator more accurate at matched particle count and reference field solve?
2. Is it faster at matched field/trajectory accuracy, once NUFFT planning, spreading/interpolation, particle push and synchronisation are counted?
3. Does it preserve long-horizon physical behaviour and transfer across particle numbers, seeds, dimensions and integrators?

No single complexity result answers all three.

## 3. Paper-reported numerical anchors (not reproduction results)

### Particle-count transfer: 2D-2V weak Landau damping

The source reports N_tr and N_p values from 50,000 to 1.6 million. On the diagonal N_p=N_tr:
- teacher-forced field error decreases from about 2.3e-3 to 7.3e-4, with fitted scaling approximately N^-0.39;
- in-loop field error decreases from about 6.7e-3 to 1.5e-3, approximately N^-0.54.

Transfer is asymmetric:
- applying a model at fewer particles than training can increase error sharply;
- applying a model at more particles than training is more benign in the reported matrix.

A source example reports teacher-forced error 7.6e-3 for a model trained at 800k and evaluated at 50k, about ten times its diagonal error and about 3.4 times the error of the model trained at 50k. In-loop field error at 50k increases from 6.7e-3 for the 50k-trained model to 2.0e-2 for the 800k-trained model.

### Multi-app behaviours

The paper includes 2D / 3D mini-apps spanning weak/strong Landau damping, two-stream, bump-on-tail, cyclotron and Penning trap.

Reported 3D five-app model:
- weak Landau damping rate -0.179, compared with Barnes–Hut -0.191 before the 163,840-particle field reaches its noise floor;
- two-stream growth rate 0.537, compared with PIF 0.547 (within about 2%);
- two-stream saturation occurs later (about t=17.3 vs 16.4) and peaks higher (about 642 vs 509, roughly 26% higher).

These are concrete replication targets and visible failure-analysis points, not proof of superiority.

### Cost

The source's 3D cost analysis at N_p=4 million reports:
- full NEOPIC model: approximately 1.245 s per step;
- a smaller untrained subnet: 183 ms per step;
- Barnes–Hut: 172 ms;
- PIF at epsilon=1e-12: 158 ms (interpolated);
- Python PIC and PIF at epsilon=1e-6 are about three times cheaper than the untrained subnet.

The smaller subnet is a cost-only measurement; it is not an accuracy-qualified speed winner. The full model is not faster than the best listed classical methods in that particular configuration. Our benchmark must be honest about this.

### Implicit integration

The paper demonstrates one-dimensional implicit-midpoint compatibility. At Delta t=1, it reports approximately 0.64 s per step, compared with 2.8 ms for leapfrog NEOPIC and 3.2 ms for PIF; the implicit case requires on average 4.2 residual evaluations and 43 Jacobian-vector products. Multi-dimensional preconditioners and an efficient implicit implementation are explicitly left as future work.

## 4. Important limitations to preserve

- The learned operator is not exactly energy conserving; the paper states that it removes one important source of numerical heating (grid aliasing) but does not remove every mechanism.
- The approximation error epsilon_NN does not vanish as N_p grows or Delta t shrinks.
- In-loop error is larger than teacher-forced error due to dynamical amplification.
- A random split of temporally correlated snapshots from one teacher run is not independent validation; use independent teacher seeds/runs.
- Noise is part of the training distribution and acts as an implicit regulariser.
- Multi-particle-number training is not simple subsampling: labels must be recomputed for the target measure, and a subsample inherits correlations from the full run.
- Quiet starts are not IID; the paper's IID sampling analysis does not automatically apply.
- Every implicit step may invoke multiple field evaluations and Jacobian-vector products.

## 5. Reproduction tracks

### A. Exact paper reproduction

1. Resolve both paper tags to commit SHA and pin all submodules, runtime, CUDA/PyTorch/NUFFT versions and config files.
2. Verify exact tag-level licenses and dependency licenses.
3. Request the authors' HDF5 training/evaluation datasets, model checkpoints and teacher/reference outputs.
4. Verify SHA-256, file sizes, seeds, particle count, run configs and source solver for each artifact.
5. Reproduce the 2D-2V weak-Landau particle-count matrix first.
6. Reproduce 2D and 3D multi-app runs and the implicit 1D experiment.
7. Keep paper reproduction separate from any changed architecture, training split or integrator.

### B. Matched field-operator tournament

Compare:
- source NEOPIC NUFNO;
- source grid FNO / deposited-density operator;
- LRC-FNO source method;
- CIC PIC;
- PIF / NUFFT;
- Barnes–Hut / tree method;
- direct summation for small-N known-answer checks.

Use the same particles, field reference, domain, precision, time integrator and test seed where applicable. Distinguish teacher-forced field error from in-loop trajectory error.

### C. Particle-count transfer matrix

Evaluate a complete matrix over N_tr and N_p, with both transfer directions:
- N_p < N_tr;
- N_p = N_tr;
- N_p > N_tr.

Report absolute particle count and ratio N_p/N_tr. Include independent teacher seeds. Record field, position and velocity errors plus spectral noise diagnostics.

Mixed-particle training is a separate research extension and cannot reuse full-run labels for particle subsets.

### D. Symmetry / noise / aliasing

- arbitrary non-grid-aligned translation equivariance on periodic domains;
- representation equivalence under changing N_p;
- field error spectra, noise covariance and PSD;
- particle noise vs grid-deposition aliasing;
- IID random vs quiet-start initialisations as separate statistical regimes;
- domain-size, dimension and boundary changes only where source equations support them.

### E. Physics and numerical behaviour

- total particle-field energy drift, separated into field method and time-integrator contributions;
- field energy per component and damping/growth rates;
- two-stream saturation time and peak;
- bump-on-tail growth/plateau;
- cyclotron guiding-centre, phase and canonical angular momentum;
- Penning-trap oscillation / confinement;
- applicable Poisson/Gauss residual and charge consistency;
- timestep convergence, stability and Lipschitz/sensitivity;
- implicit solver residual/convergence and total Jacobian-vector product cost.

Do not require exact energy conservation from the original NEOPIC model; measure it. Compare any energy-corrected variant as a distinct method.

### F. Performance / economics

Measure the complete particle-field step:
- NUFFT plan construction vs amortised plan reuse;
- particle spreading/interpolation;
- learned operator;
- particle push;
- memory and communication;
- device transfer/synchronisation;
- full time per step across N_p;
- accuracy and physical behaviour at matched wall-time, energy and accelerator budget;
- training cost and amortisation.

## 6. Prospective gates (not results)

1. Reproduce all source particle-count and multi-app results before claiming improvement.
2. Beat the strongest classical reference only on matched accuracy and physics, not asymptotic complexity alone.
3. Establish calibrated error bounds for both low- and high-particle inference relative to training particle count.
4. Improve energy or trajectory fidelity without giving back the cost/accuracy advantage.
5. Demonstrate arbitrary-shift equivariance on the actual code path.
6. Include independent teacher runs and seeds; do not validate on correlated snapshots from the same run.
7. Treat a modified energy-projection/residual-correction wrapper as a separate method.

## 7. Admission state

| Artifact | State |
|---|---|
| arXiv paper | Identified |
| operator_learning repository | Public; default branch MIT observed |
| neopic-paper-scripts repository | Public; default branch MIT observed |
| separate NEOPIC repository | Public; default branch BSD-3-Clause observed |
| paper tags and exact commit SHAs | Pending immutable resolution |
| HDF5 training/evaluation datasets | Not in repository; author request required |
| trained model checkpoints | Not in repository; author request required |
| teacher/reference HDF5 outputs | Not in repository; author request required |
| exact paper reproduction | Not performed |
| head-to-head tournament | Not performed |

## 8. Normalization, topology and NUFFT qualification

The paper's own implementation details expose an important experiment that should be elevated to a first-class reproduction gate:

- Section 4.2 normalizes positions per sample and every timestep with an unpadded min–max bounding box.
- Section 4.3 uses a single linear lifting on normalized coordinates. The paper explicitly states this lifting is not periodic.
- Appendix A.1 says the error analysis is written for a fixed normalization; the min–max bounding-box dependence on the configuration and on particle count is an additional effect not covered by the analysis.
- Remark 16 states that a linear lifting of raw periodic coordinates is discontinuous across the periodic boundary, and that the Lipschitz lemma applies only away from the seam. A periodic embedding such as (sin x, cos x) satisfies the stated periodic Lipschitz assumption for periodic problems, but is not appropriate for open-boundary inputs without additional non-periodic information.
- The NUFFT path uses an oversampled uniform internal grid for spreading, FFT and interpolation. Its complexity is O(N_p) only for fixed mode count, architecture and tolerance, with a nonzero fixed cost term. Record the actual oversampling, kernel width, tolerance, mode count, planning and amortisation.

These are not speculative critiques: the normalization and lifting caveats are explicitly acknowledged in the paper. Their practical impact on the reported experiments still requires independent reproduction. An additional Pith T0 review flags them as major theoretical issues, but its page states that no human expert review had occurred; treat that report as an independent AI-generated audit hypothesis, not peer-reviewed confirmation.

Tracking: [NEOPIC periodic topology and normalization audit #12980](https://github.com/Luminous-Dynamics/symthaea/issues/12980).

### Required falsification experiments

1. Move a particle through x=0/L while holding the other particles fixed; measure field/force jumps at fixed probe particles, local derivative jumps and resulting trajectory divergence.
2. Perturb one extremal particle by epsilon; record the change to every normalized coordinate and the field output at unaffected particles.
3. Repeat across N_p and independent seeds; estimate operator Lipschitz ratios under both max-displacement and average-displacement metrics.
4. Compare the exact source implementation with fixed domain normalization plus periodic feature lifting. Keep open-boundary variants separate and use an explicitly padded domain / free-space treatment where appropriate.
5. Re-run teacher-forced and in-loop particle-count scaling on independent teacher trajectories; do not claim the theorem transfers to a modified architecture without redoing the proof or qualifying the claim.

## 9. Source identity and prior-art update (20 November 2026)

The immutable tag targets were resolved directly from the public Git references:

| Artifact | Tag | Commit SHA |
|---|---|---|
| `srikrish/operator_learning` | `neopic-arxiv-v1` | `2cf6a15e0ddb6c408a4ef6d162c31faf1deb9431` |
| `srikrish/operator_learning` | `neopic-arxiv-v1-nufft` | `1222226acab272b8d78ba541cb67c1e877669785` |
| `NEOPIC-framework/neopic-paper-scripts` | `neopic-arxiv-v1` | `2ee358abed5c782213d6d35c1194568da26ed8f1` |

These are annotated tags dereferenced to their commit targets. Dataset/checkpoint/HDF5 provenance and usage/redistribution rights still need independent admission.

## 10. Claim ceiling

This is a reproduction protocol and source admission record. It does not claim NEOPIC was reproduced, trained, beaten, integrated into Symthaea or validated for plasma control.
