# SymFP spectral core reference v0.1

This dependency-free Rust workspace is a formula-level reproduction of the core method in Cen et al., *Spectral symmetry fingerprints in equivariant representations*, arXiv:2611.07410v1 (submitted 2026-11-05): https://arxiv.org/abs/2611.07410.

## Reproduced mathematical definitions

1. The singular values of each degree-l equivariant representation block are computed with a deterministic one-sided Jacobi SVD.
2. The SymFP is the concatenation of the ordered singular spectra across the requested degrees.
3. The numerical-rank threshold is tau = max(tau_abs, tau_rel * sigma_1^(0)), using the degree-zero leading singular value for all degrees.
4. The numerical rank sequence counts singular values strictly greater than tau.
5. Discrete point-group identification minimizes the unweighted L1 distance between the observed rank sequence and the candidate rank templates. All tied minimizers are returned; no arbitrary tie-breaking label is emitted.
6. Degree leakage is the squared spectral tail beyond the target fixed-subspace dimension, normalized by total degree energy plus epsilon_norm.
7. Pooled leakage is the uniform mean across selected degrees; pLS = -log10(LS + epsilon_log).

The 32 natural-parity crystallographic point-group rank templates for degrees l=0..11 are transcribed from Supplementary Table A.1 of the paper. The paper reports that degrees 0..9 distinguish all 32 templates, while degrees 0..6 leave D6/D6h and O/Oh tied.

## Crucial authority boundary

SymFP spectra are an orientation-invariant proposal and symmetry-breaking signal under the paper's equivariance assumptions. They are **not a symmetry proof**. A small leakage score only means that the observed spectral rank does not exceed the target fixed-subspace dimensions. Different groups can share or nest rank constraints; the paper explicitly gives cases with zero leakage against a target group that the input does not possess.

Accordingly, this crate emits a spectral-compatibility result only. It does not issue mathematical, physical, or pruning authority. Downstream qualification must independently verify the transformation action, Lie/finite group structure, physical step map, and full task semantics.

## Implemented and not yet reproduced

Implemented here: the published crystallographic rank-template table; degree-block SVD; paper numerical-rank threshold; nearest-template L1 matching with ties; degree-resolved and pooled leakage; pLS; rotation/spectrum invariance tests; subgroup monotonicity tests; and negative controls for pLS over-interpretation.

Not implemented here: the parameter-free equivariant GNN, geometric graph construction, radial bases/spherical harmonics, the paper's 32-group point-cloud dataset, learned HEGNN/TFN/MACE/Equiformer pipelines, molecular/protein datasets, or the paper's training regimes. This PR is a faithful mathematical-core reference, **not an end-to-end reproduction of the paper's reported experiments**. The unavailable or not-yet-implemented components remain explicit follow-on work.

## Reproduction

    cargo fmt --manifest-path research/symfp-spectral-core-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symfp-spectral-core-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symfp-spectral-core-001/Cargo.toml --all-targets -- -D warnings
    cargo run --release --locked --manifest-path research/symfp-spectral-core-001/Cargo.toml --example rank_template_probe

## Paper-specific numerical settings

The paper uses tau_rel=3e-5 for its degree-0..6 evaluation and tau_rel=1e-6 for degree-0..9 evaluation, selected on its validation structures. These are reproduced as named profile constructors, but they are not universally calibrated tolerances for arbitrary data. The default profile is intentionally uncalibrated and must not be used to claim paper reproduction on unseen representation families.

The paper sets epsilon_norm=epsilon_log=1e-10. The crate records these separately from the rank tolerance.

## Prospective next stage

Reproduce the paper's parameter-free equivariant representation construction with the exact centering, radial basis and spherical-harmonic conventions, then freeze the 32-group synthetic dataset and splits. Only after that should we compare SymFP, LieFlow, LieGAN and exact geometric stabilizer oracles on matched inputs. Paper-reported metrics remain paper results until independently reproduced.

## Robust ambiguity extension (stacked research increment)

The companion robust module adds a conservative uncertainty-aware decision layer beyond the paper's point-estimate formulas. For each degree l, the caller supplies an explicit operator-norm perturbation bound delta_l satisfying ||Delta V_l||_2 <= delta_l. Weyl's singular-value perturbation inequality gives intervals around each singular value. The degree-zero singular value used in the shared rank threshold is itself uncertain, so the module propagates a lower/upper threshold interval to every degree.

From these intervals it derives a minimum/maximum numerical rank per degree and a minimum/maximum L1 distance for each candidate template. It reports a guaranteed nearest template only when one candidate's worst-case distance is strictly smaller than every competitor's best-case distance. Otherwise it returns the candidates not ruled out by interval dominance and abstains on a unique label.

The perturbation bounds are assumptions supplied by the caller; the module does not prove them, derive them from the SVD orthogonality residual, or interpret them as a posterior. This is an ambiguity-control extension, not part of the published SymFP algorithm, and still provides spectral compatibility only.