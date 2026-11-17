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

## Robust decision envelope (stacked research increment)

The companion `robust` module accepts one explicit operator-norm bound per degree, `||ΔV_l||₂ ≤ δ_l`. It applies Weyl's inequality to each singular value and also propagates uncertainty in the degree-zero scale used by the common numerical-rank threshold.

A rank/template result is called guaranteed only if one candidate's worst-case L1 distance is strictly below every competitor's best-case distance. Otherwise the module returns a conservative set of templates that have not been ruled out. This is an interval decision, not a probability distribution.

The bounds are assumptions supplied by the caller. The module does not prove that the bounds hold, estimate them from the Jacobi SVD residual, or promote a spectral match into an action or pruning certificate. Its claim ceiling remains spectral compatibility only.


## Paired-matrix perturbation provenance (new)

The `paired` module closes a provenance gap in the caller-supplied Weyl bounds:

- A paired case carries the exact row-major reference and observed representation matrices for every degree, with stable reference/observation IDs.
- The reference matrices are re-SVD'd under the exact profile and must reproduce the stored SymFP spectrum, sweep count and normalized orthogonality residual bit-for-bit. A rank-only receipt without matrix/SVD provenance is rejected by this path.
- The observed-minus-reference matrix is constructed internally. The module computes an outward-rounded Frobenius norm upper bound from those finite f64 entries. Since `||A||₂ ≤ ||A||F`, this supplies a conservative operator-norm perturbation bound for that exact supplied matrix pair.
- The derived bounds feed the existing robust threshold/rank/template interval classifier. The paired receipt carries the complete matrix pair, derived bounds and classification; verification recomputes all of them.

This improves reproducibility over accepting an unexplained delta value. The exact guarantee is limited to the supplied reference/observed matrix pair and the outward-rounded numerical bound. It does **not** establish that the reference is the true physical representation, that future/noisy samples remain within the same envelope, or that a rank match proves a symmetry. The authority string remains `PAIRED_SPECTRAL_COMPATIBILITY_ONLY_NOT_A_SYMMETRY_CERTIFICATE`.

Tests cover zero-delta exact classification, a known 3-4-5 Frobenius bound, source-matrix mutation, forged-bound rejection, shape mismatch and overflow. The existing caller-supplied-bound API remains available as an explicit assumption-based path; use the paired API when the actual reference and observed representation matrices are available.


### Numerical qualification boundary

The paired path replays the reference SVD bit-for-bit and records each degree's normalized off-diagonal residual. It does **not yet propagate a rigorous floating-point SVD error bound** into the Weyl intervals. Accordingly:
- the Frobenius envelope is outward-rounded for the supplied observed-minus-reference f64 matrix entries;
- the nearest-template result is explicitly conditional on treating the reproducible reference SVD estimates as its anchor;
- `reference_svd_error_propagated=false` is carried in the machine-readable paired receipt;
- no exact-real singular-value guarantee, physical symmetry proof, or pruning authority is claimed.

The next numerical research increment is to derive and verify a conservative singular-value interval that accounts for finite-precision Jacobi rotations and residuals, or to use a separately verified interval-SVD oracle. Until then, the existing Weyl result remains an assumption-qualified spectral compatibility result.
