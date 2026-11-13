# Universal error-set recovery and decoder reuse

**Status:** the exact three-qubit bit-flip error-set baseline passed the exact-head Rust 1.96 gate at `50404c09b1f7e7e79c91fa7a38b47716309b6b2e`: formatting and Clippy passed; all 153 frontier-physics crate tests passed (0 failed). The general approximate error-set Petz construction, its Hellinger/environment-leakage metrics, and universal QSVT implementation remain future work.

## Why this is a distinct paradigm

A channel-specific decoder is optimized for one noise channel. A universal error-set decoder is built once from a code space and a declared finite error set, then reused across every channel in a mathematically controlled family.

The August 29, 2026 preprint *Universal recovery in approximate quantum error correction* establishes a framework in which one recovery map works uniformly across a family of approximate error channels. It introduces:
- E-controlled channels: each channel Kraus operator lies in the span of a fixed error set, with a coefficient-matrix contraction bound.
- The Knill–Laflamme Hellinger distance, controlling average-case performance of the error-set Petz map.
- The environment-leakage distance, controlling worst-case performance under the stronger control condition.
- A Petz recovery map constructed from the error set and code, rather than re-optimized for every channel.

Reference: https://arxiv.org/abs/2608.28962 (preprint; pin the exact version and theorem assumptions before formal reproduction).

## Mathematical objects

Let Q be a K-dimensional code with projector P and orthonormal code basis \(\{|c_i\rangle\}\). Let \(\mathcal E=\{E_k\}_{k=0}^{S-1}\) be a fixed error set.

The QEC matrix is

\[
(A_{\rm QEC})_{(i,k),(j,l)}
=\langle c_i|E_k^\dagger E_l|c_j\rangle.
\]

Exact Knill–Laflamme correction requires

\[
A_{\rm QEC}=I_K\otimes\lambda
\]

for a positive semidefinite error-space matrix \(\lambda\).

For approximate correction, the Knill–Laflamme Hellinger distance is

\[
\zeta_H(\mathcal E,Q)
=\frac{1}{\sqrt K}
\min_{\lambda\succeq0}
\left\|\sqrt{A_{\rm QEC}}-
I_K\otimes\sqrt{\lambda}\right\|_2.
\]

The environment-leakage distance is

\[
\zeta(\mathcal E,Q)
=\inf_{\lambda}\left\|\mathcal B^\mathcal E_{\lambda,Q}\right\|_\diamond,
\]

where

\[
\mathcal B^\mathcal E_{\lambda,Q}(X)
=\sum_{k,l}{\rm tr}\!\left[
\left(PE_l^\dagger E_kP-\lambda_{k,l}P\right)X
\right]|k\rangle\langle l|.
\]

The source paper relates the Hellinger quantity to average-case universal recovery and the environment-leakage quantity to worst-case guarantees under stronger E-control. Do not swap these roles or report one metric as the other.

An E-controlled channel has Kraus operators

\[
F_\alpha=\sum_k c_{\alpha k}E_k
\]

with the coefficient matrix C satisfying the paper's specified contraction condition. The strongest worst-case statement uses its stronger coefficient constraint. Every theorem application must record which norm constraint is satisfied and verify channel CPTP conditions independently.

## Executable exact baseline now added

The new module src/universal_recovery.rs implements a small exact control:

- Three-qubit repetition code \(Q={\rm span}\{|000\rangle,|111\rangle\}\).
- Error set \(\mathcal E=\{I,X_0,X_1,X_2\}\).
- Four fixed syndrome recovery Kraus operators \(R_s: \mathbb C^8\to\mathbb C^2\).
- A family of stochastic channels \(\mathcal N_p(\rho)=\sum_s p_s E_s\rho E_s^\dagger\).
- One decoder reused unchanged across all probability vectors.

The code reports:
- Exact QEC-matrix Knill–Laflamme residual.
- Noise-channel trace-preservation residual.
- Recovery-channel trace-preservation residual.
- Full logical Kraus-family recovery residual.
- Entanglement fidelity.

Tests cover each one-hot error, uniform and asymmetric mixtures, a frozen 64-member probability sweep, reuse of the same decoder, and rejection of invalid probability vectors without silent normalization.

**Claim ceiling:** this is an exact-KL bit-flip control for stochastic mixtures of four declared unitary errors. It is not the general approximate error-set Petz algorithm from the new paper, does not compute the Hellinger or diamond environment-leakage distances, and does not validate arbitrary E-controlled coefficient matrices.

## Generic QEC-matrix implementation slice

The child branch `research/universal-error-set-qec-matrix-20261113` adds `src/universal_error_set.rs` with a validated complex code isometry, a finite set of dense physical error operators, and a generic QEC matrix builder. The builder computes \(A_{(i,k),(j,l)}=\\langle c_i|E_k^\\dagger E_l|c_j\\rangle\), checks the code-isometry orthonormality contract, reports QEC-matrix Hermiticity, and projects onto the exact linear Knill–Laflamme subspace \(I_K\\otimes\\lambda\). The projection is the Frobenius-orthogonal linear projection; it does **not** solve the PSD-constrained minimization for the approximate Hellinger metric, nor does it compute a diamond norm. The dense combined QEC dimension is explicitly capped at 1024.

Four authored tests cover the exact repetition-code QEC identity, complex non-unitary error operators, invalid code/error inputs, and dimension/dense-allocation limits. Their status remains source-authored until the new branch's exact-head Rust 1.96 workflow passes.

## The larger research programme

| ID | Milestone | Required evidence |
|---|---|---|
| UR-0 | Exact bit-flip error-set control | QEC matrix, four recovery Kraus operators, TP and logical identity residuals |
| UR-1 | General error-set representation | Typed code isometry, arbitrary finite error operators, register and basis conventions |
| UR-2 | QEC-matrix diagnostics | PSD/Hermiticity, support rank, KL projection and exact known answers |
| UR-3 | Average-case approximate recovery | Independently computed Knill–Laflamme Hellinger distance and theorem-matched average error |
| UR-4 | Worst-case approximate recovery | Environment-leakage diamond norm and strong E-control checks |
| UR-5 | Universal-versus-adapted decoder study | One error-set decoder compared with per-channel optimal/Petz decoders on common channel families |
| UR-6 | Universal QSVT compilation | One polar/QSVT implementation approximates the fixed error-set Petz map across the whole controlled family |
| UR-7 | Adversarial and distribution-shift tests | Channels at the boundary of the coefficient constraints, worst-case logical inputs and out-of-family rejection |
| UR-8 | Resource and scaling | Gates, depth, queries, ancillas, condition numbers, runtime, memory and certification cost |

## A potentially important fusion: universal QSVT recovery

The next paradigm-level experiment is to build the error-set Petz decoder once, then compile its polar isometry with QSVT once.

For an admissible family \(\{\mathcal N_C\}\), the target is a single approximate decoder \(\mathcal R^\theta_{\mathcal E,Q}\) for which the same approximation receipt certifies

\[
\sup_{C\in\mathfrak C}
d\!\left(
\mathcal R^\theta_{\mathcal E,Q}\circ\mathcal N_C|_Q,
I_Q
\right)
\]

under the exact metric and coefficient constraints of the selected theorem.

A bound on the approximation of the fixed decoder can be propagated through any CPTP encoding/noise channel by diamond-norm contractivity, but that alone does not establish that the ideal universal decoder corrects the family. The family-level error theorem and the QSVT approximation error must be reported separately and then combined with a justified inequality.

This is the most promising integration of the August 2026 universal-recovery result and the October 2026 polar/QSVT-Petz construction. It should be treated as a new research target, not an achieved result.

## External comparisons and falsification

- Compare with channel-specific Petz: the universal decoder must use identical channel samples but cannot use channel-specific recovery parameters at runtime.
- Compare with exact recovery SDP or a theorem-qualified bound on small cases.
- Compare average and worst-case metrics separately; do not infer worst-case guarantees from mean entanglement fidelity.
- Freeze channel-family definitions and coefficient-matrix constraints before evaluation.
- Include channels outside the declared span and channels violating coefficient constraints as typed rejection controls.
- Keep hardware and logical resource estimates separate from classical channel simulation.

## Negative controls

- Re-optimize the decoder separately for every channel while labelling it universal.
- Evaluate only the training mixture or average channel and claim uniform recovery.
- Use the Hellinger metric as a worst-case certificate.
- Use environment leakage as an average-case fidelity.
- Omit the stronger coefficient constraint when citing a worst-case theorem.
- Confuse an arbitrary channel Kraus basis with the declared error-set basis.
- Silently project a non-PSD QEC matrix onto the PSD cone.
- Compare an error-set decoder with channel-specific Petz at different access sets.
- Claim QSVT approximation error proves family-level recoverability without the ideal error-set theorem.

## Claim ceiling

A passing general implementation would qualify a declared finite-dimensional code/error-set family and its certified recovery map. It would not by itself establish black-hole interior reconstruction, a macroscopic evaporation model, gravitational islands/QES, or hardware advantage.
