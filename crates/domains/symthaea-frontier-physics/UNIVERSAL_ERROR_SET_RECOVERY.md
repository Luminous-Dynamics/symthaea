# Universal error-set recovery and decoder reuse

**Status:** the exact three-qubit bit-flip baseline passed Rust 1.96 (153/153) at `50404c09…`. The general complex QEC-matrix constructor passed at `870ab5bf…` (157/157). The support-aware QEC square root, closed-form Hellinger metric and strongly-controlled Petz Bures bound are now source-authored; validation of the latest branch head is pending. The environment-leakage diamond norm and general error-set Petz recovery channel remain unimplemented.

## Why this is a distinct paradigm

A channel-specific decoder is optimized for one noise channel. A universal error-set decoder is built once from a code space and a declared finite error set, then reused across every channel in a mathematically controlled family.

The August 29, 2026 preprint *Universal recovery in approximate quantum error correction* establishes a framework in which one recovery map works uniformly across a family of approximate error channels. It introduces:
- E-controlled channels: each channel Kraus operator lies in the span of a fixed error set, with coefficient matrix spectral/operator norm `||C||_infinity <= 1`.
- The Knill–Laflamme Hellinger distance, controlling average-case performance of the error-set Petz map.
- The environment-leakage distance, which controls worst-case AQEC error.
- Strongly E-controlled channels, which additionally satisfy the paper's stronger coefficient constraint `||C||_2 <= 1`; the paper derives a direct operator-norm worst-case bound for the error-set Petz map on this subfamily.
- A Petz recovery map constructed from the error set and code, rather than re-optimized for every channel.

Reference: https://arxiv.org/abs/2608.28962v2 (2026 preprint). The distinctions below follow Definitions 2, 4 and 7 and Theorem 11 of the current version; pin the arXiv version when producing external reproduction receipts.

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

The paper defines E-controlled channels by the spectral/operator-norm constraint \(\|C\|_\infty\leq1\), where \(F_\alpha=\sum_k C_{\alpha k}E_k\). The stronger subclass of *strongly E-controlled* channels uses \(\|C\|_2\leq1\), with the paper's Schatten-2/Hilbert–Schmidt convention. Record these as different constraints; do not replace one by the other. Every theorem application must independently verify that the resulting map is CPTP.

An important implementation shortcut is that the Hellinger quantity has a closed form:

\[
\zeta_H(\mathcal E,Q)
=\sqrt{
\frac{\operatorname{tr}(A_{\rm QEC})}{K}
-\frac{1}{K^2}
\left\|\operatorname{tr}_{K}\sqrt{A_{\rm QEC}}\right\|_2^2
}.
\]

This removes an explicit optimization over \(\lambda\), but it still requires a support-aware positive-semidefinite square root of the QEC matrix. Do not add an arbitrary positive eigenvalue floor.

For the error-set Petz map, Theorem 11 also gives a direct worst-case Bures-distance bound on strongly E-controlled channels:

\[
d_{\rm wc}\!\left(
\mathcal R^{\rm Petz}_{\mathcal E,Q}\circ\mathcal N|_{L(Q)},I_{L(Q)}
\right)
\leq
\frac{1}{\sqrt 2}
\left\|
\sqrt{A_{\rm QEC}}-
I_K\otimes\frac{1}{\sqrt K}
\sqrt{\operatorname{tr}_K(A_{\rm QEC})}
\right\|_\infty.
\]

This operator-norm Petz bound is distinct from the environment-leakage diamond distance. Report both only when both are actually computed, and apply the Petz bound only after verifying strong E-control.

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


## General complex QEC-matrix constructor now added

The new module `src/error_set_qec.rs` accepts a complex code isometry V and any finite list of square complex error operators. It constructs the QEC Gram matrix
[
A_{(i,k),(j,l)}=\langle E_k c_i|E_l c_j\rangle
]
and computes the Frobenius-orthogonal projection onto the exact-KL subspace:
[
\lambda_{kl}=\frac{1}{K}\sum_i A_{(i,k),(i,l)}.
]
It reports code-isometry, Hermiticity and absolute/relative KL-projection residuals. The projection preserves dependent error-set members and rank-deficient \(\lambda\); no basis orthogonalization or spectral flooring is performed.

The exact-head Rust 1.96 audit passed formatting, Clippy and **157/157** frontier-physics crate tests at `870ab5bf2728f542e9ac64e2236b80795b9cb7` (workflow 43380572096). Five new tests exercise the exact repetition-code error set, a phase-error negative control, duplicated errors, invalid code/operator shapes and non-finite input.

This is UR-1 and part of UR-2. The QEC matrix is a Gram matrix and hence PSD in exact arithmetic, but this implementation does not yet return a floating-point minimum-eigenvalue certificate. It also does not validate channel CPTP constraints or compute either approximate-recovery metric.


## Support-aware square root, Hellinger metric and strongly controlled Petz bound

The new implementation uses nalgebra's complex Hermitian eigensolver and returns a receipt containing the minimum/maximum eigenvalue, numerical support rank, Hermiticity residual and square-root reconstruction residual.

The policy is fail-closed:
- Materially negative eigenvalues are rejected.
- Negative eigenvalues within the declared relative roundoff tolerance may be mapped to zero.
- Positive eigenvalues are never floored, including values below the support-rank threshold. The support threshold is a reported numerical-rank diagnostic, not a matrix regularizer.
- A small accepted Hermiticity residual is projected to its Hermitian part only for eigensolving; the original residual remains reported and reconstruction is checked against the original input.

The average-case Hellinger function implements the closed form from arXiv:2608.28962v2, Eq. (8):
\[
\zeta_H^2=\frac{\operatorname{tr}(A_{\rm QEC})}{K}
-\frac{\left\|\operatorname{tr}_K\sqrt{A_{\rm QEC}}\right\|_2^2}{K^2}.
\]

The strongly E-controlled worst-case Petz bound uses Theorem 11:
\[
d_{\rm wc}\leq \frac{1}{\sqrt2}
\left\|\sqrt{A_{\rm QEC}}-
I_K\otimes\frac{1}{\sqrt K}\sqrt{\operatorname{tr}_K(A_{\rm QEC})}
\right\|_\infty.
\]
Before returning this bound, the implementation checks the coefficient matrix Frobenius/Schatten-2 norm \(\|C\|_2\leq1+\epsilon\), constructs the Kraus operators \(F_\alpha=\sum_k C_{\alpha k}E_k\), and verifies \(\sum_\alpha F_\alpha^\dagger F_\alpha=I\) to the declared tolerance. Complete positivity follows from the supplied Kraus form. This is distinct from computing the environment-leakage diamond norm.

New authored tests cover rank-deficient PSD support, preservation of a small positive eigenvalue, rejection of non-Hermitian/materially indefinite inputs, zero Hellinger for the exact-KL repetition-code family, positive Hellinger for a phase-error negative control, zero Theorem-11 bound for an exact-KL strongly controlled TP family, rejection of coefficients outside the Schatten-2 ball, and rejection of a coefficient-induced non-TP channel. Exact-head Rust validation is pending; these are not yet described as passed.

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
