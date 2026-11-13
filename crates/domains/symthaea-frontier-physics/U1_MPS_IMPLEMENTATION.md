# Complex U(1)-symmetric MPS state representation

**Status:** the complex charge-labelled MPS core and exact dense fixed-charge to charge-block MPS factorization are source-authored; exact-head Rust 1.96 validation is pending. Two-site gate application, TEBD gate sequencing, truncation and scaling remain unimplemented.

## Scope of this increment

The module \`src/u1_mps.rs\` introduces a real state representation, not a pedagogical parameter-count placeholder.

Each site has a complex rank-3 tensor \(A^{[n]}_{\ell,s,r}\), physical qubit charge \(q(s)=s\), and explicit integer charges on its left and right virtual bonds. The selection rule is

\[
q_r=q_\ell+s.
\]

The open-boundary MPS starts with a single charge-zero left virtual state and ends with a single right virtual state carrying the declared total charge \(K\). This makes the total U(1) sector a structural property of the tensor network.

## Implemented API

- \`U1MpsTensor::new(left_charges,right_charges,data)\`: finite-shape complex qubit tensor with explicit virtual-charge labels.
- \`ComplexU1Mps::new(tensors,total_charge,tolerance)\`: validates boundaries, adjacent bond-label identity and every tensor's U(1) selection rule.
- \`ComplexU1Mps::product_state(bits)\`: exact product-state constructor with MSB-first bit ordering.
- \`ComplexU1Mps::to_dense(maximum_sites)\`: explicitly bounded exact dense expansion for small-instance oracles; no normalization or truncation.
- \`ComplexU1Mps::norm_squared(tolerance)\`: double-layer transfer-environment contraction, rather than dense expansion.

The module also reports maximum bond dimension and selection-rule residual.

## Known-answer fixtures

1. Four-site product state \(|1010\rangle\): charge 2, unit norm, single bond dimension and exact dense basis index 10.
2. Entangled three-qubit one-excitation W state:
   \[
   (|001\rangle+|010\rangle+|100\rangle)/\sqrt{3}.
   \]
   This is represented with bond dimension 2 and total charge 1; its dense amplitudes and transfer-contracted norm are checked independently.
3. Negative controls for nonzero forbidden-charge tensor entries, mismatched adjacent bond labels, wrong total-charge boundary, non-finite tensor entries, empty chain and over-limit dense expansion.


## Dense fixed-charge to MPS factorization

The follow-up branch adds \`ComplexU1Mps::from_dense_fixed_charge\`. At each cut, it groups local rows by the intermediate prefix charge \(q\), groups suffix columns by the complementary charge \(K-q\), and performs a separate complex SVD for each block. The left singular vectors become the site tensor; singular-value-weighted right singular vectors become the remaining suffix state.

This is a charge-block factorization path with no Schmidt truncation:
- It removes only exactly zero singular directions; every positive singular value is retained, including very small positive Schmidt weights.
- It applies no bond cap, positive singular-value cutoff, normalization or renormalization.
- It rejects dense components outside the declared total-charge sector when they exceed the configured tolerance.
- It records maximum out-of-sector input amplitude, discarded out-of-sector squared weight, maximum dense reconstruction residual, L2 reconstruction residual and SVD rank by charge/cut.

Tests include dense-to-MPS reconstruction of the W state, a non-normalized \(|1010\rangle\) basis state, and a tolerated tiny charge-leakage component whose loss must appear in the receipt. This is the exact bridge needed before testing TEBD gate application. The factorization is numerically faithful to the reported SVD/reconstruction residual, not a claim of symbolic exactness in floating point. Exactly-zero SVD directions are omitted as nullspace, but every positive singular value is retained.

## Explicit limitations

This increment does **not** yet implement:
- Two-site \(U(1)\)-conserving gate application.
- Canonical-form sweeps or two-site gate SVD updates with per-gate truncation receipts.
- TEBD or nonlocal MPO evolution.
- Per-truncation discarded-weight accounting.
- Cumulative state-vector / reduced-state error bounds.
- Any N=16–64 scaling or matched Quimb/ITensor performance claim.

Those are the next stages. The implementation should add them in this order: exact two-site gate contraction without truncation, sector-resolved Schmidt decomposition, truncation receipts and dense-overlap qualification, then controlled scaling.

## Competitive benchmark policy

Quimb CircuitMPS exposes maximum bond dimension, singular-value cutoff and state-norm-based error/fidelity estimates; those are useful external controls, but raw elapsed time alone is not comparable. Match the gate tensors, qubit ordering, initial state, seeds, checkpoints, observables, bond cap and truncation policy. Compare error envelopes and dense overlap on small N before comparing throughput. ITensor's QN-conserving MPS facilities are another specialist implementation reference.

Primary external references:
- Quimb CircuitMPS API: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- Quimb MPS circuit guide: https://quimb.readthedocs.io/en/latest/circuit/circuit-mps.html
- ITensor QN conserving MPS documentation: https://docs.itensor.org/ITensorMPS/stable/tutorials/QN.html

## Claim ceiling

A passing version of these increments would establish validated complex U(1)-sector-labelled MPS storage, charge-block dense-to-MPS factorization with explicit leakage/reconstruction receipts, small-N dense expansion and norm contraction. It would not establish TEBD, an efficient tensor-network simulator, black-hole evaporation, Hawking radiation, gravitational backreaction or quantum advantage.
