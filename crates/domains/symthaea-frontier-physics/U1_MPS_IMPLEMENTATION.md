# Complex U(1)-symmetric MPS state representation

**Status:** source-authored implementation now includes complex U(1)-symmetric MPS storage, dense fixed-charge factorization, validated local gates, exact and truncated charge-block two-site SVD, deterministic even/odd TEBD layers, mixed-canonical gauge receipts, a gate/model-projection/truncation error ledger, analytic U(1) Hamiltonian gates, and a second-order even/odd/even Strang stepper. The latest consolidated exact-head Rust 1.96 harness is the remaining validation gate. Long-range MPO/TDVP and matched large-N performance remain unimplemented.

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


## U(1)-conserving two-site gate semantics

The next branch adds a validated local gate interface in \`src/u1_mps.rs\`.

- \`U1TwoSiteGate::new\` accepts a complex 4×4 matrix in basis order \(|00\rangle,|01\rangle,|10\rangle,|11\rangle\). It independently checks unitarity and forbids matrix elements between distinct total-excitation sectors.
- \`U1TwoSiteState::from_adjacent_tensors\` contracts one internal MPS bond without splitting or truncating it.
- \`U1TwoSiteState::apply_gate\` applies the validated gate in the two-qubit physical basis and checks the resulting virtual-charge selection rule.
- Identity, SWAP and a complex U(2) single-excitation block provide deterministic controls.
- Small two-site open-boundary states can be expanded to four dense amplitudes for direct comparison.

The operation is deliberately **pre-SVD**. It establishes the gate convention and charge-preserving contraction, not a full MPS circuit simulator. The next work is the charge-block two-site split, followed by per-sector and global truncation accounting.


## Charge-block two-site split

The gate branch is now extended by \`U1TwoSiteState::split_exact\`, which performs the inverse side of the local-gate contraction.

For each intermediate charge \(q\):
- Rows are grouped by \((q_\ell,s_1)\) with \(q=q_\ell+s_1\).
- Columns are grouped by \((s_2,q_r)\) with \(q_r=q+s_2\).
- A complex SVD is performed only inside that charge block.
- Exact-zero singular directions are omitted as nullspace; every positive singular value is retained, including very small values.
- The returned diagnostics include rank by charge, retained rank, exact-zero count and the max/L2 residual after re-contracting the two returned site tensors.

This is the exact, untruncated SVD split required to implement TEBD. It still does not impose a bond cap or approximate cutoff. The next stage is a *separate* truncating split that must return discarded squared Schmidt weight, normalization change, bond-cap saturation and a conservative error ledger; do not mix that approximation into \`split_exact\`.


## Explicit truncation policy and cumulative error ledger

The new approximate path is separate from \`split_exact\`:

- \`U1TwoSiteTruncationPolicy\` carries a global maximum bond dimension, a relative singular-value cutoff \(r\in[0,1]\), and a numerical tolerance.
- Cutoff semantics are explicit: retain \(\sigma_i\ge r\,\sigma_{\max}\).
- The cutoff is applied first; the global bond cap is then applied to the remaining singular values ranked across *all* charge sectors.
- Charge labels remain attached to every retained Schmidt direction, so the cap cannot mix charge sectors.
- The returned tensors remain unnormalized. Diagnostics report pre/post norm squared and the factor that would restore the original norm, without applying it.
- Cutoff-discarded weight and bond-cap-discarded weight are separate, as are their singular-value counts.
- The ledger accumulates \(\sum_j\sqrt{w_j}\), a conservative L2 error upper bound by the triangle inequality for a sequence of unitary gates and orthogonal Schmidt projections. This bound is not claimed when non-unitary operations or nonlinear renormalization are interleaved.

The benchmark suite must compare against Quimb CircuitMPS with matched gate sequence, qubit ordering, initial vector, precision, cutoff semantics, bond cap, and hardware. Quimb already exposes maximum bond dimension, cutoffs, swap-and-split gate contraction, and norm-based fidelity/error estimates. ITensor already provides Abelian QN block-sparse MPS and automatic symmetry enforcement. The proposed advantage is not the novelty of MPS or U(1) conservation; it is the auditable end-to-end receipt for each symmetry-sector operation and the separation of approximation sources.


## Deterministic even/odd TEBD layer executor

The new scheduler composes the local gate, exact charge-aware contraction, and explicit truncation components:

- \`U1TeBDLayer::new(site_count, parity, gates)\` validates parity \(p\in\{0,1\}\), bond range, uniqueness and parity membership, then sorts gates by bond index.
- A parity layer contains only disjoint nearest-neighbour bonds (even: 0,2,4,…; odd: 1,3,5,…).
- \`U1TeBDLayer::apply\` applies each validated two-site unitary, performs a charge-block split and the requested global truncation policy, then revalidates the entire MPS.
- The result includes input/output norm squared, total charge, output selection-rule residual, bond order, per-gate truncation receipts and the local truncation ledger.
- There is no implicit renormalization.

**Important error-bound qualification:** the sum of local \(\sqrt{w_j}\) values is not automatically a global state-vector bound for an arbitrary MPS gauge. The global triangle-inequality bound requires a mixed-canonical representation with orthonormal exterior left and right block bases at each truncation, as well as unitary gates and no nonlinear renormalization between projections. This scheduler currently returns \`global_l2_bound_qualified = false\`. A canonicalization sweep and a machine-checkable gauge receipt are required before upgrading that claim.

This is a layer executor, not yet a complete TEBD time-stepper: second-order even/odd/even or odd/even/odd Trotter composition, Hamiltonian-to-gate exponentiation, adaptive time stepping, and a global canonical-gauge error certificate remain future increments.


## Mixed-canonical gauge qualification

The canonical-gauge branch adds charge-block left and right sweeps and a measured receipt for a two-site center:

- \`ComplexU1Mps::canonicalize_for_bond(bond, tolerance)\` makes sites left of the center pair left-isometric and sites right of it right-isometric.
- Each sweep factors only charge-compatible matrix blocks; no positive singular-value cutoff or bond cap is applied.
- \`ComplexU1Mps::inner_product\` computes the overlap between two MPS representations with different virtual bond dimensions, without dense expansion.
- The candidate gauge is built transactionally. If factorization, validation or overlap checks fail, the original MPS is unchanged.
- The receipt includes each exterior site's Gram-isometry residual, the maximum residuals, conservative exterior operator-norm upper bounds, factorization residuals, a state-reconstruction L2 residual computed from the cross-MPS overlap, norm drift, charge residual and a qualification bit.

For a site's left-isometry matrix \(A\), the receipt measures \(\|A^\dagger A-I\|_F=r\), which implies \(\|A\|_2\le\sqrt{1+r}\). The corresponding right-isometry bound is obtained from \(\|AA^\dagger-I\|_F\). Products of these per-site bounds provide a conservative upper bound for the exterior block maps. The local discarded Schmidt norm can then be multiplied by the left/right exterior operator-norm product to obtain a global projection-error contribution, provided the gate is unitary and the gauge receipt passes. Canonicalization's own state-reconstruction residual must also be included in the ledger.

This receipt is an independent check of the canonical form, not a declaration that an arbitrary MPS was already canonical. The TEBD layer now calls this method immediately before each local gate/truncation, then keeps the canonicalization receipt adjacent to that gate's truncation and gate-quality measurements.

The global error ledger separates three contributions per gate:

1. **Canonicalization representation drift:** the measured cross-MPS state-reconstruction L2 residual.
2. **Gate quality:** a polar-unitary correction bound derived from the Frobenius residual \(\|G^\dagger G-I\|_F\), plus a charge-off-block bound \(\sqrt{10}\,r_Q\) for the ten forbidden entries in a two-qubit \(4\times4\) gate, each multiplied by the pre-gate state norm.
3. **Schmidt projection:** \(\sqrt{w_\mathrm{discarded}}\) multiplied by the measured product of left/right exterior operator-norm upper bounds.

The per-gate contributions are accumulated by the triangle inequality, and the global flag remains false as soon as a gauge receipt or gate-quality check fails. No post-truncation renormalization is performed. The bound applies to the circuit formed by the closest charge-conserving unitary represented by each accepted gate matrix; gate defect contributions make the distinction from the supplied floating-point matrix explicit. It still excludes Trotter/discretization error because the layer executor is not yet a Hamiltonian time-stepper.




## Hamiltonian-driven second-order Strang TEBD

The latest increment connects local gates to an actual nearest-neighbour Hamiltonian evolution path.

- \`U1TwoSiteHamiltonian\` validates finite entries, Hermiticity and U(1) charge conservation.
- Its nearest Hermitian, charge-block-diagonal projection is computed explicitly. The Frobenius residual from the supplied matrix is recorded rather than silently discarded.
- The \(q=0\) and \(q=2\) sectors exponentiate as scalar phases; the \(q=1\) sector uses the analytic exponential of a \(2\times2\) Hermitian matrix. No generic \(4\times4\) eigensolver is needed.
- Every generated gate carries a Duhamel operator-norm upper bound for the difference between the supplied tolerance-qualified Hamiltonian and the Hermitian U(1)-block projection actually exponentiated.
- \`U1TeBDStrangStepper\` constructs a symmetric even-half / odd-full / even-half composition. The same truncation/error ledger is carried through all three layers and can be continued across successive time steps with \`apply_with_ledger\`.

### Four distinct error channels

1. **Hamiltonian model projection:** discrepancy between the supplied tolerance-qualified local matrix and its exact Hermitian, U(1)-block projection. This is included in the global ledger using the measured Frobenius residual and a Duhamel exponential bound.
2. **Representation/evolution implementation:** canonicalization reconstruction, gate unitarity defect and charge-sector leakage, propagated by subsequent gate operator-norm bounds.
3. **Compression:** discarded Schmidt weight measured only after the relevant mixed-canonical exterior-isometry receipt passes. No implicit renormalization is performed.
4. **Time discretization / Trotter splitting:** deliberately excluded from the global state-error ledger. The result reports \`time_discretization_error_qualified = false\`. The second-order claim is a numerical-method property, not a certified trajectory error; it must be established by a step-refinement envelope or an independent commutator bound.

The dense three-site one-excitation test checks the exact hopping-chain solution and a roughly fourfold reduction in global error when the time step is halved. A separate multi-step test checks that the global ledger is carried forward instead of being reset. These are authored acceptance tests until the consolidated exact-head runner passes.


## Goal-driven error-budget compilation

The new adaptive path reverses the traditional tuning workflow. Instead of selecting a relative singular-value cutoff and inspecting the error afterward, the caller supplies an absolute local two-site L2 projection target and a global maximum bond dimension:

- \`U1TwoSiteState::split_with_error_budget\` estimates the smallest bond rank from the exact charge-block Schmidt spectrum.
- It then checks the measured reconstruction residual. If finite-precision SVD error makes the residual exceed the target, it retains more modes and retries.
- The policy is global across charge sectors; selection is by largest Schmidt values, with charge labels preserved.
- If the maximum bond cannot meet the requested residual, the method returns \`TruncationBudgetInfeasible\` with the best measured residual under the cap. It does not return an under-qualified state.
- No implicit renormalization occurs.
- The receipt reports requested and achieved local residual, selected/minimum bond dimension, cap headroom and SVD factorization passes. Candidate ranks reuse the exact charge-block factors; the current path performs one factorization pass regardless of how many retained ranks are checked.

The accompanying \`U1MpsErrorBudgetPlan\` compiles an end-to-end L2 target into explicit canonicalization, gate-quality, Hamiltonian-projection, compression and time-discretization allowances. Fractions must be finite, non-negative and sum to at most one. The unallocated fraction is retained as an explicit reserve. Its equal-per-gate compression allocation is a baseline for unitary/equal-sensitivity circuits; a later sensitivity-weighted planner must account for downstream operator-norm amplification and exterior gauge factors before claiming global optimality.

**Error-domain boundary:** \`maximum_local_l2_error\` is a two-site tensor Frobenius target. It becomes a global state-vector target only after multiplying by qualified canonical-gauge exterior operator-norm bounds and accounting for all other error channels in the global ledger. The API does not silently reinterpret local residuals as global fidelity guarantees.


## Observable-specific accuracy envelopes

The next layer converts a state-vector guarantee into an error interval for a bounded observable. Given an operator-norm upper bound \(a\), an approximate-state norm upper bound \(n_a\), an exact-reference norm upper bound \(n_e\), and a qualified global state L2 bound \(\delta\), the receipt uses
\[
\left|\langle A\rangle_a-\langle A\rangle_e\right|
\le a\,\delta\,(n_a+n_e).
\]

The API supports two directions:

- \`U1MpsObservableErrorTarget::compile_state_l2_error_budget\` converts a requested absolute expectation error into a sufficient global state-L2 budget.
- \`U1MpsObservableErrorTarget::certify\` uses the qualified global ledger to return an expectation-error upper bound and whether the requested observable tolerance is met.

The exact-state norm upper bound is caller-supplied and must be justified from the reference dynamics; for an exactly unitary reference trajectory, the initial state norm is sufficient. The approximate norm is measured from the actual MPS, without assuming it is normalized. The receipt explicitly sets `trotter_error_included = false`: for Strang TEBD it bounds the implementation relative to the discrete gate sequence, not the continuous-time Hamiltonian trajectory.

**Important boundary:** this is an error-envelope/certification helper, not yet an observable evaluator. It does not implement arbitrary MPO expectation contraction or infer the observable operator norm. Applications must supply a valid norm upper bound (for a Pauli string, 1 is exact; for a weighted sum, the sum of absolute coefficient bounds is a simple conservative choice). An unqualified local truncation ledger is rejected; a zero operator is the trivial exact exception.

## Explicit limitations

This increment includes charge-block mixed-canonical sweeps, measured exterior-isometry receipts, gate and Hamiltonian-projection quality checks, a global ledger for representation/model-projection/compression errors, and a nearest-neighbour second-order Strang TEBD stepper. It does **not** yet implement:
- A Trotter-error certificate, adaptive time-step controller, or higher-order TEBD4 integrator.
- Long-range MPO application, lazy/nonlocal gate contraction, or TDVP.
- Independent Hamiltonian-to-MPO and local-effective-Hamiltonian qualification.
- N=16–64 scale-out or accuracy-matched Quimb/ITensor/SeeMPS/cuQuantum performance evidence.

The next stages are observable evaluation and MPO norm bounds, sensitivity-weighted error-budget allocation, Trotter-envelope qualification, canonical-gauge TDVP/MPO exploration, lazy/nonlocal MPO support, and controlled accuracy-matched comparisons.

## Competitive benchmark policy

Quimb CircuitMPS is a strong direct TEBD/MPS baseline: it supports maximum bond dimension, singular-value cutoff, swap-and-split two-qubit gates, nonlocal MPO contraction, lazy delayed compression, and norm-based fidelity/error estimates. Its lazy path is a required comparator, not an optional extra. ITensor supplies Abelian QN block-sparse MPS and symmetry-aware tensor operations. SeeMPS/TDVP and the standard TDVP reference are important long-range baselines: 2TDVP handles general MPO Hamiltonians and dynamic bond dimensions, while 1TDVP has different energy/norm-conservation and bond-growth trade-offs. We should compare at matched accuracy and include these modes rather than asserting TEBD is universally superior.

Match the Hamiltonian, MPO or gate tensors, site ordering, initial state, precision, bond cap, truncation threshold, Krylov tolerance, time-step schedule, hardware/backend and output checkpoints. Compare error envelopes and dense overlap for small N before throughput.

Primary external references:
- Quimb CircuitMPS API: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- Quimb MPS circuit guide: https://quimb.readthedocs.io/en/latest/circuit/circuit-mps.html
- ITensor QN conserving MPS documentation: https://docs.itensor.org/ITensorMPS/stable/tutorials/QN.html
- Quimb lazy MPS and nonlocal MPO compression: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- TDVP errors and conservation trade-offs: https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html
- TEBD time-step error and higher-order splitting: https://tensornetwork.org/mps/algorithms/timeevo/tebd.html

## Claim ceiling

A passing exact-head harness would establish validated complex U(1)-sector-labelled MPS storage, charge-block dense factorization, local gates, exact and approximate charge-block splits, deterministic parity layers, mixed-canonical gauge receipts, a global bound covering canonicalization/gate/model-projection/compression defects, and a nearest-neighbour second-order Strang stepper. It would **not** establish a Trotter-error certificate, long-range MPO/TDVP support, advantage over Quimb/ITensor, large-N speedup, black-hole evaporation, Hawking radiation, gravitational backreaction or quantum advantage.
