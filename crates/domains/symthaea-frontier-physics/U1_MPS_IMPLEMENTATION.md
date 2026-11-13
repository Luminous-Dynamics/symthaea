# Complex U(1)-symmetric MPS state representation

**Status:** the complex charge-labelled MPS core, charge-block dense factorization, U(1)-conserving two-site gates, exact two-site charge-block split, and explicit global truncation/error ledger are source-authored. The latest truncation branch is awaiting its exact-head Rust 1.96 harness. Full TEBD sequencing and scaling remain unimplemented.

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

This receipt is an independent check of the canonical form, not a declaration that an arbitrary MPS was already canonical. The next integration step is to make TEBD call this method immediately before each local truncation and promote the global error flag only when every receipt and contribution passes.


## Globally qualified TEBD truncation receipts

The integrated layer executor now canonicalizes the current MPS around each active bond immediately before applying that gate and its truncation. It stores the per-gate \`U1CanonicalGaugeReceipt\` and passes it to the truncation ledger.

For each qualified gate \(j\), the global error contribution is bounded by
\[
e_j \le \delta^{\mathrm{gauge}}_j
+ \|L_j\|_{\mathrm{upper}}\|R_j\|_{\mathrm{upper}}\sqrt{w_j},
\]
where \(\delta^{\mathrm{gauge}}_j\) is the measured MPS state-reconstruction L2 residual introduced by finite-precision canonicalization, \(w_j\) is the local discarded squared Schmidt weight, and the two exterior operator-norm upper bounds come from the receipt's per-site Gram residuals. The layer's cumulative global bound is the sum of these contributions, valid for unitary gates and orthogonal Schmidt projections without nonlinear renormalization.

A single missing, mismatched or failing receipt invalidates the global bound for the entire layer. The local discarded-weight ledger remains available, but the global upper bound is cleared and cannot be silently re-enabled later in that layer. Every result carries both the individual receipts and the global qualification flag.


## U(1)-conserving Hamiltonian evolution and Strang time stepping

The real-time layer connects local physics to the already-qualified gate/split/compression path.

- \`U1TwoSiteHamiltonian::new\` validates finite entries, Hermiticity and the U(1) block structure in basis \(|00\rangle,|01\rangle,|10\rangle,|11\rangle\).
- \`real_time_gate(dt)\` analytically evaluates \(e^{-i\,dt\,h}\): independent phases in the zero- and two-excitation sectors plus a closed-form Hermitian \(2\times2\) exponential in the single-excitation sector.
- \`U1TeBDStrangStepper::new\` takes one nearest-neighbour Hamiltonian per bond and creates even half-step and odd full-step gate layers.
- \`apply\` executes even-half / odd-full / even-half, returning each layer's canonical-gauge and truncation receipts, charge, norm and the cumulative *truncation-only* global L2 bound.
- Trotter/product-formula error is separate: \`trotter_error_bound = None\` and \`trotter_error_qualified = false\` until an independent estimator is implemented. The global truncation bound must never be presented as a bound against the exact continuous-time Hamiltonian flow.

Known-answer tests cover the analytic two-site XY exponential, rejection of non-Hermitian or charge-changing Hamiltonians, exact two-site forward/reverse evolution, and second-order convergence of a three-site XY chain against its analytic solution. For second-order Strang, halving \(\Delta t\) should reduce the global discretization error by approximately four in the asymptotic regime.

## Step-doubling, Hamiltonian projection and typed error-budget diagnostics

The step-doubling path compares one coarse Strang step at \(dt\) with two fine steps at \(dt/2\):

- The coarse/fine state-vector defect is measured by transfer-environment norms and cross-state overlap; no dense expansion is needed.
- For second-order Strang, the asymptotic Richardson estimate for the fine solution uses the factor \(1/(2^2-1)=1/3\).
- The estimate includes both the coarse and fine canonical-gauge truncation bounds as contamination terms, rather than mislabeling compression loss as Trotter error.
- The result carries the measured defect, separate coarse/fine truncation bounds, gauge qualification status and Richardson estimate. A missing gauge receipt suppresses the estimate.

Each bond Hamiltonian also reports its Hermiticity residual, charge-leakage residual and Frobenius projection residual. The analytic exponential uses the explicit Hermitian, charge-block projection, with the single-excitation coupling symmetrized before exponentiation.

A typed error-budget receipt keeps four sources separate: Hamiltonian projection, canonicalization rebase, Schmidt truncation, and the step-doubling/Richardson estimate. It deliberately leaves the total physical error upper bound empty and its qualification flag false. The Richardson term is asymptotic, not a formal Trotter theorem; f64 residuals are not interval-enclosed. This is an evidence-carrying budget, not yet a total physical-error certificate.

## Product Abelian symmetries: U(1)^K charge vectors

The new abelian_mps module establishes a generic charge-vector contract without changing the validated scalar-U(1) TEBD path.

- AbelianCharge<K> stores K independent integer-valued conserved quantities with checked component-wise addition and subtraction.
- AbelianChargeTensor<K> accepts an arbitrary local physical basis, with an explicit K-vector charge for every physical state, and enforces the component-wise selection rule \(q_R=q_L+q_s\).
- AbelianMps<K> checks charge-zero left boundary, declared total-charge right boundary, and exact adjacent virtual-charge identity.
- AbelianChargeBlockPlan<K> partitions a two-site tensor into independent intermediate charge-vector sectors: rows use \(q_M=q_L+q_{s_1}\), columns use \(q_M=q_R-q_{s_2}\).
- The current two-component fixture uses \((N,2S^z)\): vacuum \((0,0)\), spin-up \((1,1)\), spin-down \((1,-1)\), and double occupancy \((2,0)\).

This is a structural tensor contract and deterministic block planner, not yet a generic multi-charge SVD, truncation ledger, TEBD executor, or non-Abelian fusion implementation. The next step is to reuse the charge-block plan in a generic sector-wise SVD, then port the truncation receipts and the error-ledger machinery without weakening the scalar-U(1) qualification gates.

## Exact U(1)^K charge-block two-site SVD

The block planner now feeds an exact generic two-site split:

- Contract the adjacent site tensors using their shared virtual charge-vector bond.
- For each intermediate K-vector charge, assemble only the rows with \(q_M=q_L+q_{s_1}\) and columns with \(q_M=q_R-q_{s_2}\).
- Run an independent complex SVD within each matching block; no singular vector can mix distinct charge vectors.
- Omit exactly-zero singular directions as nullspace and retain every positive singular value, however small.
- Rebuild two charge-labelled site tensors and report total/retained SVD dimensions, exact-zero count, rank by intermediate charge vector, and max/L2 reconstruction residual.
- No bond cap, positive cutoff, normalization or renormalization is applied.

Known-answer controls reconstruct a two-component \((N,2S^z)\) sector, preserve a \(10^{-18}\) positive Schmidt component, and reject zero states and mismatched virtual charge vectors. This is the first generic U(1)^K numerical decomposition, but it remains a two-site exact primitive: generic multi-charge truncation receipts, canonicalization, TEBD, Hamiltonian gates and matched performance are not yet implemented.

## Explicit limitations

This increment does **not** yet implement:
- A formal Trotter error theorem or commutator-norm bound; the new step-doubling quantity is an asymptotic Richardson estimate calibrated against known-answer cases, not a rigorous bound.
- Adaptive time-step control driven by a separately allocated total-error budget.
- Higher-order Suzuki compositions (fourth and above) or a Trotter-versus-truncation global error budget.
- Nonlocal MPO evolution, periodic-boundary MPS and long-range Hamiltonian compression. The new U(1)^K module now provides exact two-site charge-block SVD and reconstruction receipts. Generic multi-charge truncation, canonicalization, TEBD/Hamiltonian evolution and non-Abelian sectors are still unimplemented.
- Outward-rounded/interval arithmetic for a formal floating-point proof of every residual; current receipts are numerically qualified in f64 and must be checked against dense known-answer oracles.
- N=16–64 scaling and matched Quimb/ITensor performance evidence.

The next steps are: (1) calibrate the step-doubling estimate over longer runs and add energy/local-observable drift receipts; (2) allocate an explicit total-error budget among Hamiltonian projection, Trotter, truncation and floating-point contributions; (3) compare matched workloads against Quimb and ITensor; (4) generalize charge labels to multiple commuting conserved quantities; (5) explore hybrid TEBD / two-site TDVP / Krylov evolution when entanglement growth makes fixed-bond TEBD inefficient.

## Competitive benchmark policy

A fair comparison is not "same \(N\)" or "same max bond" alone. Freeze and publish the initial state, conserved sector, exact gate/Hamiltonian tensors, qubit/site ordering, time step, dtype, arithmetic backend, thread count, hardware, compression semantics, checkpoints and observables.

**Quimb.** Current \`CircuitMPS\` exposes \`max_bond\`, \`cutoff\`, \`gate_opts\`, and compression choices including direct SVD, density-matrix and successive-randomized compression. It also provides norm-based fidelity/error estimates. Its documentation includes 960-gate examples (one reported \(D_{\max}=143\), error about \(3.09\times10^{-3}\); a permuted-MPS variant reports \(D_{\max}=99\), error about \(1.75\times10^{-3}\) at lower runtime). These examples are useful reference points, not directly comparable benchmark results: circuits, qubit ordering and hardware settings must be matched.

**ITensor/ITensorMPS.** The current TEBD tutorial constructs nearest-neighbour gates from local Hamiltonians, applies them to an MPS with a cutoff, and explicitly describes the second-order Trotter error. The QN tutorial uses QN-enabled indices and an initial state with a fixed total quantum number; the library then preserves that QN sector. This is a strong baseline for both symmetry and physical time evolution.

**Our intended differentiator** is the receipt chain: per-gate charge-block ranks, local discarded weights, canonical-gauge checks, exterior norm factors, a dense sequential-circuit falsification test, and separate Trotter/truncation status. We should aim to win on *verified error per unit memory/time*, not by claiming MPS, U(1) conservation, TEBD or block-sparse tensors as novel.

The benchmark ladder should be:
1. **Exact regime:** \(N=2\ldots12\), exact dense overlap, fixed-charge random states, W/Dicke states and small XY/XXZ chains.
2. **Numerical evolution:** Strang order-two convergence, norm/charge/energy drift, local densities and correlations, and an independent dense oracle wherever feasible.
3. **Controlled approximation:** vary \(D_{\max}\), relative cutoff and \(\Delta t\) independently; report truncation and Trotter errors separately, and compare the dense residual against the canonical-gauge truncation bound.
4. **Scaling:** \(N=16,32,64\) at fixed filling and several entanglement regimes; publish peak/median bond, total retained charge-block ranks, wall time, peak RSS, allocations, energy drift and bound tightness.
5. **External matched runs:** compare Quimb direct/DM/SRC modes and ITensor QN-TEBD at equal initial states, Hamiltonians, site ordering, precision, hardware, bond caps and measured global error.

Proposed research targets—not current results—are: a second-order convergence ratio between 3.3 and 4.8 under step halving; dense-oracle residual below the reported truncation bound plus the documented roundoff allowance; and, at equal measured error, at least 20% lower peak memory or runtime on a pre-registered U(1)-dominated workload. If neither resource target is achieved, retain the auditability value proposition and do not claim SOTA superiority.

Primary references:
- Quimb CircuitMPS API: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- Quimb MPS circuit guide: https://quimb.readthedocs.io/en/latest/circuit/circuit-mps.html
- ITensor QN-conserving DMRG: https://docs.itensor.org/ITensorMPS/stable/tutorials/QN_DMRG.html
- ITensor MPS time evolution / TEBD: https://docs.itensor.org/ITensorMPS/stable/tutorials/MPSTimeEvolution.html
- ITensor MPS/MPO gate application and canonicalization: https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html

## Claim ceiling

A passing version of these increments would establish validated complex U(1)-sector-labelled MPS storage, charge-block dense factorization, local U(1) gates, truncation receipts, mixed-canonical gauge checks, a numerically qualified global truncation bound, and a second-order nearest-neighbour real-time stepper. It would not establish a total physical-error bound until Trotter and roundoff terms are bounded, nor SOTA performance, large-scale tensor-network superiority, black-hole evaporation, Hawking radiation, gravitational backreaction or quantum advantage.
