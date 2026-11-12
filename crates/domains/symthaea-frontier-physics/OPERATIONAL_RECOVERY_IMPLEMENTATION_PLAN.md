# Operational recovery and scalable quantum-information implementation plan

**Status:** research implementation plan and benchmark contract; QSVT-Petz circuits and a U(1)-preserving MPS/TEBD backend are not implemented by this change.

## Current verified boundary

The current Symthaea branch contains exact finite-dimensional density-matrix tools, unconstrained and fixed-charge Page references, seeded scrambling/transport experiments, ordinary Petz recovery, rotated-Petz recovery, and Petz-optimality diagnostics. The current operational Hayden–Preskill toy channel is small: a message qubit \(M\) and a black-hole/early-radiation qubit pair \(B,R\), scrambled by a seeded \(U(4)\). The decoder receives only \(A=E\otimes R\) (dimension four); \(B'\) is traced out and is never supplied to the decoder.

The rotated-Petz source explicitly states that it is a decoder-family candidate, not an optimality claim or an efficient physical circuit. The returned sample reports entanglement fidelity, modular-power and trace-preservation residuals, dimensions and Kraus count; it does not report a gate-level circuit cost.

The existing tensor-network fixture already specifies charge-resolved Schmidt truncation, per-truncation discarded-weight receipts, conservative error envelopes, dense overlap and matched-accuracy benchmarking. However, the current tensor_networks.rs module is only a pedagogical state/parameter-count skeleton; it does not implement complex MPS tensors, two-site gates, charge-block SVD or TEBD. Do not mark the TN-0 through TN-3 contract implemented until those operations exist and pass exact dense overlap tests.

## Paradigm shift: from recovery map to operational decoder

A recovery channel can exist mathematically and still be infeasible to construct or execute. The research objective therefore becomes a four-way comparison on identical channels and scrambler seeds:

1. Ordinary and rotated Petz recovery: channel-level baselines.
2. Exact optimal recovery on tiny instances: an SDP or an independently certified optimality theorem/bound, not a finite modular-parameter grid.
3. Polar/QSVT approximate Petz: a gate-oriented implementation that approximates the polar sign transform.
4. Hardware/noise-aware execution: a later, explicitly separated layer; classical simulation is never reported as a hardware result.

The October 14, 2026 preprint *Petz Recovery through a Single Polar Transformation for Approximate Quantum Error Correction* proposes a QSVT construction of approximate Petz recovery without postselection. It gives a channel-specific spectrum-weighted loss and an upper bound on the diamond distance between the approximate and ideal Petz encoding-recovery channels. This is a preprint, not a peer-reviewed benchmark standard: https://arxiv.org/abs/2610.20515.

For an odd QSVT polynomial \(p_\theta(x)\) of degree at most \(2k+1\), scaled singular values \(x_i=\lambda_i/\alpha\), and the paper's normalization convention, track the spectrum-weighted loss

\[
\mathcal I^\theta=1-\left|\sum_i\lambda_i^2p_\theta(x_i)\right|^2
\]

and the corresponding bound

\[
\Delta^\theta\leq\sqrt{\|\sigma^{-1}\|_\infty\mathcal I^\theta}.
\]

The benchmark must verify the precise normalization and hypotheses against the pinned preprint equations before using these formulas in code. This bound is not a substitute for an independently computed channel distance on small cases.

## Decoder-access firewall

Every run must state the input/access set and the hidden environment. The decoder is given only the declared accessible radiation \(A\). It must not receive:

- the remaining black-hole register \(B'\);
- the full scrambler unitary as an execution-time input unless it is explicitly provided through a declared oracle/block-encoding interface;
- a purified global state or hidden amplitudes unavailable to the decoder;
- simulator-only labels or seed material that would reveal the hidden environment.

The simulator may use the full channel to construct a reference oracle and certify the experiment, but that reference must be sandboxed outside the decoder interface. Add negative controls that deliberately leak \(B'\), the global state or hidden random seed and require the access audit to fail.

## QSVT-Petz implementation ladder

| ID | Work | Acceptance evidence |
|---|---|---|
| QD-0 | Freeze finite channel and exact Kraus convention | Channel CPTP checks, register order, channel digest, decoder access declaration |
| QD-1 | Build stacked Petz isometry and singular spectrum | Independent small-matrix SVD, support-rank and normalization checks |
| QD-2 | Construct a block-encoding | Explicit normalization \(\alpha\), query model, controlled reflections and ancillas |
| QD-3 | Implement odd QSVT phase sequence | Polynomial parity/boundedness, angle digest, degree \(2k+1\), singular-value response |
| QD-4 | Extend recovery isometry without postselection | Complete isometry/channel trace preservation and no discarded-success branch |
| QD-5 | Match ideal and approximate recovery | Entanglement fidelity, Choi/process error, small-instance diamond norm or certified upper/lower bounds |
| QD-6 | Compare decoder families | Ordinary Petz, rotated-Petz \(t\)-grid/optimization, QSVT-Petz, exact optimum where independently available |
| QD-7 | Resource qualification | Oracle queries, QSVT iterations, polynomial degree, ancillas, logical gate counts/depth, two-qubit count, T count, memory and runtime |
| QD-8 | Ensemble and scaling | Frozen common random seeds, confidence intervals, spectrum/conditioning stratification, no post-selection of favorable channels |

### Required resource report

For every QSVT angle sequence and channel, report:

- smallest nonzero singular value, support rank, condition number and block-encoding normalization;
- QSVT iteration count \(k\), polynomial degree, phase-angle sequence digest and number of block-encoding/oracle calls;
- ancilla count, controlled-reflection count, logical one- and two-qubit gate counts, depth and T-count for a declared decomposition;
- ideal logical circuit cost separately from any device-calibrated physical estimate;
- no-postselection trace-preservation residual; if an optional postselected construction is studied, report success probability and expected repetitions;
- entanglement fidelity and channel distance at matched resource budgets.

Do not equate QSVT iteration count with physical gate count. Do not claim an efficient decoder solely because the exact Kraus map is small.

## Exact optimum and falsification

For small channel dimensions, construct an independent recovery-optimization oracle. If the Petz theorem's exact hypotheses apply, use the theorem's necessary-and-sufficient criterion and verify all preconditions. When they do not apply, compare with an SDP over recovery Choi matrices subject to complete positivity and trace preservation. The SDP implementation must be independent from the Rust recovery implementation, use frozen channel fixtures, and report primal feasibility, dual feasibility and a primal-dual gap.

A rotated-Petz grid is not an exact optimum oracle. A high entanglement fidelity is not proof of optimality. A QSVT approximation to Petz cannot exceed the Petz map's performance unless a different recovery family or optimization is introduced.

## U(1)-preserving MPS/TEBD work remains a separate, mandatory scale path

The existing tensor-network contract should be implemented rather than replaced. Required architecture:

1. Complex MPS tensors with declared left/right bond dimensions and physical dimension two.
2. Explicit integer charge labels on virtual bonds and charge-compatible tensor blocks.
3. Nearest-neighbour two-site gate application for \(U(1)\) blocks \(1\oplus U(2)\oplus1\).
4. Two-site SVD split independently within conserved charge blocks; no mixing of charge sectors.
5. Per-bond/per-layer discarded squared Schmidt weight before any optional normalization.
6. A conservative accumulated state-vector error envelope and reduced-state trace-distance envelope; entropy errors need a dimension-dependent continuity bound or dense overlap, not a casual inference from trace distance.
7. Frozen gate tensors and seeds shared with the dense reference; exact dense overlap at small sizes.
8. MPS checkpoints for entropy, density profile, bond current, operator influence (when an MPO implementation exists), bond growth, memory and wall time.

Start with N=8, 10, 12 dense overlap; then N=16, 24, 32, 48, 64 with controlled bond-cap and cutoff sweeps. Compare with pinned Quimb CircuitMPS/TEBD or ITensorMPS only at matched gates, seeds, observables and accuracy. The 2026 monitored-circuit paper demonstrates a useful validation pattern: an untruncated MPS is first compared with exact statevectors and then larger-system results include explicit discarded-weight accounting. It is a pattern to reproduce, not evidence that this repository's MPS backend exists: https://pmc.ncbi.nlm.nih.gov/articles/PMC13703094/.

## External benchmark positioning

- A 2025 Phys. Rev. Research experiment demonstrated Hayden–Preskill recovery and OTOC protocols on 20-qubit trapped-ion processors. This is a hardware benchmark, not a fair direct comparison to the current four-qubit classical channel simulation: https://doi.org/10.1103/PhysRevResearch.7.023032.
- The 2025 PRL gives a necessary-and-sufficient Petz-optimality condition with a useful commutator specialization in commuting cases. Use it to label theorem-qualified instances, not as a universal Petz-optimality claim: https://doi.org/10.1103/PhysRevLett.134.200602.
- The 2026 QSVT-Petz preprint adds an explicit circuit and spectral-conditioning target. Reproduce its stated examples and compare at matched fidelity/resource budgets before claiming an implementation advantage.
- Quimb/ITensor are specialist tensor-network baselines. Compare accuracy, truncation, conserved charge, runtime and memory on identical circuits—not raw throughput from unmatched tasks.

## Proposed qualification targets

These are prospective gates to calibrate, not achieved performance:

- Exact channel/Kraus and decoder-isometry residuals: \(10^{-10}\) for well-conditioned small fixtures.
- QSVT no-postselection trace-preservation residual: \(10^{-9}\) initially, then tightened against precision studies.
- Approximate-versus-ideal Petz channel error: demonstrate a monotone envelope against QSVT degree on preregistered channels; target \(\Delta\leq10^{-3}\) where the singular spectrum permits it.
- Exact small-instance oracle: report primal-dual SDP gap \(\leq10^{-8}\), or a theorem-qualified result with all hypotheses and residuals exposed.
- MPS overlap: reduced-state trace distance \(\leq10^{-6}\) on declared dense-overlap fixtures; no extrapolation beyond measured bond-cap coverage.
- Charge and norm residuals: \(10^{-10}\) on exact unitary circuits; truncation-induced deviation separately budgeted.
- Ensemble reporting: at least 64 frozen common seeds per channel family/size when feasible; publish confidence intervals and every attempted instance.
- Industry comparisons: report fidelity/error at equal gate, depth, query, memory and runtime budgets; a single best seed never establishes superiority.

## Negative controls

- Provide the decoder with \(B'\), hidden global amplitudes or an undeclared seed.
- Use a postselected branch while reporting a deterministic trace-preserving decoder.
- Omit the block-encoding normalization \(\alpha\) or change it without recomputing the phase sequence.
- Compare different seed schedules across decoder families.
- Compare only average fidelity while hiding worst-case channels or poor conditioning.
- Apply the commuting Petz criterion when its hypotheses fail.
- Treat the rotated-Petz parameter grid as an exact optimum.
- Count QSVT iterations as physical gates.
- Drop charge labels during MPS truncation.
- Omit discarded weight or report an exact result after the bond cap saturates.
- Compare MPS and dense runtimes at unequal error.

## Claim ceiling

A passing QSVT/SDP benchmark would qualify the declared finite-dimensional channels and decoder circuits. A passing U(1)-MPS/TEBD benchmark would qualify the declared finite spin-chain circuits. Neither alone demonstrates microscopic gravitational evaporation, a Page island/QES calculation, a black-hole interior reconstruction theorem, or hardware-scale quantum advantage.
