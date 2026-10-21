# Entropy–Causality Benchmark: A Small-System Reference Kernel

**Status:** Research scaffold. This is not an IIT implementation, not a validated phi estimator, and not evidence about consciousness.

## Why this benchmark exists

Symthaea has several quantities historically called phi, including spectral connectivity, HDC similarity-based proxies, and small-system partition calculations. They are not interchangeable. The current integrated-information source maps similarity s through 1 - H2(s), then compares bundled-state and partition scores. Similarity alone does not identify a joint probability distribution, and 1 - H2(s) is not mutual information without additional assumptions.

This is a paradigm shift from optimizing a scalar called phi to building an **evidence ladder**: exact finite-state causal models first, independently implemented reference comparisons second, and scalable approximations only after measured error is known.

## Deliverable

**examples/entropy_causality_reference.rs** is a standard-library-only Rust program. It enumerates every state of a small synchronous Boolean network and its full conditional transition distribution under an independent per-node bit-flip noise model.

Run and test on a machine with the Rust toolchain:

~~~sh
mkdir -p results
git rev-parse HEAD
cargo run --example entropy_causality_reference --release > results/entropy_causality_reference.tsv
sha256sum results/entropy_causality_reference.tsv
cargo test --example entropy_causality_reference
~~~

The harness intentionally caps exhaustive enumeration at 8 nodes: it evaluates every nontrivial bipartition and every transition row, so the partition count grows exponentially and transition tables grow exponentially in both dimensions. Larger systems require bounded or sampled cuts rather than raising this cap blindly. Its tests are source definitions until a Rust runner executes them. No compilation or test pass is claimed in this change.

## Measures

For a supplied distribution over current states X(t) and transition model P(X(t+1) | X(t)), the program reports:

- **H_prev_bits:** Shannon entropy H(X(t)) under the declared input-state ensemble.
- **H_next_bits:** Shannon entropy H(X(t+1)), in bits.
- **H_next_given_prev_bits:** conditional entropy H(X(t+1) | X(t)), in bits.
- **I_prev_next_bits:** mutual information I(X(t); X(t+1)) = H(X(t+1)) - H(X(t+1) | X(t)), in bits, under that ensemble.
- **min_cut_sensitivity_tv:** minimum across nontrivial bipartitions of the prior-weighted mean total-variation distance between original and diagnostic cut transition rows.

Every TSV row now records both versioned canonical rule/noise and prior specifications; no ensemble is implicit in the emitted artifact. The public helper `measure()` is a uniform-prior convenience wrapper around `measure_with_state_distribution()`, which rejects invalid weights instead of silently renormalizing them.

For the cut diagnostic, a target whose rule has an input dependency crossing the cut receives a fair-random unnoised output before the same bit-flip noise is applied. The diagnostic compares the full and cut conditional distributions. It is operationally defined, bounded by 1, and exhaustively computable; **it is not IIT's formal partition operation and must not be called phi**. Results depend on this explicit cut convention and the uniform distribution over current states.

## Falsifiable starter hypothesis

> State entropy alone does not determine temporal information or sensitivity to causal cuts.

The strongest smoke test compares two three-node networks:

1. Three independent self-memory nodes with flip probability 0.5. The next state is maximally entropic (3 bits), while transition uncertainty is maximal, temporal mutual information is zero, and cut sensitivity is zero.
2. A deterministic three-node directed ring. The next state is also maximally entropic (3 bits), while transition uncertainty is zero, temporal mutual information is 3 bits, and cross-cut dependencies make diagnostic cut sensitivity positive.

The same marginal entropy can therefore coexist with very different dynamics. This is a constructive counterexample to any universal claim that entropy alone is an adequate integration score. It does not show that this diagnostic equals IIT phi.

A second counterexample holds the transition rules fixed but changes the state ensemble. For a deterministic 3-node ring, a uniform input prior gives H(X(t+1)) = 3 bits and I(X(t); X(t+1)) = 3 bits. Concentrating prior mass 0.7 and 0.3 on two input states yields H(X(t)) = H(X(t+1)) = 0.881291 bits and temporal mutual information of 0.881291 bits; transition entropy remains zero. This value is a direct analytic calculation, not Rust-run output. The lesson is that an entropy/dependence number is a property of **dynamics plus the stated state distribution**, not dynamics alone.

Other fixtures exercise common-driver dependence, feed-forward chains, XOR and AND nonlinearities, noise sweeps, prior-distribution sensitivity, transition normalization, and node-relabeling invariance.

## Next experimental program

### Phase A — qualify the kernel

- Execute all fifteen Rust tests on the exact branch head.
- Add property-based checks for stochastic-row normalization, nonnegativity, entropy bounds, deterministic/noise extremes, and permutation invariance.
- Independently implement the tiny transition-model calculations or use a separately written analytical enumerator; cross-check every TPM row and metric, not just summary statistics.
- Freeze emitted rows as a versioned deterministic CSV/JSON artifact with schema, input-model digest, source SHA, and run receipt. Do not hand-edit measurements into documentation.

### Phase B — benchmark against references

| Baseline | Role | Limitation |
|---|---|---|
| Shannon entropy and entropy rate | Uncertainty / temporal predictability | Not integration |
| Pairwise mutual information | Statistical dependence | Can miss higher-order synergy and common-cause structure |
| Transfer entropy / conditional mutual information | Directed predictive dependence | Predictiveness is not automatically causal influence |
| Spectral connectivity lambda-2 | Graph connectivity and mixing | Not IIT phi |
| Causal-cut sensitivity in this example | Transparent transition-model intervention diagnostic | Not IIT; the cut convention is not theory-neutral |
| Symthaea HDC entropy proxy | Fast approximation candidate | Must be validated against an independent target |
| PyPhi, pinned to a named formalism | Reference for small systems under that formalism | Formalism/configuration-dependent |

PyPhi's current documentation distinguishes IIT 3.0, IIT 4.0 (2023), and IIT 4.0 (2026). PyPhi 2.0 has a breaking API transition from `Network`/`Subsystem` to `Substrate`/`System`; the revised reference call uses `Substrate` and an explicit `formalism="IIT_3_0"`. The validation dependency is pinned to `pyphi[emd]==2.0.0` (stable) and explicitly selects `IIT_3_0`; the version and formalism must still be recorded in every receipt. Pinning a stable dependency does not qualify the benchmark: the same-model Rust adapter, executed tests, and exact-head evidence remain mandatory. [Migration guide](https://pyphi.readthedocs.io/en/latest/migration/migration-2.0.html). The target is a declared formalism, not an interchangeable notion of ground truth.

The legacy path **validation/pyphi_crossvalidation.py** has now been changed to fail closed: no topology-name estimates are substituted, unavailable scores stay missing, and an unqualified run exits nonzero. Its PyPhi reference model is explicitly a noisy OR-neighbor TPM, with a digest reported per topology. This reference fixture is **not yet the same model as the Rust Boolean-network kernel**, and the Symthaea adapter intentionally raises a blocked status until a shared canonical-model interface exists. No paired accuracy or correlation result can be claimed yet.

IIT is not simply H(whole) minus the sum of H(parts). Formal calculations depend on the selected IIT version, state, transition probability matrix, mechanism/purview repertoires, partition scheme, and distance/normalization choice. This kernel is intentionally a pre-reference diagnostic only.

### Phase C — synthetic suite and adversarial controls

The starter kernel now emits a TSV header plus each case's canonical Boolean rule specification and noise level. Save the complete stdout artifact, then hash the exact bytes (for example with `sha256sum`) and record the exact Git head; this hashes the actual output rather than using a homemade pseudo-hash. A shared model schema and per-model SHA-256 receipts are still required before the PyPhi and Rust rows can be joined.

Create a frozen small-system suite across node counts 2–8 where tractable, including:

- Independent units with matched marginal entropies.
- Common-driver systems and randomized common causes.
- Feed-forward chains, recurrent loops, stars, rings, XOR parity, AND, and mixed nonlinear networks.
- Deterministic, noisy, and near-random transition kernels.
- Balanced/unbalanced cuts, reducible subsystems, duplicated nodes, and redundant signals.
- State-distribution changes with transition structure held fixed.
- Transition-structure changes with marginal entropy held approximately fixed.
- Node permutation and equivalent Boolean-encoding controls.
- Negative controls in which observational dependence remains but causal cross-unit links are removed.

Use paired random seeds and held-out topology families. Publish seed schedules and include the rule set, state distribution, noise, metric version, and digest with every case.

### Phase D — make a scalable estimator earn its status

Only after small-system results are qualified should HDC summaries or sparse/spectral approximations be evaluated at increasing sizes. Publish an **accuracy–cost frontier**: latency, peak memory, approximation error, partition recovery, stability across seeds, sensitivity to representation changes, and failure rates. A fast proxy that disagrees with its reference is a different metric, not an accelerated phi implementation.

## Proposed acceptance gates

These are research targets, not current results:

1. Every finite-state row is nonnegative and sums to 1 within 1e-12.
2. Deterministic, independent-noise, and symmetry fixtures satisfy analytical invariants.
3. Node relabeling does not change results beyond 1e-12 under the same arithmetic path.
4. Independent implementations agree on TPM entries and Shannon quantities within 1e-12 for tiny systems.
5. No failed reference call can produce a number presented as measured.
6. On held-out cases, report Pearson r, Spearman rho, MAE, RMSE, rank reversals, confidence intervals, and per-family breakdown. Use r and rho >= 0.90 only as an initial target, never as the only pass criterion.
7. Report false-positive rates on negative controls. Do not assert consciousness from any scalar score.
8. A validator PASS requires all cases paired, equal per-case reference/approximation model digests and prior digests, an explicit shared-model attestation, a stable (non-RC) PyPhi version, the declared formalism, Pearson and Spearman > 0.90, and >= 90% of cases within 10% relative error.
9. Publish exact source commit, tool versions, formalism preset, inputs, outputs, and verification receipt for every claimed pass. Queued or skipped workflows are not passes.

## SOTA and industry comparison protocol

These methods are not all competing estimators of one quantity. Compare them on a common model and report the target each one actually answers.

| Baseline family | Correct comparison question | Pass condition |
|---|---|---|
| Shannon entropy / entropy rate | Does the system have uncertainty or temporal predictability? | Analytic invariants and exact probability checks |
| Pairwise MI / conditional MI / transfer entropy | Does statistical or directed predictive dependence exist? | Known common-cause, redundancy, and XOR counterexamples |
| Spectral connectivity (lambda-2) | Does the wiring graph remain connected / mix quickly? | Correct graph-theoretic values; no IIT label |
| This causal-cut diagnostic | How sensitive is this explicit TPM to the declared cut operation? | Invariance, negative controls, and cross-implementation agreement under the same cut definition |
| PyPhi 2.0 under IIT_3_0, IIT_4_0_2023, and IIT_4_0_2026 | What does each declared IIT formalism calculate for this substrate and state? | Version-pinned outputs, canonical TPM digest, exact state/formalism, and independent receipts |
| Symthaea HDC approximation | How closely does the proxy approximate a declared reference target? | Held-out error distribution plus speed/memory savings at matched accuracy |

PyPhi 2.0 makes these distinctions operational: it has explicit per-call formalisms, and IIT 3.0, IIT 4.0 (2023), and IIT 4.0 (2026) can return different values for the same substrate and state. A correlation table that mixes formalism versions or models is not a valid SOTA comparison. [PyPhi migration guide](https://pyphi.readthedocs.io/en/latest/migration/migration-2.0.html) · [Formalism configuration](https://pyphi.readthedocs.io/en/latest/reference/configuration.html).

## How this could outperform the state of the art

Do not claim novelty merely for combining entropy with phi. The candidate contribution is a **fail-closed measurement and approximation benchmark** that makes metric semantics, causal models, state distributions, partition conventions, and reference provenance explicit.

After an identical model and target are wired across both implementations, aim for the following stretch targets—not current results:

- **Reference correctness:** all tiny-model TPM rows agree within 1e-12; all documented analytical invariants pass; no fallback-derived measurements.
- **Approximation fidelity:** held-out Pearson and Spearman >= 0.95, median relative error <= 5%, 95th-percentile relative error <= 10% where the reference is nonzero, and 100% explicit accounting of failed cases. Report MAE/RMSE and confidence intervals too.
- **Adversarial robustness:** zero false-positive "integration" claims on the frozen negative-control set; no target-family leakage between tuning and evaluation; no unexplained rank reversal on the predeclared critical contrasts.
- **Efficiency:** demonstrate >= 10x lower wall-clock or memory cost at a matched error budget, measured with identical hardware, concurrency, cache policy, and model digest. Otherwise publish the accuracy–cost frontier without claiming superiority.
- **Reproducibility:** exact source SHA, environment lock, input-model digest, formalism/state/config, stdout artifact digest, and independently reproduced receipts for every pass.

Until those gates are met, the correct product is a transparent experimental diagnostic—not a faster IIT estimator. The current Rust kernel and PyPhi reference fixture still use different model representations, and the cross-validation adapter remains intentionally blocked.


## References

- Oizumi, Albantakis, and Tononi (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1003588
- Mayner et al. (2018). PyPhi: A toolbox for integrated information theory. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1006343
- Oizumi et al. (2016). Measuring integrated information from the decoding perspective. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1004654
- Albantakis et al. (2023). Integrated information theory (IIT) 4.0. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1011465
- Zaeemzadeh and Tononi (2024). Upper bounds for integrated information. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1012323
- PyPhi formalism/configuration reference: https://pyphi.readthedocs.io/en/latest/reference/configuration.html
