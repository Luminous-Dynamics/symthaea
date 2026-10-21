# Entropy–Causality Benchmark: A Small-System Reference Kernel

**Status:** Research scaffold. This is not an IIT implementation, not a validated phi estimator, and not evidence about consciousness.

## Why this benchmark exists

Symthaea has several quantities historically called phi, including spectral connectivity, HDC similarity-based proxies, and small-system partition calculations. They are not interchangeable. The current integrated-information source maps similarity s through 1 - H2(s), then compares bundled-state and partition scores. Similarity alone does not identify a joint probability distribution, and 1 - H2(s) is not mutual information without additional assumptions.

This is a paradigm shift from optimizing a scalar called phi to building an **evidence ladder**: exact finite-state causal models first, independently implemented reference comparisons second, and scalable approximations only after measured error is known.

## Deliverable

**examples/entropy_causality_reference.rs** is a standard-library-only Rust program. It enumerates every state of a small synchronous Boolean network and its full conditional transition distribution under an independent per-node bit-flip noise model.

Run and test on a machine with the Rust toolchain:

~~~sh
cargo run --example entropy_causality_reference --release
cargo test --example entropy_causality_reference
~~~

The harness intentionally caps exhaustive enumeration at 8 nodes: it evaluates every nontrivial bipartition and every transition row, so the partition count grows exponentially and transition tables grow exponentially in both dimensions. Larger systems require bounded or sampled cuts rather than raising this cap blindly. Its tests are source definitions until a Rust runner executes them. No compilation or test pass is claimed in this change.

## Measures

For a uniform distribution over current states X(t) and transition model P(X(t+1) | X(t)), the program reports:

- **H_next_bits:** Shannon entropy H(X(t+1)), in bits.
- **H_next_given_prev_bits:** conditional entropy H(X(t+1) | X(t)), in bits.
- **I_prev_next_bits:** mutual information I(X(t); X(t+1)) = H(X(t+1)) - H(X(t+1) | X(t)), in bits, under the chosen uniform input distribution.
- **min_cut_sensitivity_tv:** minimum across nontrivial bipartitions of the mean, per-current-state total-variation distance between the original transition row and a diagnostic cut model.

For the cut diagnostic, a target whose rule has an input dependency crossing the cut receives a fair-random unnoised output before the same bit-flip noise is applied. The diagnostic compares the full and cut conditional distributions. It is operationally defined, bounded by 1, and exhaustively computable; **it is not IIT's formal partition operation and must not be called phi**. Results depend on this explicit cut convention and the uniform distribution over current states.

## Falsifiable starter hypothesis

> State entropy alone does not determine temporal information or sensitivity to causal cuts.

The strongest smoke test compares two three-node networks:

1. Three independent self-memory nodes with flip probability 0.5. The next state is maximally entropic (3 bits), while transition uncertainty is maximal, temporal mutual information is zero, and cut sensitivity is zero.
2. A deterministic three-node directed ring. The next state is also maximally entropic (3 bits), while transition uncertainty is zero, temporal mutual information is 3 bits, and cross-cut dependencies make diagnostic cut sensitivity positive.

The same marginal entropy can therefore coexist with very different dynamics. This is a constructive counterexample to any universal claim that entropy alone is an adequate integration score. It does not show that this diagnostic equals IIT phi.

Other fixtures exercise common-driver dependence, feed-forward chains, XOR and AND nonlinearities, noise sweeps, transition normalization, and node-relabeling invariance.

## Next experimental program

### Phase A — qualify the kernel

- Execute all ten Rust tests on the exact branch head.
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

PyPhi's current documentation distinguishes IIT 3.0, IIT 4.0 (2023), and IIT 4.0 (2026). PyPhi 2.0 has a breaking API transition from `Network`/`Subsystem` to `Substrate`/`System`; the revised reference call uses `Substrate` and an explicit `formalism="IIT_3_0"`. The validation dependency is pinned to the available `2.0.0rc4` pre-release plus the EMD extra, so its pre-release status must be visible in every receipt. The benchmark must rerun against a final stable version before use as publication evidence. [Migration guide](https://pyphi.readthedocs.io/en/latest/migration/migration-2.0.html). The target is a declared formalism, not an interchangeable notion of ground truth.

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
8. Publish exact source commit, tool versions, formalism preset, inputs, outputs, and verification receipt for every claimed pass. Queued workflows are not passes.

## How this could outperform the state of the art

Do not claim novelty merely for combining entropy with phi. The promising contribution is a rigorous, reusable test harness for distinguishing uncertainty, dependence, and intervention-sensitive irreducibility with exact small-system controls and explicit estimator boundaries.

The potential advantage is a cleaner route from toy systems to Symthaea's large hyperdimensional dynamical models: every approximation is paired with measured error, known counterexamples, and an independent reference. This could yield practical speedups without sacrificing semantic honesty, and surface dynamical regimes where entropy and causal integration diverge.

## References

- Oizumi, Albantakis, and Tononi (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1003588
- Mayner et al. (2018). PyPhi: A toolbox for integrated information theory. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1006343
- Oizumi et al. (2016). Measuring integrated information from the decoding perspective. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1004654
- Albantakis et al. (2023). Integrated information theory (IIT) 4.0. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1011465
- Zaeemzadeh and Tononi (2024). Upper bounds for integrated information. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1012323
- PyPhi formalism/configuration reference: https://pyphi.readthedocs.io/en/latest/reference/configuration.html
