# Abstract Pattern Interaction Benchmark (operational “Platonic space”)

## Purpose

This benchmark asks a narrow, testable question: when patterns are represented as abstract high-dimensional structures, which interactions are preserved by composition, which are destroyed by interference, and how robust is recall after perturbation?

“Platonic space” is an operational metaphor for an abstract pattern space: a representation in which the same relational structure can be studied apart from a particular embodiment. This experiment does **not** assume the existence of a literal metaphysical realm, nor does success establish consciousness, agency, or biological equivalence.

## Included experiments

1. **Role–filler composition.** Bind a role to a filler and unbind it. Bipolar element-wise binding is self-inverse; exact recovery is an algebraic control.
2. **Causal ablation.** Replace binding with superposition, then attempt the same unbinding. The reduction in filler similarity estimates the value of the binding operation relative to this simple ablation.
3. **Antipattern cancellation.** Bundle a pattern with its exact complement and measure whether the result has low similarity to the original instead of preserving a spurious member advantage.
3. **Order sensitivity.** Encode A→B and B→A using permutation and binding. The two sequences should not collapse into the same representation.
4. **Basis / embodiment invariance.** Apply the same seeded coordinate permutation to all patterns. Similarities should be preserved, and consistent coordinate re-encoding should commute exactly with binding.
5. **Noise robustness.** Corrupt a stored pattern at controlled flip rates and retrieve against a fixed prototype library. Report top-1 accuracy and the target-vs-best-competitor similarity margin.
6. **Superposition capacity.** Bundle K stored patterns and compare mean member similarity, maximum distractor similarity, and top-1 member recall as K increases.
7. **Associative capacity.** Bind distinct role/filler pairs, bundle all pairs into one memory trace, query by unbinding one role, and measure target similarity, strongest distractor similarity, and top-1 retrieval as the number of relations increases. This is the most direct abstract-pattern test of whether composition preserves relationships under interference.
8. **Iterative associative cleanup.** Compare nearest-prototype retrieval with a projected softmax-weighted pattern update over the same stored library. The latter is a Hopfield-inspired baseline (softmax attention over stored patterns plus bipolar projection), not an exact replication of a named modern Hopfield model. Report both accuracy and target/distractor margin across the same corruption rates.
9. **Cross-algebra associative capacity.** Compare the current bipolar MAP/BSC-style majority-bundle memory with an HRR implementation using circular convolution, correlation unbinding, and additive superposition, on matched relation counts, trial IDs, candidate-library size and seeded target queries. Report Recall@1, Wilson intervals, target/distractor similarity, elapsed time and payload-only storage. The runner now reports both same-dimension HRR and byte-matched HRR (HRR dimension = MAP dimension / 8 for f64 versus bipolar i8 payloads). It matches candidate-library size, trial count, and capacity schedule; this first comparison still uses different algebra-specific random representations, so it is not a claim of fully identical inputs. This is an initial comparison; it does not yet cover FHRR, tensor-product representations, or an exact modern Hopfield model.

Every random pattern and noise mask is deterministic for the supplied seed. The report exposes random-chance retrieval as a baseline. Use multiple independent seeds for publication-quality comparisons; the included default seed is for reproducibility, not statistical evidence.

## Run

From the repository root:

    cargo test -p symthaea-core pattern_interaction
    cargo run -p symthaea-core --example platonic_pattern_interaction

The smoke mode uses 16,384 coordinates, 64 candidate prototypes, 16 trials, noise rates 5/10/20/30/35/40/45/50%, and bundle/association sizes 1/3/5/9/17/33/49. The stress mode is invoked with --stress and uses 4,096 coordinates, 256 candidates, 128 trials, and capacities 1/7/31/63/127/255 to probe a much higher interference regime. The larger stress trial count narrows single-run binomial intervals relative to the original 32-trial sweep, but is still one predeclared seed; multiple seeds remain mandatory for publication-grade comparisons. Both emit paired MAP-versus-HRR capacity results, but these initial settings do not yet establish matched-byte K_95 or state-of-the-art performance.

## Capacity target: K95 (tested-grid estimate)

The runner now emits `max_tested_k95` for MAP raw retrieval, MAP projected-softmax cleanup, same-dimension HRR, and byte-matched HRR. Here K95 means the **largest capacity in the initial consecutive prefix of the configured sweep** for which every tested point's lower 95% Wilson bound on Recall@1 remains at least 95%. The algorithm stops at the first failed point; a later noisy point cannot requalify a larger capacity. The output says `k95_tested_grid_only=true` deliberately: this is not a claim about untested capacity values or a continuous maximum.

Two important qualifications:

- The smoke run has only 16 trials. Even 16/16 observed successes do not make the lower Wilson bound reach 95%, so smoke-mode K95 may correctly be `NA`. Do not replace that with observed accuracy or omit the row.
- The stress run now uses 128 trials but remains a single fixed seed. It may provide a useful high-interference diagnostic and narrower intervals, but publication-grade K95 needs multiple independent seeds, larger test counts near the 95% boundary, and a budget sweep that accounts for total memory (not only candidate payload).

K95 is an explicit target metric for the next cross-algebra study, not a reported win. Compare methods only at the same operational definition, byte budget, candidate count, noise/composition conditions, and confidence procedure.

## Interpretation and controls

- Exact unbinding and basis equivariance are algebraic invariants, not learning results.
- Nearest-prototype retrieval is intentionally a simple recall baseline, not a learned attractor or autonomous repair mechanism.
- Bundle recall is expected to degrade with increasing bundle size; report the full curve rather than choosing only a favorable K.
- The bundling-ablation comparison is a controlled operator comparison, not evidence that binding is universally superior for every task.
- Use held-out pattern seeds, multiple bundle sizes, and unrelated distractors. Compare the nearest-prototype baseline against the projected-softmax cleanup baseline using the same stored items, seeds, and corruption masks. Do not tune thresholds on the evaluation trials.
- Before claiming an advantage, compare against nearest-neighbor, random-choice, and task-specific baselines at matched dimensionality and compute; report confidence intervals across independent seeds.

## Follow-on biology-inspired tests

This synthetic layer can later supply controlled abstract patterns to the Cell Foundry equifinality work: keep target constraints fixed while changing the embodiment mapping, communication graph, or damage pattern. Measure whether the system reaches the same functional target across routes. That would test functional equifinality in a computational model, not validate the model against living tissue; biological claims require comparison to suitable published or experimentally collected data.

## Falsification / stop conditions

The invariance tests should be exact for the implemented algebra. Retrieval claims fail if noisy top-1 accuracy does not outperform the random baseline across held-out seeds. A proposed new mechanism should be dropped or redesigned if it does not outperform the simpler baseline on predefined metrics, or if it merely increases compute without improving recovery, generalization, or uncertainty calibration.
