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

Every random pattern and noise mask is deterministic for the supplied seed. The report exposes random-chance retrieval as a baseline. Use multiple independent seeds for publication-quality comparisons; the included default seed is for reproducibility, not statistical evidence.

## Run

From the repository root:

    cargo test -p symthaea-core pattern_interaction
    cargo run -p symthaea-core --example platonic_pattern_interaction

Default experiment dimensions are 16,384 bipolar coordinates, 64 prototypes, 16 trials per condition, noise rates 5/10/20/30%, and bundle sizes 1/3/5/9/17. Override these in research code to study scaling and uncertainty.

## Interpretation and controls

- Exact unbinding and basis equivariance are algebraic invariants, not learning results.
- Nearest-prototype retrieval is intentionally a simple recall baseline, not a learned attractor or autonomous repair mechanism.
- Bundle recall is expected to degrade with increasing bundle size; report the full curve rather than choosing only a favorable K.
- The bundling-ablation comparison is a controlled operator comparison, not evidence that binding is universally superior for every task.
- Use held-out pattern seeds, multiple bundle sizes, and unrelated distractors. Do not tune thresholds on the evaluation trials.
- Before claiming an advantage, compare against nearest-neighbor, random-choice, and task-specific baselines at matched dimensionality and compute; report confidence intervals across independent seeds.

## Follow-on biology-inspired tests

This synthetic layer can later supply controlled abstract patterns to the Cell Foundry equifinality work: keep target constraints fixed while changing the embodiment mapping, communication graph, or damage pattern. Measure whether the system reaches the same functional target across routes. That would test functional equifinality in a computational model, not validate the model against living tissue; biological claims require comparison to suitable published or experimentally collected data.

## Falsification / stop conditions

The invariance tests should be exact for the implemented algebra. Retrieval claims fail if noisy top-1 accuracy does not outperform the random baseline across held-out seeds. A proposed new mechanism should be dropped or redesigned if it does not outperform the simpler baseline on predefined metrics, or if it merely increases compute without improving recovery, generalization, or uncertainty calibration.
