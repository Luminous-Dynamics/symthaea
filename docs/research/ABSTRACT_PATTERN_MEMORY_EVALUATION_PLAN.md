# Symthaea Abstract Pattern / Associative Memory Evaluation Plan

## Research claim we are testing

Symthaea should not claim that its high-dimensional patterns inhabit a literal metaphysical realm. “Platonic space” is an operational term for an abstract state space in which compositional structure, relations, transformations, and invariants can be tested separately from one specific embodiment.

The claim worth testing is narrower and stronger:

> A goal-conditioned, compositional pattern system can preserve relational information under interference and then recover useful function after its state, context, or physical realization changes, with measurable advantages over simpler associative-memory baselines at matched memory and compute budgets.

This is a research hypothesis, not an established result.

## Baseline families to include

1. **Random choice and exact lookup.** Sanity floor and an oracle ceiling where an oracle is appropriate. Never compare to random alone.
2. **Nearest prototype / cosine or Hamming retrieval.** The simple baseline included in the current experiment.
3. **Binary VSA/HDC (implemented first).** XOR or elementwise bipolar binding, permutation for sequence, and majority bundling.
4. **HRR / real-valued VSA (initial implementation added in this PR).** Circular convolution and correlation unbinding are tested against the same stored-relation capacity schedule. The runner now includes both dimension-matched HRR and a byte-matched HRR condition (HRR dimension = MAP dimension / 8 for f64 versus bipolar i8 payloads). It matches candidate-library size, trial count and relation-count schedule. Algebra-specific random patterns are not byte-identical input objects, so later comparisons should additionally define task-level paired seeds and held-out relation structures before claiming strict method superiority. Broader HRR tasks still need parity tests and baseline comparisons.
5. **MAP and, where useful, FHRR / tensor-product representations.** These test whether results are specific to a chosen vector algebra.
6. **Modern Hopfield / attention-style associative retrieval.** Use a faithful, separately validated reference implementation before making a “beats Hopfield” claim. The current projected-softmax bipolar cleanup is explicitly only a Hopfield-inspired baseline; it is not an exact reproduction of a named paper.
7. **Kanerva-style sparse distributed memory.** A relevant alternative for noisy address-based recall.
8. **For regeneration tasks:** graph diffusion, a central target controller, local-only recovery, and the existing coupling/homing mechanisms as separately ablated models.

Use trusted implementations or reproduce published equations with verification tests. Pin versions and settings. Equalize stored items, dimensions, precision, memory budget, and available compute wherever possible; report unmatched-budget results separately.

## Benchmark matrix

| Capability | Test | Primary metrics | Critical control |
|---|---|---|---|
| Algebraic correctness | bind/unbind, inverse, permutation, basis change | exactness, invariant error | mathematically generated expected result |
| Relational memory | retrieve filler by role after bundling K role/filler pairs | Recall@1/5, target–distractor margin | same roles, fillers, seed, and stored memory across methods |
| Compositional generalization | hold out role/filler combinations and relation chains | exact match, unseen-composition accuracy | split by combination, not merely random examples |
| Noise tolerance | sweep coordinate flips and erasures from 0% through failure | Recall@k, margin, area under the noise–accuracy curve | nearest-prototype and random baselines |
| Interference/capacity | increase number of stored associations and correlated distractors | capacity at 95% Recall@1, collision rate | predeclare retrieval and failure thresholds |
| Attractor behavior | perturb a clean memory, apply iterative retrieval, and record the trajectory | recovery rate, steps to convergence, basin estimates, energy/score monotonicity where defined | verify the stated energy/update relationship for each method |
| Goal-preserving repair | remove network nodes, edges, memory shards, or alter the embodiment mapping | functional success, recovery cost, time, catastrophic failure rate | local-only, centralized, diffusion, and full system |
| Cross-structure invariance | rename symbols, permute coordinates, rewire topology, change dimension | performance drop under each transformation | train/configure on one family; test an unseen family |
| Efficiency | same workload and target recall under controlled hardware | latency distribution, throughput, peak memory, bytes per stored item, energy when directly measured | warmup, repeated runs, pinned revision, hardware and toolchain recorded |
| Reliability | held-out seeds and adversarial cases | confidence intervals, calibration, worst-decile performance | publish every seed and every failure |

## Pre-registered stretch goals (targets, not measured achievements)

- **Correctness:** exact pass for all algebraic invariants and all malformed-input rejection tests.
- **Noiseless relational retrieval:** at least 99% Recall@1 on the held-out set at a fixed capacity chosen before evaluation.
- **Noisy relational retrieval:** at least 95% Recall@1 at 10% independent bit flips for the stated dimension/capacity, while reporting the full curve through failure.
- **Capacity:** establish K_95 (largest number of stored relations achieving at least 95% Recall@1) and compare K_95 at equal bytes, not just equal vector dimension. Improve K_95 over the strongest baseline by a preregistered 20% before using “better than baseline” language.
- **Generalization:** hold-out-composition and held-out-topology results must remain above the task-specific success floor; no claims from in-distribution random splits alone.
- **Efficiency:** demonstrate a Pareto improvement: better recall for the same measured memory and latency budget, or equivalent recall with materially lower measured cost. Do not use an unmeasured energy estimate as a result.
- **Regeneration:** show a replicated improvement in functional success or integrated error over all simpler baselines, with confidence intervals and ablation evidence that the proposed mechanism causes the improvement.

These are milestones for this research program, not statements about current Symthaea performance. If the system misses one, record it and tune only on development seeds.

## Experimental discipline

- Separate development seeds from held-out evaluation seeds.
- Predeclare primary metrics, pass thresholds, dimensions, memory limits, and treatment of timeouts.
- Report means, medians, 95% confidence intervals, worst-decile outcomes, and all negative results.
- Use paired trials and bootstrap intervals or an appropriate paired test where applicable.
- Do not pool different tasks into a single opaque “intelligence score.”
- Keep algebraic invariants separate from learned generalization and from biological-model validation.
- The same random seed does not prove a fair comparison if implementations use different seed streams; persist actual pattern IDs / bytes for cross-method paired experiments.

## Industry and SOTA comparison

The VSA field contains multiple representation families, not one benchmarked system. Use the IEEE HDC/VSA task-force overview and current surveys to keep the comparison set current; include conventional nearest-neighbor and attention/Hopfield baselines on matching tasks. A useful initial source set:

- IEEE CIS Task Force on HDC/VSA: https://cis.taskforce.ieee.org/hdcvsa/
- Kleyko et al., *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II*: https://doi.org/10.1145/3558000
- Bhattacharjee & Martin, *Accuracy and capacity of modern Hopfield networks with synaptic noise*, Physical Review E 112, 035313 (2025): https://doi.org/10.1103/nwbb-n1bc
- Nunes & Barriga, *Bioelectricity in Morphogenesis*, Annual Review of Cell and Developmental Biology 41 (2025): https://doi.org/10.1146/annurev-cellbio-101323-032747
- Levin, *The Multiscale Wisdom of the Body*, BioEssays (first published 2024): https://doi.org/10.1002/bies.202400196

Literature equations and claims must be checked against the full papers before a replication or SOTA claim. A survey is for finding baselines, not evidence that a particular implementation is superior.

## Decision rule

Advance a mechanism only when the same-head code, tests, and recorded experiment demonstrate a statistically credible improvement on at least one preregistered primary metric without unacceptable regressions elsewhere. Otherwise keep the simplest baseline that meets the requirement and document the negative result.
