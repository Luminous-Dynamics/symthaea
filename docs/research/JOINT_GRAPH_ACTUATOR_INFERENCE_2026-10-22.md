# Joint Graph–Actuator Inference Pilot — 2026-10-22

**Status:** exact local path enumeration; embedded self-test and 6/6 unit tests passed. This is a tiny diagnostic prototype, not an external benchmark run or a state-of-the-art claim.

## Question

Can active causal discovery stop treating intervention compliance as known ground truth? Earlier audits found that a learner assuming perfect do(X=x) can assign zero probability to valid observations when actuation is partial. This experiment makes compliance a latent nuisance variable and jointly maintains a posterior over graph and compliance hypotheses.

The design is related to established work on stochastic/off-target interventions; it does not claim to invent unknown-target or unreliable-intervention causal discovery. Its focus is an exact integrated evaluation of joint graph/actuator inference, graph-versus-compliance action information, model-reference falsification, and the cost of a naive perfect-actuator assumption.

## Protocol

Implementation: validation/joint_graph_actuator_inference.py; tests: validation/test_joint_graph_actuator_inference.py.

- Candidate SCMs: a three-node Boolean chain and fork, with known truth tables and unknown graph identity.
- Joint learner hypotheses: both graphs crossed with intervention-compliance probabilities q in {0.50, 0.75, 1.00}, with uniform prior over the six graph-times-q hypotheses.
- Reference family: collider AND and collider OR mechanisms, crossed with the same q grid.
- Held-out worlds: an XOR collider and a reversed chain.
- Actual compliance: 0.50, 0.75, 1.00; mechanism noise: 0.10, 0.25; adaptive action budget: two distinct hard-intervention requests.
- Policies: joint posterior but graph-EIG acquisition; graph-EIG plus 0.5 × compliance-EIG; graph-EIG plus 0.5 × compliance-EIG plus 0.5 × reference JSD; oracle graph-EIG given the true q; and a naive graph learner fixing q=1.
- Outputs: OOD-surprise AUC, graph-information reduction, compliance-information reduction, in-class probability mass assigned zero predictive support, and expected intervention count.

The oracle is an upper-information baseline because q is given to it; it is not an achievable competitor when q is unknown. The reference bank is human-authored and small. Results are exact within the specified finite grid but do not represent a sample from the distribution of real-world causal systems.

## Results

Aggregate rows are averaged over two held-out worlds and all three actual compliance values. AUC 0.50 is chance ranking.

| Noise | Policy | Mean OOD-surprise AUC | Graph information (bits) | Compliance information (bits) | In-class zero-support mass |
|---:|---|---:|---:|---:|---:|
| 0.10 | Joint graph-EIG | 0.8785 | 0.4024 | 0.3083 | 0.0000 |
| 0.10 | Joint graph + compliance | 0.9036 | 0.4024 | 0.3246 | 0.0000 |
| 0.10 | Joint graph + compliance + JSD | 0.9036 | 0.4024 | 0.3246 | 0.0000 |
| 0.10 | Oracle, true q known | 0.6658 | 0.4024 | 0.0000 | 0.0000 |
| 0.10 | Naive q=1 | 0.6391 | 0.3824 | 0.0000 | 0.2215 |
| 0.25 | Joint graph-EIG | 0.7327 | 0.1580 | 0.2506 | 0.0000 |
| 0.25 | Joint graph + compliance | 0.7449 | 0.1580 | 0.2537 | 0.0000 |
| 0.25 | Joint graph + compliance + JSD | 0.7447 | 0.1580 | 0.2537 | 0.0000 |
| 0.25 | Oracle, true q known | 0.6964 | 0.1580 | 0.0000 | 0.0000 |
| 0.25 | Naive q=1 | 0.5999 | 0.1400 | 0.0000 | 0.2025 |

At q=0.50 and noise 0.10, the naive q=1 baseline has 43.95% in-class zero-support mass, whereas joint graph/actuator inference has zero zero-support mass on the declared q grid. The joint balanced policy improves mean OOD AUC from 0.7990 to 0.8250 across the two held-out worlds at this q/noise setting, leaving graph-information reduction unchanged at 0.3026 bits and raising compliance-information reduction from 0.4751 to 0.5029 bits.

At q=0.75 and noise 0.10, joint balanced acquisition improves mean OOD AUC from 0.8867 to 0.9164 and preserves graph information at 0.4024 bits. At high noise (0.25), improvements are smaller and not uniform across q. The JSD add-on is practically identical to balanced graph+compliance acquisition in most cells; it has not shown a separate consistent benefit in this joint-posterior setup.

Surprise AUC is not a graph-recovery metric. The fact that the oracle's AUC can be lower than joint inference is not proof the oracle is worse at graph identification; the oracle policy optimizes graph EIG and is not tuned for OOD ranking. Graph-information reduction, compliance inference, zero support and OOD detection must stay separate.

## Paradigm shift

Represent the causal experiment as a joint latent model:

P(graph, mechanism, actuator reliability, observation | requested action)

rather than assuming the request itself equals the realized intervention. Maintain separate posterior summaries for graph identity, action compliance, and model-set adequacy. Choose experiments to reduce graph uncertainty and actuator uncertainty; apply the falsification objective only behind a validity constraint. If evidence has zero likelihood under every joint hypothesis, fail closed with NOT_IDENTIFIED / ACTION_OR_MODEL_MISMATCH and expand the model, rather than forcing a posterior graph.

The longer-term version should replace scalar per-action compliance q with an action-conditional distribution over actual target sets, dose and off-target effects. This connects to established stochastic intervention work rather than pretending scalar q solves the general problem.

## Research comparisons

- **ACDB (September 2026)** defines a common observe/intervene/submit interface and separates skeleton recovery, DAG recovery and intervention efficiency. It assumes perfect hard interventions and linear-Gaussian SCMs, so it is a useful matched discovery baseline but not a direct numeric comparator for this Boolean reliability pilot. https://arxiv.org/abs/2609.31675
- **CausalArena (September 2026)** includes 1,200 executable SCM specifications across synthetic, semantic-operational and formula-grounded scientific families and observes substantial cross-family ranking shifts. Our six fixed model specifications and two held-out worlds are far narrower. https://arxiv.org/abs/2609.11897
- **Causal Discovery under Off-Target Interventions (AISTATS 2024)** models an attempted intervention as acting on a random subset of vertices, action-dependently. That is richer than the scalar target-compliance model here and should be a required comparator for the next implementation. https://proceedings.mlr.press/v238/choo24a.html
- **TCD-Arena (ICLR 2026)** provides a stress-test framework over 33 time-series causal discovery assumption violations and a much larger evaluation volume. Its setting is not identical, but it establishes a robustness standard worth emulating. https://tcd-arena.github.io/

## Proposed qualification gates (not achieved)

- At least 1,000 seeded SCMs separated by topology, mechanism, intervention reliability and measurement regime.
- Explicit partial, delayed, off-target and unknown-target interventions, including action-dependent distributions over target subsets.
- Compare graph-only, joint graph/actuator, and falsification-aware acquisition to oracle and naive baselines at equal budgets.
- Report SHD, skeleton and directed precision/recall, OOD AUROC/AUPRC, false alarms at a calibrated 5% threshold, power at predeclared OOD severities, q-calibration/Brier or log score, zero-support mass, coverage and intervention count.
- Use disjoint calibration/test worlds, paired confidence intervals clustered by SCM, and family-wise score reporting.
- Only claim superiority when a preregistered primary metric improves with uncertainty intervals excluding zero while the intervention-validity endpoint does not regress.

## Reproduction and provenance

From the repository root:

    python3 validation/joint_graph_actuator_inference.py --self-test
    python3 -m unittest -v validation/test_joint_graph_actuator_inference.py
    python3 validation/joint_graph_actuator_inference.py --source-git-blob 80939b1e17dbd5bd1384824586ea12e9d0cfe08b --output /tmp/joint-graph-actuator.json

- Implementation Git blob: 80939b1e17dbd5bd1384824586ea12e9d0cfe08b; SHA-256 ed49dd785ca5894dc452caf42dddffc2ce85c81695960f1a4387468b4ea6f4bb; 14,455 bytes.
- Test Git blob: 6c464e13deffc18974a930131599eb41da7a5a1f; SHA-256 37a2eef94a69c9f2e7b6487b1901da4098cdd05fa42806527b1d45b5c520e2be; 6 tests.
- Embedded self-test passed; 6/6 tests passed locally against exact GitHub source/test blobs.
- Full 60-row artifact: 27,572 bytes; SHA-256 153e8d61e2928ccc459308273d7c502e8c7b42843d080d9eb6206d34e02509eb; artifact binds implementation Git blob 80939b1e17dbd5bd1384824586ea12e9d0cfe08b.
- Local test evidence is not a GitHub Actions pass; no remote pass is inferred.
