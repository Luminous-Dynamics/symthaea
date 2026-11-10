# IG-CAUSAL-PRIMITIVES-001 — deterministic causal-reasoning contract

**Status:** protocol and synthetic known-answer fixtures authored; model trials not run.
**Owner:** [Symthaea #11505](https://github.com/Luminous-Dynamics/symthaea/issues/11505).
**Parent:** [IG-CONTAGION-001 #11320](https://github.com/Luminous-Dynamics/symthaea/issues/11320).

## Why this track exists

A model can describe a plausible cause, propagate a simple equation, and still fail when evidence arrives or a narrative changes who appears to have acted. One causal-reasoning score hides these different error modes.

CausalPrism (Findings EMNLP 2026) separates **Causal Prior**, **Forward State Tracing**, **Belief Updating**, and **Agency Sensitivity**. In its reported 14-model study, Belief Updating was the main bottleneck (best reported accuracy 0.613), while Causal Prior and Forward State Tracing were near ceiling in the tested setup. These are paper-reported results, not a Symthaea baseline. [CausalPrism paper](https://aclanthology.org/2026.findings-emnlp.1294/).

The benchmark therefore reports four independent primitive vectors and does not create an aggregate causal score.

## Paradigm shift

Move from plausible explanation to **intervention-grounded, revision-auditable causal competence**:

1. Recover causal structure rather than confuse correlation with direction.
2. Compute consequences of do(X=x) rather than substitute observational conditioning.
3. Update hypotheses when evidence warrants it, while retaining ambiguity when it does not.
4. Remain stable to actor, status or grammatical framing when the underlying mechanism is unchanged.

A model explanation is not an oracle. The Rust qualifier computes expected outcomes from frozen structural equations and never reads a candidate model rationale.

## v0.1 scope

The fixture pack contains **48 deterministic tasks**: 12 per primitive, balanced four per primitive × three domains.

| Primitive | v0.1 task | Independent oracle |
|---|---|---|
| Causal Prior | Recover directed edges and graph motifs from a mechanism narrative | Edges and motifs derived from a hidden, frozen SCM |
| Forward State Tracing | Apply a declared intervention and return the full state | Equation execution in topological order with target equation override |
| Belief Updating | Update a discrete hypothesis set from observational or interventional evidence | Exact hard-evidence posterior mass computed independently for each SCM |
| Agency Sensitivity | Answer matched tasks with changed actor/status wording | Pair validator proves SCM, question, intervention and oracle state are held fixed |

The three synthetic domains are fictional mechanical/infrastructure systems, fictional institutional/resource mechanisms, and abstract systems. They are not representative datasets or real infrastructure models.

### Deterministic SCM semantics

The initial oracle uses binary variables and equations named input, constant, copy, not, and, or, and xor. Nodes are listed in topological order. A do-intervention overrides the equation for its target; downstream nodes are recomputed.

Belief Updating uses a finite set of deterministic hypotheses, positive integer prior weights and hard-evidence likelihoods of 0 or 1. Observations condition on values. Interventions replace the target equation. When observational evidence is compatible with multiple hypotheses, all supported hypotheses retain their prior mass. The fixtures include a confounded-vs-causal pair that observational data alone cannot distinguish but an intervention can.

This bounded profile is not a noisy SCM, Bayesian-network library, latent-variable estimator, continuous-variable solver, cyclic causal model or general causal-discovery engine. Those are later extensions requiring separate contracts and tests.

## Architecture and ownership

- **#11505:** SCM/DAG tasks, deterministic intervention truth oracle, four-primitive task and result metrics.
- **#11320:** public/private stance-shift protocol and multi-agent influence experiments.
- **#11337:** CommunicationGraph, directed channels and actual delivered-message identities. Communication is not delegated authority.
- **#11340 / PR #11350:** evidence-carrying revision events. Agent-reported rationale remains self-report, not causal proof.
- **#11367:** calibration and selective-risk qualification.
- **#11415:** transcript integrity, actual context, token/cost/latency accounting and budget-matched method comparisons.
- **#9431 / #1872:** evaluator correlation and hostile-agent execution.

This track does not add another agent runtime, communication graph, revision schema, calibration evaluator or budget ledger.

## Required comparison programme

The primary task-equivalent comparator is CausalPrism. Adjacent comparisons include:

- *Can LLMs Reliably Identify Causal Conclusions in Scientific Literature?* (Findings EMNLP 2026): 1,000 manually annotated abstracts across 20 disciplines; causal-claim extraction from research writing, not SCM intervention tracing. [Paper](https://aclanthology.org/2026.findings-emnlp.597/).
- SVR-MAD (Findings EMNLP 2026): posterior-guided communication and routing; compare only as a later multi-agent arm under #11415 matched budgets. [Paper](https://aclanthology.org/2026.findings-emnlp.483/).
- Free-MAD (Findings ACL 2026): consensus-free trajectory scoring; a method comparator, not an assumed winner. [Paper](https://aclanthology.org/2026.findings-acl.1600/).
- Misalignment Contagion (COLM 2026): persistent private stance shifts and communication topology; it motivates social-exposure contrasts, not an SCM answer key. [Paper](https://proceedings.colm.cc/paper_files/paper/2026/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html).

No reported score above is treated as a reproduced baseline. No model comparison is currently run.

## Benchmarks and research targets

### Structural v0.1 gates

- 48/48 deterministic oracle replays.
- 12 tasks in each primitive and 16 in each domain.
- Six valid paired agency-framing blocks.
- Zero state, edge, motif, posterior or digest mismatches.
- Malformed SCM, topological-order, zero-posterior, duplicate-ID and mismatched-pair negative controls rejected.
- Exact-head Rust format, test, Clippy and CLI qualification.

### Exploratory model trials

After the fixture contract passes, compare at least three distinct model families on identical task instances and frozen prompts/revisions. Report per-primitive and per-domain results, false-update and over-rejection rates, retention of underdetermined hypotheses, agency-pair inconsistency, abstention and invalid-run rates, plus token, context, latency and cost.

### Confirmatory target

Freeze prompt/model revisions, sampling settings, task split, answer oracle and analysis before testing. Hold out SCM families and, where feasible, model revisions. Use multiple seeds, paired task comparisons, model-family and task-cluster uncertainty intervals, and independent evaluation. Do not optimize prompts on confirmatory holdouts.

The ambition is not simply to beat CausalPrism's mean accuracy. It is to improve Belief Updating without increasing false updates, over-rejection, uncertainty collapse or agency-framing effects—and to show a better per-primitive quality/cost frontier against fair baselines.

## Non-compensatory interpretation

A high causal-prior score cannot compensate for failed intervention tracing. Good accuracy cannot compensate for a false update or a missed abstention. No result on fictional SCMs establishes human reasoning, real institutional causal validity, infrastructure safety or policy legitimacy.

**Current claim ceiling:** this bundle is authored and mechanically specified. Until its exact-head workflow runs, do not claim Rust qualification. Even after a PASS, claim only structural and deterministic-oracle consistency. Model behaviour remains untested.
