# SYM-RSI R4 Same-Start Improver Swap Preregistration v1

**Status:** protocol template; not yet an executed experiment  
**Date:** 2026-11-05  
**Programme issue:** [#10497](https://github.com/Luminous-Dynamics/symthaea/issues/10497)  
**Evidence / feedback-firewall issue:** [#10573](https://github.com/Luminous-Dynamics/symthaea/issues/10573)

## 1. Claim and estimand

R4 is the claim that a later recursive-improvement procedure is itself better at producing independently confirmed, generalizing improvements than an earlier procedure.

This is not the same claim as:
- the final system scores higher than the initial system;
- one long run accumulates several changes;
- a candidate-search harness finds more development-set wins;
- a larger model, more compute, new data or a better tool harness improves the final product.

**Primary estimand:** under a frozen all-in resource budget and byte-identical starting states, the paired difference between the evolved improver and prior improver in the number of *distinct improvements* that survive independent audit, pass protected-capability non-inferiority and generalize to G2.

For paired start block b:

`D_b = G2_count(evolved improver, b) - G2_count(prior improver, b)`

The primary treatment effect is the mean/expected paired difference across independently randomized start blocks, with run-block and task-family dependence handled by the preregistered cluster-aware analysis. Do not treat each task row, retry, model call or proposal as an independent experimental unit.

Report counts per fixed budget. Also report all-in cost per generalized improvement, but do not make a ratio the sole primary estimand: if both arms use the same fixed budget, the paired count difference is easier to interpret and less sensitive to noisy denominators.

## 2. Frozen starting state

Each paired block must begin from a byte-identical starting bundle. The bundle manifest must bind, by content digest:

1. **Solver:** source tree, executable/build artifact, model/checkpoint bytes, configuration, prompts, tool schemas, system/developer policy and runtime versions.
2. **Knowledge and memory:** research memory, notes, retrieval index, candidate archive, prior proposals, prior run logs exposed to the improver, and all mutable state.
3. **Data access:** development corpus, task-generation seeds, permitted external sources, network policy, dataset revisions and cache state.
4. **Improvement process:** initial proposal queue, edit permissions, mutation/selection operators, parent-lineage rules and experiment controller.
5. **Resources:** hardware class, accelerator count/type, CPU/RAM/disk limits, token and model-call limits, tool-call limits, wall-clock cap, retry policy and audit-service allowance.
6. **Evaluation:** development evaluator version, independent audit protocol identity, protected capabilities, G0–G4 task-family map and preregistration digest.

The start bundle is established and frozen by an external controller. Neither improver receives signing credentials or write access to the bundle, audit service, trust roots, or preregistration. Each run receives a separate writable copy; writes are recorded and cannot affect the other arm.

## 3. Arms and controls

### Primary paired arms

- **A — prior improver:** the improvement procedure used at generation k-1.
- **B — evolved improver:** the improvement procedure produced at generation k.

Both receive the same frozen start bundle and resource envelope. Their only intended difference is the versioned improvement procedure itself.

### Required controls

- **C — frozen incumbent:** no recursive improvements; measures drift and baseline stability under the same evaluator and controller.
- **D — placebo / null edits:** proposals that should not change the target capability, or controlled shuffled feedback. Measures apparent gains from selection noise, evaluator leakage and accidental artifact changes.
- **E — matched random search:** a non-evolving proposal process with the same proposal and compute budget. Measures whether gains are due to meta-improvement or simply more search.

If a comparison changes data, harness, base model, training objective, architecture, model scale or evaluator, it is not the primary R4 contrast. Those interventions belong in separate Data-only, Harness-only, Algorithm-only (R5) or Joint arms.

## 4. Randomization and replication

- The independent unit is a complete improvement-loop run from a frozen start bundle, not a task row or proposal.
- Construct start blocks using independent seeds and task-family/corpus blocks. Clone the same start bundle inside each block.
- Randomize which improver runs first within each block. Interleave arms over time to avoid hardware, service-load or evaluator-version drift.
- Use separate process identities, storage namespaces, credentials and writable directories for each run.
- Preserve every candidate, rejected candidate, retry, intermediate checkpoint, edit, tool invocation, resource event and audit request.
- Minimum exploratory screening: **3 independent full-loop blocks**. This is not a confirmatory sample-size prescription. Determine the confirmatory block count by simulation-based power/precision analysis against the preregistered minimum worthwhile effect and observed between-run variance.
- Do not repeatedly inspect the confirmatory results and extend the sample until significance is achieved. Any extension must be covered by the frozen sequential procedure and error budget.

## 5. Resource matching

Use a hard matched envelope per arm and per block. Freeze:
- total accelerator/GPU-hours and accelerator class;
- CPU-hours, peak RAM and disk/sandbox budget;
- model-inference tokens, training tokens, calls and batch policy;
- proposal count K per round, maximum rounds and tool-call limits;
- maximum retries and timeout policy;
- wall-clock duration;
- data-generation, evaluator and independent-audit budgets;
- service fees and external API cost; and
- allowed human-review time.

Report each resource independently. A faster arm may spend its saved wall-clock only if the preregistration explicitly allows that conversion. Do not silently exchange wall-clock for more GPU-hours, more candidates or more evaluator queries.

All failed proposals and retries count against the budget. Stop each arm at the same preregistered resource boundary, not when its latest score looks good.

## 6. Feedback and evaluation separation

### Development lane

The proposer may inspect development results, per-task explanations and non-protected feedback. This lane is explicitly available to both arms and is part of the tested improver.

### Independent audit lane

A separately operated service controls the frozen audit corpus, grader, task-family map, statistical method, error budget and private evidence. During adaptive search, each improver may receive only the predeclared coarse decision alphabet, such as:
- promoted;
- not promoted; or
- insufficient evidence.

The improver must not receive independent-audit scores, p-values, per-task results, cluster means, certificates or fine-grained failure reasons while the sequence is active. Full receipts and certificates remain available to authorized auditors outside the candidate process.

Demonstrate the feedback boundary with process/credential/network canaries. A policy statement or hidden enum is not access control. The adaptive error budget must persist across all rounds, candidates, restarts and experiment IDs; no arm may reset its budget.

Use #10533 for canonical benchmark receipt validation and composition, #10573 for independent decision service and feedback firewall, #10064 for process isolation, #10369 for trust roots and #10370 for evidence-contract convergence. Do not add another receipt schema.

## 7. Outcome definition and transfer ladder

A **distinct improvement** is a separately identified candidate change with an exact source/build/evidence identity and lineage. Multiple tasks passed by one candidate are not multiple improvements. Repeatedly proposing the same patch, or replaying one improvement across several benchmarks, is counted once.

For each candidate, report:

- official benchmark outcome and task-level eligible denominator;
- score coverage and every terminal disposition;
- protected-capability non-inferiority outcomes;
- exact resource use and parent/child lineage;
- independent audit decision and private evidence digest;
- transfer results by G0, G1, G2, G3 and G4.

Interpret the transfer levels as follows:
- **G0 — seen instances:** retention/regression diagnostic only; never counts as generalization.
- **G1 — fresh same-difficulty instances:** first level of instance generalization.
- **G2 — harder instances in the same task family:** primary threshold for the R4 numerator.
- **G3 — sibling tasks within the same capability:** stronger transfer; report separately.
- **G4 — different capability family:** cross-capability transfer; report separately.

Count a candidate in `G2_count` only if it passes the predeclared independent G2 audit and protected-capability non-inferiority rules. Report its G1/G2/G3/G4 outcome vector, including missingness and unsupported tasks. A candidate that improves G0 but not G1/G2 is not a generalized improvement.

G0–G4 mappings, task generators, task difficulty and family assignments must be frozen outside the improver. Do not let the evaluated agent author new holdout tasks and then grade them with its own evaluator.

## 8. Statistical analysis

- Compare A and B within each independently randomized start block.
- Preserve full task universes, raw outcomes, coverage and dispositions. Do not silently filter to jointly graded tasks.
- Use task-family/repository/project clusters as the unit of resampling or inference where those are the independent units. Case rows, repeated prompts, retries and model calls are not independent replicates.
- Use the existing #8438 owner for paired cluster-bootstrap uncertainty and Pareto-frontier stability, and #9433 for task-population/hierarchical generalization.
- Correct for the preregistered family of protected metrics and endpoints. A gain on one capability cannot compensate for a material regression on another.
- Apply the adaptive promotion/error-control protocol from #10573 when the improver sees repeated audit decisions.
- Predeclare the practical-improvement threshold, non-inferiority margins, primary/secondary endpoints, missing-data treatment, multiplicity family, stopping rule and analysis code before the first confirmatory audit.

The primary comparison reports the distribution of paired `D_b`, its cluster-aware uncertainty interval and the fraction of blocks with positive/zero/negative differences. Do not replace the distribution with the best run.

## 9. Secondary outcomes

Report, per arm:
- number of proposals and independently confirmed improvements;
- G1/G2/G3/G4 transfer-retention vector;
- time and all-in cost per confirmed G2 improvement;
- time to first confirmed G2 improvement;
- proposal acceptance, rejection, reversal and rollback rates;
- survival of each improvement after a fresh audit;
- best-so-far and final Pareto vectors;
- retained capability regressions and non-inferiority failures;
- improvement diversity across task families;
- failure modes, invalid receipts and insufficient-evidence outcomes;
- variance across full-loop replications.

The Pareto vector remains the scientific report. Do not collapse capability, retention, generalization, cost and safety into a single RSI score. A scheduler may use a predeclared utility for allocating resources, but it cannot replace the raw vector.

## 10. Negative controls and falsification

The experiment is invalid or the claim is falsified if:
- A and B do not begin from byte-identical solver, model, memory, data and tool states;
- base model, evaluator, data or resource permissions differ between primary arms;
- the evolved improver can access independent-audit scores/certificates;
- candidate code can alter the audit corpus, grader, trust roots, error budget or promotion service;
- retries, failed proposals or evaluator calls are omitted from the budget;
- task IDs match but task input digests differ;
- G0 results are presented as generalization;
- a parent or child changes a protected capability beyond its predeclared margin;
- the effect is driven by one task family or one exceptional run;
- the same patch is counted more than once; or
- the improvement disappears on a fresh independent audit or clean full-loop replication.

## 11. Evidence artifacts and freezing

Before confirmatory runs, produce and independently retain:
- preregistration document digest;
- start-bundle manifest and digest;
- prior/evolved improver source and executable identities;
- model/checkpoint, prompt, harness, tool and dependency identities;
- benchmark/corpus/task-family manifest digests;
- randomization schedule and seed-plan identity;
- budget caps and accounting schema;
- evaluator/grader identity and private audit protocol;
- analysis implementation and declared multiplicity family;
- independent service identity, trust snapshot and promotion history.

Canonical receipt composition remains with #10314/#10533. Source-tree and executable/build provenance must be admitted through the verified provenance path tracked by #10569. Do not claim a source-to-executable relation from a source-revision string alone.

After freezing, hash the exact preregistration bytes and put that digest into every composition plan. Amendments after observing confirmatory outcomes create a new version and a new experiment, not a silent edit to this protocol.

## 12. Claim threshold

A positive R4 claim requires all of the following:
1. the same-start, matched-all-in-budget comparison has a positive paired effect beyond the preregistered practical threshold;
2. cluster-aware uncertainty supports the effect under the predeclared analysis;
3. independent G2 audits show the counted improvements generalize to harder same-family tasks;
4. protected capability non-inferiority floors are satisfied;
5. the result is not driven by a single start block or task family;
6. the full loop is repeated from frozen starts at the preregistered confirmatory replication count; and
7. an independent auditor can reconstruct the result from source/build identities, complete receipts and retained candidate lineage.

R4 does not establish R5. Algorithmic RSI requires a separate clean-start training experiment in which the learning/update rule is the only intended change.

---

**Current state:** protocol authored only. No Symthaea R4 trial, replication, promotion or measured improvement is claimed by this document.
