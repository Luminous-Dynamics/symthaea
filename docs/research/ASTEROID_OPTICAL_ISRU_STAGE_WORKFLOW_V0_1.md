# Optical ISRU v0.2 — causal stage workflow and resumability reference v0.1

**Status:** standalone Rust reference; 16 tests authored, exact-file validation pending  
**Parent:** Symthaea #12538 — stage-aware, resumable and asynchronous campaign planning  
**Related:** #12525 likelihood EIG/EVSI; #12534 propensity receipts; #12536 multi-resource campaign budgets; Symtropy #5612 typed evidence graph; #5682 bitemporal calibration; #5656 dynamic fouling; #5697 separate QCM/witness evidence

## Why another workflow abstraction is needed

A physical resource-recovery campaign is not one scalar measurement. It is a sequence of causally dependent operations and observations. A typical qualification chain may include feedstock characterization, optical-reference qualification, controlled exposure, intermediate thermal/material observations, volatile capture, purification, independent product assay and apparatus recovery. The actual sequence and safe hold points must be supplied by a rig-qualified operating protocol; this reference does not prescribe or authorize a physical sequence.

The 2026 multi-stage Bayesian optimization (MSBO) work by Torresi and Friederich extends BO to multi-stage workflows, uses intermediate proxy measurements, and supports stopping and resuming partially processed samples through a sample inventory. The published study optimizes a scalar final measurement and demonstrates a framework for flexible process sequences; it does not establish optical-ISRU process safety, metrology provenance or independent product acceptance. https://doi.org/10.1039/D5DD00572H

The 2026 pessimistic asynchronous BO work by Volk et al. compares asynchronous placeholder policies to serial sampling in simulated optimization environments. It emphasizes that asynchronous assignment can increase sample throughput and that pessimistic placeholders can help avoid premature convergence in some tested cases. Its reported compression assumptions include multiple experiments in flight, new work starting as previous experiments finish, equal experiment durations and negligible algorithm/equipment scheduling overhead. Those assumptions are not generally true for a multi-stage ISRU rig with exclusive instruments, heterogeneous stage durations, shared resource budgets, failure, fouling and calibration holds. https://doi.org/10.1039/D5DD00477B

The present reference starts with the workflow correctness conditions that those optimizer comparisons do not solve.

## First implementation

`research/asteroid-optical-isru-v0.2/stage_workflow.rs` supplies:

1. **Typed causal stage DAG:** stable stage IDs, prerequisite IDs, output kind, qualified-observation requirement, product-acceptance requirement, campaign-close requirement and optional exclusive apparatus lock.
2. **DAG validation:** duplicate IDs, missing/self/duplicate prerequisites and cycles are rejected.
3. **Concurrent stage admission:** independent stages may run concurrently; two active stages cannot hold the same declared exclusive resource.
4. **Explicit lifecycle:** NotStarted → Running → PartiallyObserved → Completed, with Paused → Resumable → Running; Failed is terminal in this reference.
5. **Pending versus qualified observations:** registering an asynchronous measurement does not satisfy a stage. A pending result must later receive calibration and qualification receipt references or an explicit rejection receipt.
6. **Lineage-bound pause/resume:** resume requires exact workflow, sample, batch, protocol-digest and graph-digest equality plus the exact checkpoint receipt ID.
7. **Product evidence type boundary:** an IntermediateProxy, CapturedProduct or PurifiedProduct observation cannot be converted to IndependentProductAssay.
8. **Independent acceptance gate:** all required acceptance stages must complete and the selected observation must be a qualified IndependentProductAssay from a designated required assay stage, accompanied by a separate independent-assay receipt.
9. **Closeout gate:** required close stages must finish, no active/paused stages may remain, and no pending observations may remain.
10. **Append-only event trace:** event sequence numbers are contiguous and recorded times are non-decreasing. Rejected time-order transitions are validated before state mutation.

## The research boundary: what the kernel does not claim

This is an execution-state and evidence-type reference, not an optimizer and not a physical laboratory controller.

- A graph digest, protocol digest, sample ID, batch ID, checkpoint ID or receipt ID is supplied by the caller. This module checks presence/equality; it does not compute digests, verify signatures, prove custody or authenticate issuers.
- A calibration receipt string is not proof of calibration. It must resolve to Symtropy's bitemporal calibration assessment and an independently governed authority receipt.
- A Qualified observation is only as good as the upstream verifier and its physical applicability. This module does not calculate measurement uncertainty or prove calibration range coverage.
- Exclusive resource locks model a declared resource identity. They do not implement hardware interlocks, resource scheduling, distributed leases, crash recovery, clock synchronization or split-brain protection.
- The event trace is append-only in memory for this reference, not durable storage. Production integration must use a transactional durable log, idempotency keys and recovery receipts.
- Campaign closeout may be used for non-product research campaigns. Product acceptance is a separate state and is never inferred from campaign closure.

## Paradigm shift: optimize the whole causal campaign

The next generation should not ask only, "Which parameter gives the highest predicted oxygen release?" It should ask, "Which sequence of admissible measurements and process stages most efficiently produces an independently accepted resource, while maintaining the evidence quality needed to trust that decision?"

That requires a joint planner over:
- **material state:** regime, morphology, volatile content and batch/family;
- **apparatus state:** contamination, focus, source output and recovery;
- **metrology state:** calibration validity, clock quality, covariance and shared causes;
- **workflow state:** stage DAG, partial work, pending measurements and asynchronous completion;
- **resource state:** the multi-dimensional campaign budget from #12536; and
- **decision state:** whether another measurement, pause, maintenance intervention, assay or acceptance is valuable.

The planner's next action should be allowed to be a stage transition, evidence request, requalification or pause—not only a new material exposure.

## Benchmark programme

Compare on identical sealed workflow worlds and externally admitted candidate sets:

- fixed sequential processing;
- one-shot GP-EI over a final scalar outcome;
- standard BO that ignores intermediate proxies;
- multi-stage BO with nested stage-aware acquisition (Torresi/Friederich);
- asynchronous BO with realistic/constant-liar placeholders;
- pessimistic asynchronous BO (Volk et al.);
- Symthaea EIG and EVSI, with audit propensities and resource feasibility;
- stage-aware EIG/EVSI over the workflow DAG.

### Required metrics

| Metric | Definition |
|---|---|
| Time to qualified decision | elapsed campaign time until a decision is supported by required qualified evidence |
| Accepted product per total campaign hour | independent accepted product divided by all rig occupancy, waits, assays, maintenance and downtime |
| Stage completion / abandonment | counts and time per stage state, by batch and material family |
| Partial-work salvage | fraction of paused/partially processed work that resumes under identical lineage and reaches a valid terminal state |
| Proxy-to-final calibration | held-out predictive calibration of intermediate proxies against final independently assayed outcomes |
| Async utilization | instrument occupancy, queue time and time-to-decision, with real heterogeneous stage durations |
| False product promotion | count of proxy/captured/purified signals accepted as final product; target zero |
| Evidence completeness | proportion of stage outcomes with required calibration, qualification, custody and protocol links |
| Decision regret | loss against the best admissible policy on the frozen hidden workflow |
| Resource efficiency | accepted product and decision value per rig-hour, energy meter plane, feedstock, assay budget and maintenance minute |

### Initial target hypotheses (not achieved results)

1. Reduce median time-to-qualified-decision by at least 1.5× versus the strongest eligible serial and asynchronous baseline, with a paired lower 95% confidence bound above 1.0.
2. Preserve non-inferior decision regret and interval calibration on whole-batch and whole-family holdouts.
3. Achieve zero false proxy-to-product promotion across frozen adversarial workflows.
4. Achieve zero unauthorized stage starts in prerequisite, lock-contention, closed-campaign, stale-calibration and lineage-mismatch controls.
5. Demonstrate at least 90% salvage of resumable partial work in worlds where the original sample/protocol lineage remains physically compatible; report incompatible cases separately rather than counting them as software failures.
6. Report all resource dimensions and campaign outcomes; no single scalar can hide a hard budget violation.

Targets 1, 2 and 5 are programme hypotheses to calibrate against rig realities and comparator performance. Targets 3 and 4 are correctness requirements for the modeled contract, not claims about a physical rig.

## Authored tests

The 16 tests cover DAG validity and cycle/missing-node failures, duplicate/self dependencies, prerequisite ordering, concurrent independent stages and exclusive-resource contention, pending observation completion blocking, immutable output-kind enforcement, proxy/product distinction, checkpoint and exact-lineage resume, asynchronous result arrival while paused, rejected observation handling, independent assay acceptance, campaign close constraints, duplicate IDs, event-time order and late-result rejection after close.

### Reproduce locally

```sh
rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/stage_workflow.rs -o /tmp/isru-stage-tests
/tmp/isru-stage-tests
```

**Validation honesty:** authored tests are not considered executed until the dedicated exact-file workflow has passed. A green repository workspace build does not substitute for compiling this standalone research file.

## Production integration path

1. Replace caller-provided opaque IDs with versioned references to the Mycelix authority/provenance receipts and Symtropy evidence graph.
2. Persist each transition transactionally in Symtropy, with idempotent event IDs, custody lineage and recoverable checkpoints.
3. Bind each stage resource lock to an externally enforced rig reservation/lease; do not rely on this in-memory model for equipment safety.
4. Add typed units, clock identity/uncertainty, stage duration distributions and resource-consumption observations.
5. Implement asynchronous arrival/reconciliation with late, duplicate, missing and out-of-order results, while preserving acquisition and receipt time separately.
6. Add dynamic optical-state and calibration changes from Symtropy #5682 / #5697 to the action eligibility model.
7. Feed qualified stage observations into Symthaea's likelihood models; evaluate EIG, EVSI, propensities and Pareto resource constraints together.
8. Seal the campaign worlds and compare all baselines with paired seeds, whole-batch/material-family holdouts, cluster-aware intervals and independently retained result artifacts.

This is the state-machine foundation for #12538, not a completed multi-stage optimizer or physical qualification result.
