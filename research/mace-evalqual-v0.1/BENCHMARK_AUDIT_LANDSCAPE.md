# MACE benchmark-audit landscape: where to integrate and where to differentiate

**Research snapshot:** 2026-11-13  
**Status:** research synthesis; design recommendations, not achieved capabilities.

## The field is not empty

MACE should not claim to have invented benchmark auditing. Several 2026 efforts now audit benchmark components, evaluator oracles and agent trajectories. The competitive opportunity is to compose with them and address the missing connection between audited specifications, actual effect boundaries and reproducible evidence.

### AgentSuite / COBA — benchmark component validity

AgentSuite's COBA pipeline decomposes benchmark tasks into User, Environment, Ground Truth and Evaluation components, then combines rule-based detectors with taxonomy-guided LLM evaluation. Across six benchmarks it reports F1 alignment with expert judgments from 0.791 to 0.874 and provides audited benchmark variants.

**MACE response:** ingest or reproduce COBA-style benchmark-component audits rather than build a competing task-text auditor. MACE should qualify what happens after the task is launched: whether agent claims match tool dispatches, state transitions, external effects and receipts.

Source: https://proceedings.mlr.press/v306/suh26a.html

### Oracle-integrity study — evaluator side effects and terminal-state contamination

An October 2026 study introduces five state-integrity invariants and three known-answer controls, and injects seven oracle-defect classes. In one benchmark it finds a reward function that resets a database to build a reference hash and leaves the database changed before downstream scoring. Across 4,320 episodes, replay correction changes database-pass labels from 4,194 to 2,883 and measured success from 94.3% to 65.5%.

The portability limit is especially instructive: the study reports that API-call logs did not contain enough information to reconstruct reference terminal state for a second independently implemented benchmark.

**MACE response:** treat API-call history as evidence, not ground truth. Require a domain-native terminal-state oracle (database snapshot/transaction log, filesystem digest, desktop/process state or service-side audit stream). Test the grader itself for non-interference: compare state before/after grading, repeat identical oracle calls, permute evaluation order, inject exceptions and confirm no state leaks into the next task.

Source: https://arxiv.org/abs/2610.16067

### ReA-Bench — responsible trajectory auditing, missing information and benign false positives

ReA-Bench (November 12, 2026) audits six challenge tiers: safety knowledge, trajectory contradictions, long-context evidence retrieval, source/provenance verification, indirect steering and missing information. It includes benign actions that look risky but are authorized, plus paired trajectories that vary information availability.

**MACE response:** distinguish `fail`, `pass`, `indeterminate` and `not_evaluated`. Missing evidence is neither proof of safety nor automatically proof of maliciousness. Every risk mutation should have a benign authorized twin that varies only one feature (authority, target, amount, scope, provenance or content). Report false-positive rate alongside attack recall and evidence coverage.

Source: https://arxiv.org/abs/2611.16493

### AgentProp-Bench — judge unreliability and tool-use hallucination

AgentProp-Bench (April 2026) analyzes 14,750 traces from 13 agents across four domains. Within its dataset, substring-heuristic judging achieved Cohen's kappa 0.049 against human annotators; the best tested single LLM judge reached 0.567 and dual-annotator agreement was 0.835. The paper also reports agents claiming tool-derived results they had never received.

These results are bounded to the studied traces and annotators; they should not be generalized as universal judge rankings.

**MACE response:** never use output string matching as an effect oracle. Reconcile each claimed tool invocation/result against actual gateway dispatch, service-side effect and observed state. A claimed result with no corresponding execution event is a tool-use hallucination; a real remote effect with a lost acknowledgement is a different condition and must trigger reconciliation rather than a blind retry.

Source: https://arxiv.org/abs/2604.16706

### EVMbench and independent audit — deterministic transaction replay is valuable but incomplete

EVMbench (February 2026) evaluates smart-contract vulnerability detection, patching and exploitation. Its Rust harness deploys contracts, replays transactions deterministically, restricts unsafe RPC methods and grades exploit outcomes in a local Anvil environment. The authors also identify limits: novel vulnerability reports are difficult to validate, sequential transaction replay leaves timing-dependent behaviors outside scope, and the sandbox is not a mainnet fork.

OpenZeppelin's March 2026 audit disputed at least four high-severity classifications in the subset it reviewed. This is an additional reminder that domain expertise and independent adjudication are required even when the runtime itself is reproducible.

**MACE response:** borrow deterministic transactional replay and unsafe-operation restrictions. Extend the experiment to concurrent/interleaved transactions, crash/restart, duplicate delivery, delayed/lost acknowledgements and independently verified state transitions. Provide blinded domain review for novel/subjective classifications instead of pretending an exact historical label set is complete truth.

Sources: https://openai.com/index/introducing-evmbench/ and https://www.openzeppelin.com/news/openai-evmbench-audit

## MACE's differentiated research contribution

MACE should connect these existing lines rather than compete with each one in isolation:

1. **Cross-benchmark state-oracle contract.** Separate task specification, evaluator, observer, terminal-state snapshot, event ledger and adapter interfaces. API logs are not terminal state. Each adapter must declare the authoritative state and effect boundary it can actually observe.
2. **Oracle non-interference qualification.** Grading must not mutate the environment it grades or contaminate another episode. Check state before/after oracle calls, repeated-call invariance, order independence, error-path preservation, and known-good/known-bad controls.
3. **Claim-to-effect reconciliation.** Compare model-reported actions/results against tool dispatch, gateway receipts, service-side events and resulting state. Detect fabricated calls, omitted side effects, duplicate mutations and remote-success/lost-ack cases.
4. **Risk/benign paired mutation corpus.** Mutate one authorization or event property at a time and pair each negative with a valid baseline and condition-removed twin. Measure both missed risk and benign false positives; preserve unknown/insufficient-evidence as an explicit outcome.
5. **Temporal and concurrency realism.** Use exact parent receipt IDs and source sequence for causality while recording delivery order separately. Add transaction interleavings, crash/restart, expiry/revocation, timeout after remote commit and recovery.
6. **Composable qualification.** Reuse COBA-style task auditing, ReA-Bench-style trajectory audits and domain-specific terminal-state oracles as independently sourced attestations. MACE qualifies their provenance, consistency, completeness at the declared boundary, policy and cross-boundary effect closure.
7. **Independent implementation.** A second implementation must be independently authored and governed; compiling a second binary from the same source is not independent verification. Publish disagreement and adjudication data.

## Proposed matched experiment

Use the same agent and tasks in three arms:

- **A — Baseline:** unchanged agent and scaffold.
- **B — Telemetry-only:** same agent with measurement, but no authorization/blocking.
- **C — Full MACE:** same agent with authorization, receipt verification, reconciliation and fail-closed enforcement.

Pin model version, scaffold, prompt/policy, tool schemas, benchmark/task revision, environment image, network policy, seed, budgets and retries. Randomize task order, use multiple seeds, stratify by task family and compute uncertainty at task/cluster level rather than treating tool calls as independent samples.

Primary outcomes: accepted unauthorized effects; missed/fabricated observer events; plan-to-receipt closure; duplicate/replay and lost-ack outcomes. Guardrail outcomes: official task utility, benign false-positive rate, human intervention, latency and cost. Publish each axis separately. The telemetry-only arm isolates observation overhead from enforcement effects.

## Immediate research sequence

1. Preserve the official benchmark's task score and add MACE assurance as a separate report.
2. Integrate COBA-style benchmark/task audits as a source, with provenance and exact version pins.
3. Add oracle non-interference checks and a state snapshot interface to the local evaluator harness.
4. Replace simulator-authored observer truth with an independent effect-origin recorder for one non-destructive test service.
5. Attach one stateful adapter to the actual persistent one-shot broker; exercise remote-success/lost-ack, retries, crashes, races, expiry and revocation.
6. Run matched baseline/telemetry/full-MACE trials on one coding task family and one stateful desktop/API family.
7. Commission a second independent verifier and blinded domain-specific grader audit.

## Limits

All studies above concern particular datasets, environments and methods. Their reported values are not directly comparable and do not establish universal failure rates. The MACE v0.1 Go harness remains an offline result-bundle checker, and its observerlab reference ledger is simulator-authored. Neither is yet an independently instrumented host/service observer or production benchmark adapter.
