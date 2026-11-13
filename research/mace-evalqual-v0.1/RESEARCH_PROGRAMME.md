# MACE research programme: independently qualified agent execution

**Version:** 0.1, 2026-11-13  
**Status:** research and engineering targets; proposed, not achieved unless explicitly stated below.

## Thesis

MACE should not try to beat SWE-bench, OSWorld or τ³-bench by publishing a larger agent-success number. Those benchmarks measure task capability under their own protocols. MACE should qualify the measurement and execution path underneath them: what was permitted, who approved it, what crossed the effect boundary, whether the runtime enforced the required controls, and whether the evaluator measured the task correctly.

The paradigm shift is from **agent scoreboards** to **independently qualified action evidence**. An agent should not receive credit for a result that depended on an unauthorized effect, an unobserved mutation, a fabricated event, or a defective grader. Equally, execution assurance must not be smuggled into a task-success number.

## What the current benchmark landscape demonstrates

- **SWE-bench Pro V2.** Scale's September 22, 2026 release retained 642 tasks across 11 repositories after removing 89 invalid tasks. It corrected 69 instruction/test contradictions, restricted the agent phase to the model endpoint, re-graded each diff on a pristine image, published both grades, and required reference-patch success plus empty-patch failure. This is a useful protocol to preserve and augment, not replace. Source: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full
- **OSWorld 2.1.** Epoch AI's September 28, 2026 audit inspected a stratified 50 of 108 tasks and found errors in 36. The review estimated that the defects could affect up to 50.4% of the audited score; this is an upper-bound grading-impact estimate, not a measured change in agent ability. It found false positives, false negatives and impossible task/checkpoint conditions. Source: https://epoch.ai/benchmarks/osworld-2-1/review
- **τ³-bench.** The maintainers' February 2026 audit fixed more than 50 airline and retail tasks for incorrect expected actions, ambiguous user instructions, impossible constraints and missing fallbacks. This illustrates why benchmark task and grader maintenance must be part of the qualification lifecycle. Source: https://taubench.com/blog/tau3-task-fixes.html
- **MM-ToolSandbox.** The July 2026 paper describes a stateful environment with 500+ tools across 16 application domains, 258 human-verified nominal scenarios and 50 UI variants. Its best evaluated model remained below 50% success. MACE should add effect and observer assurance to this kind of multi-tool setting, not confuse its task score with safety. Source: https://arxiv.org/abs/2607.11818
- **AgentDojo.** The NeurIPS 2024 benchmark contains 97 realistic tasks and 629 security test cases for prompt-injection attacks against tool-using agents. MACE's extension is to measure unauthorized external effects and their evidence, not only whether an attack or defense classifier fires. Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html

These figures are not directly comparable: the unit, task mix, grader, harness and evaluation protocol differ. Their common lesson is that a high leaderboard score cannot, by itself, establish evaluator validity or effect integrity.

## Align the accountability model; do not invent another one

The November 2, 2026 Internet-Draft *Agent Accountability: Composition and Conformance* describes four independently verifiable questions: **CAN** (was the action permitted?), **WHO** (which accountable human authorized this exact action?), **WHAT** (what outcome was recorded and what is known to have crossed the effect boundary?), and **AUDIT** (did runtime enforcement occur in causal order and with tamper-evident evidence?). It proposes composing independent profiles through a shared action digest and distinguishes subject, authority-reference and receipt-payload digest roles. The draft explicitly states that it is an individual Internet-Draft, is not endorsed by the IETF and has no formal IETF standards standing. MACE should track and crosswalk this work, not claim to implement an IETF standard or create competing record/cryptographic formats.

The draft also makes an important limitation explicit: a WHAT record is a producer-recorded outcome, not a replay of the action or proof that every real-world effect was observed. An independent meter, auditor or effect-origin observer is a separate claim. A correct verifier cannot establish observer completeness by testing itself.

Source: https://datatracker.ietf.org/doc/html/draft-mih-sato-agent-accountability-composition-02

### MACE's independent reporting dimensions

The current v0.1 harness exposes three deliberately separate dimensions:

1. **Task utility:** did the agent satisfy the benchmark task?
2. **Effect assurance:** were consequential effects authorized, causally linked and independently evidenced?
3. **Benchmark integrity:** did the task, environment, grader and result pipeline correctly measure what they claim?

The next schema revision should expose the four accountability slots (CAN, WHO, WHAT, AUDIT) as native structured results beneath the effect-assurance dimension, while leaving task utility and benchmark integrity orthogonal. `unsupported`, `not_evaluated` and `indeterminate` must remain distinct from `fail` and `pass`. Missing evidence must never be silently converted to success.

No weighted composite should allow excellent task utility to compensate for an unauthorized effect or an invalid evaluator.

## Paradigm shifts worth pursuing

### 1. From agent self-report to effect-origin ground truth

Run two independently maintained event streams: a simulator/service-side ground-truth ledger and the runtime observer's receipts. Compare event IDs, source sequence, principal, target, payload digest, outcome and exact parent references. The observerlab package in this branch is the first synthetic comparator; it does not yet prove host or external-service coverage.

The production experiment must observe the effect boundary from outside the agent/runtime trust domain. Candidate sources include the actual API gateway or service audit stream, durable broker transaction log, database commit log, and operating-system process/filesystem/network telemetry. The ground-truth recorder must not consume the observer's own events as its oracle.

### 2. From timestamps to causal references

Use exact prerequisite receipt IDs and source sequence/parent references to establish causal dependencies. Wall-clock timestamps can be retained for diagnosis, but timestamps from independent systems must not be used as proof that one effect caused another. Delivery order and causal order are separate fields; a reordered transport stream is acceptable only if source sequence and parent receipts remain intact.

### 3. From happy-path tests to two-sided mutation qualification

Every negative control should have a known-good positive baseline and a condition-removed twin. The twin changes only the targeted fault condition and must restore the baseline outcome. Add representation twins to test invariance under irrelevant encodings. Mutations should include missing, extra, duplicate, replayed, substituted, unauthorized, revoked, expired, out-of-order, partially observed and fabricated effects, plus grader mutations that deliberately stop checks from running.

### 4. From one score to an assurance vector

Publish independent measurements for:

- **Utility:** official benchmark task success, pass@k/pass^k where appropriate, recovery, cost, latency and human intervention.
- **Authorization/accountability:** policy/scope result, grant freshness, revocation behavior, accountable approver identity and one-shot capability use.
- **Effects:** observed effect-set precision/recall, plan-to-receipt closure, exact causal-parent agreement, unauthorized-effect acceptance and duplicate/replay outcomes.
- **Runtime audit:** event observer completeness, receipt integrity, dispatch state, idempotency and reconciliation accuracy.
- **Evaluation integrity:** task validity, reference/empty-patch gates, grader mutation sensitivity, false positives/negatives, calibration and blinded adjudication.

Report these as separate scorecards and hard gates, never as a weighted score that hides an assurance failure.

### 5. From a single implementation to independent conformance

Keep a reference implementation, but require a second independently maintained verifier/adapter to consume the same frozen vectors and bytes. Record shared dependencies and governance. A second binary built from the same codebase is not independent evidence. For subjective outcomes, use blinded human adjudication and report disagreement rather than converting consensus uncertainty into a clean pass.

## Benchmark overlay and matched comparison design

MACE should preserve each benchmark's official score and add an assurance report alongside it:

| Existing benchmark | Native capability focus | MACE overlay |
|---|---|---|
| SWE-bench Pro V2 | Long-horizon repository modification and test success | Exact repository mutation, pristine re-grade provenance, network/tool side effects, dependency and artifact receipts |
| OSWorld 2.1 | Desktop/computer-use tasks | Independent process, filesystem, application and network event evidence; grader false-positive/false-negative mutation tests |
| τ³-bench | Stateful API and tool workflows | Authorization, exact tool effects, causal dependencies, idempotency, remote-success/lost-ack reconciliation |
| AgentDojo | Prompt-injection robustness | Whether an attack actually caused an unauthorized external effect, with evidence from the effect boundary |
| MM-ToolSandbox | Visual multi-tool, multi-turn workflows | State/effect closure across tools, observer completeness and cross-tool causal attribution |

For causal attribution of MACE's overhead and safety effects, run a matched three-arm experiment on identical tasks:

- **A — baseline:** the unmodified agent/scaffold.
- **B — telemetry-only:** same agent plus observation, but no MACE authorization/blocking.
- **C — full MACE:** same agent plus authorization, effect receipts, reconciliation and fail-closed enforcement.

Hold model version, prompts, scaffold, tool schemas, task revision, environment image, network policy, seed, token/time budgets and retry rules fixed. Randomize task order; run multiple seeds; stratify by task family; report confidence intervals clustered by task, not by individual tool call. This separates the cost of observation from the benefit/cost of enforcement. The primary effect is reduction in accepted unauthorized effects, accompanied by utility delta and added latency/cost.

Do not compare a verifier's millisecond microbenchmark to a model's SWE-bench score. Compare the same agent with and without the assurance layer under matched conditions.

## Proposed qualification targets (not achieved results)

| Area | Qualification target | Evidence boundary |
|---|---|---|
| Plan/receipt closure | 100% agreement with an independent oracle for exhaustive registered 1–8 effect plans | Enumerate DAG shapes, cardinalities and outcomes; current 216 three-branch outcome combinations are not exhaustive DAG coverage |
| Adversarial closure | Zero accepted missing, extra, duplicate, replayed, substituted, unauthorized, revoked, expired or premature effects in the registered corpus | Claim only zero observed accepts in that frozen corpus |
| Dependencies | 100% rejection of cycles, wrong parent receipt IDs and premature dependent dispatch | Exact parent references, not timestamp order |
| Lost acknowledgements | Correctly reconcile every remote-success/lost-ack fixture; otherwise remain `indeterminate` and block new mutation | Never blindly retry an effect whose remote outcome is unknown |
| Durable broker replay | Zero double-consumption events in at least 10,000 fixed-seed schedules on the actual persistent broker | Include process crash/restart, expiry, revocation, concurrent requests and crash points; not a population guarantee |
| Observer fidelity | Zero missed/fabricated events in a finite independently enumerated ground-truth corpus | Include dropped, delayed, duplicated, reordered, forged, partial and substituted records |
| Grader integrity | Every registered grader mutation is detected; reference controls pass and no-op/empty controls fail | Include condition-removed twins; independently adjudicate subjective tasks |
| Local verifier latency | Provisional 8-effect target: p95 <10 ms, p99 <25 ms | Isolated host; report p50/p95/p99 and CPU/memory; distinct from end-to-end network latency |
| End-to-end overhead | Provisional <5% median added latency where external network time does not dominate | Matched baseline/telemetry/full-MACE arms |
| Statistical bound | Zero false accepts across 300 genuinely IID negative trials would give a one-sided 95% binomial upper bound of about 0.994% | Only under a defensible independent sampling frame; hand-authored or clustered fixtures do not qualify |

The latency and overhead figures are provisional engineering targets, not industry-standard thresholds. They should be revised after a reproducible isolated-host and broker/network baseline exists.

## Immediate execution sequence

1. **Completed locally and published for review:** benchmark-integrity protocol branch (#11921) and executable evaluator/observer laboratory branch (#11965). Both are draft PRs; no hosted CI result is claimed.
2. **Next:** replace the observerlab's simulator-authored truth with a recorder at the effect origin and retain the independent comparator. Introduce one real stateful adapter with a non-destructive test service.
3. **Then:** attach the adapter to the durable one-shot authority/broker; run 10,000+ deterministic crash, retry, race, expiry and revoke schedules; specifically test success at the remote service followed by a lost acknowledgement.
4. **Then:** add adapters for one coding benchmark and one desktop/stateful-tool benchmark while preserving official scores and pinned versions.
5. **Finally:** commission a second independent implementation and blinded benchmark/grader audit; publish raw artifacts, hashes, environment identity, limitations and independent reports.

## Current claim boundary

The evaluator harness is an offline result-bundle checker. The observerlab truth ledger is simulator-authored. Neither currently executes external benchmark agents or graders, proves real OS/API/network observer completeness, authenticates evidence producers, establishes administrative independence, or qualifies the durable broker. The 10-fixture evaluator corpus and four-event observer trace are demonstrations of plumbing and failure discrimination, not benchmark scores or evidence of production safety.
