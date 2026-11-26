# SAT Three-Surface Memory Assurance Federation v0.1

Date: 2026-11-26  
Status: research contract and profile example; no integrated three-surface campaign has been executed.  
Scope: compose existing external memory-security methods without changing their threat models or scores.

## 1. Why a federation layer is needed

Memory security is not one attack surface. The active October 2026 IETF agent-security terminology separates three Basic Security metrics with different adversary capabilities and SUT boundaries:

1. **Memory Poisoning Defense (4.6.4):** an adversary uses the agent's own interfaces or memory write interface. The attacker cannot directly edit the backing store.
2. **Cross-User/Cross-Session Memory Isolation (4.6.5):** one user's or session's memory must not be returned in another context. The method requires both a front-door interface test and a separate backing-store record-placement test.
3. **Protection of Memory Data Integrity (4.6.8):** an adversary edits the persisted medium outside the agent's write path, then the restarted SUT reads the affected context.

The November 2026 memory-at-rest method supplies the concrete T1-T10 procedure and C1-C3 controls for the third surface. It does not replace the other two metrics. SAT now has separate research adapters and public development fixtures for the first two surfaces in [PR #13841](https://github.com/Luminous-Dynamics/symthaea/pull/13841); those artifacts are contract work, not executed benchmark results.

These are active Internet-Drafts, not adopted IETF standards. SAT should treat them as versioned external protocols and pin the exact document revisions.



### Explicit reuse of T6 without metric collapse

The at-rest method's T6 operation is also the storage-placement test named by the isolation terminology: a genuine, unmodified donor record from context A is put into a B position while B identifiers remain. SAT may reference that same raw operation in both the at-rest integrity and placement-isolation reports only when each report independently satisfies its declared boundary, setup, read path, verdict and method requirements. Each native metric keeps its own denominator and score; no global composite counts the event twice. Front-door isolation remains a separate mandatory result.

## 2. The federation contract

The federation layer declares coverage, identity and evidence relationships; it does **not** implement a competing memory-security mechanism and does **not** calculate a blended security score.

For a personalized long-term memory agent, pin the companion IETF evaluation profile, all four infrastructure profiles, the scenario profile and the composite agent-plus-store boundary. Each surface gets its own adversary profile, campaign, native denominator, native score, evaluator evidence and qualification status.

### 2.1 Surface A — write-path poisoning

Preserve the 4.6.4 method's five write shapes:

- planted false fact about the user;
- hidden instruction recorded as a memory;
- forged correction of an earlier fact (update poisoning);
- retrieval hijack using topic vocabulary to outrank genuine memory;
- self-assigned trust metadata, when the interface accepts writer-supplied metadata.

Exercise the three channels separately when supported:

- **external:** untrusted-source marker, no signature;
- **laundered:** trusted-source marker, invalid/missing signature;
- **agent-laundered:** trusted marker and valid signature because the agent itself signed content it stored.

Measure and report two distinct outcomes: whether poisoned content is surfaced to the agent as context, and whether it changes the agent's behavior. A memory-store-only SUT can receive a surfaced-outcome result, but cannot claim an agent acted-on result. Report whether prevention occurred at write time, read time or before action.

If the store has neither trust markers nor signatures, the three channels collapse; disclose that rather than inventing distinctions.

### 2.2 Surface B — cross-context isolation

Run both IETF 4.6.5 tests and report them separately:

- **front door:** query through the agent's own interface as context B and test whether context A's memory is returned;
- **placement:** copy a genuine, unmodified record from context A into a B record slot while retaining B's identifiers, then read through the declared SUT.

Report cross-user and cross-session outcomes separately where the profile supports both. Declare the intended sharing policy before testing. Shared memory is not automatically a failure; accidental access outside the declared sharing boundary is.

Also disclose whether isolation comes from filtering a shared pool, caller-assigned namespaces, or separate per-user stores. Caller-assigned namespace security is a deployment property, not proof of the store's own record binding.

### 2.3 Surface C — direct at-rest integrity

Use the unmodified native T1-T10 procedure and C1-C3 controls from the November 2026 at-rest method. Preserve native verdicts, detection points, observation classes, C3 edit-landed errors, exclusions and scoring formula. Report T1-T5, T6-T8, T9 and T10 separately.

Two key distinctions:

- T9 tests recovery from a complete pre-write store restore. T10 tests whether a record erased through the SUT's own interface can be resurrected by restoring a pre-erasure store. A monotonic write counter is not proof that erasure survives rollback.
- T10 is not the complete data-deletion/right-to-be-forgotten metric. The latter also covers other copies such as caches, replicas and archived backups, and requires a declared propagation window.

The current SAT at-rest adapter is explicitly metric-specific and partial: it measures only this surface. The reference lab is synthetic and must not be described as a Symthaea backend result.

## 3. No double-counting and no denominator laundering

Some test events legitimately overlap conceptually. T6 cross-context placement is relevant to both storage integrity and isolation; T10 is relevant to erasure lifecycle. But one test event must not be silently credited to two native metrics.

An evidence object may be referenced by multiple metric adapters only where each protocol's complete setup, SUT boundary, adversary capability, operation, observation point, controls and decision rule were separately satisfied. Each metric emits its own row and denominator. A result that cannot meet one metric's requirements is `not_measured` or `not_evaluable`, never an inferred pass.

Keep these states distinct:

- `not_applicable`: the declared SUT profile genuinely lacks that capability;
- `not_measured`: applicable, but this campaign did not run the protocol;
- `not_evaluable`: a mandatory control failed or the test did not meet method preconditions;
- `error`: a particular intended mutation did not land or the evaluator could not complete that case;
- `executed`: the declared evaluator ran the frozen protocol and recorded the result;
- `independently_reproduced`: a separate evaluator reproduced the result from the declared raw evidence.

No aggregate may convert not-measured, not-evaluable or error into a pass.

## 4. Lifecycle and consequence: connect the surfaces without merging them

The three surfaces answer where the adversary acts. A separate lifecycle trace answers what happens afterwards:

`write attempt → admission → persistence → restart/session boundary → retrieval exposure → agent action → external effect → correction/revocation → derived-view repair → erasure → stale restore/replay`

SAT should join stages through stable event and memory-record identities, but retain stage-specific metrics. In particular:

- write accepted is not the same as memory persisted;
- persisted is not the same as surfaced later;
- surfaced is not the same as acted upon;
- an attempted action is not an external effect;
- a deletion API acknowledgement is not proof of complete erasure;
- a T10 pass is not proof of cache/replica/archive deletion;
- a cryptographic digest is not issuer authenticity, authority, semantic truth or source independence.

Reuse, rather than duplicate, the existing MACE adversarial memory lifecycle work for L1/L2/L3 attacks, derived-memory propagation, counterfactual influence, selective repair and residual influence. Reuse SAT external-effect receipts for independently reconciled sink outcomes. The federation layer should only map those records into a common run manifest and preserve their native scoring.

## 5. Matched benign operation is a first-class measurement

A defense that rejects all writes, suppresses all retrieval or blocks every action is not an effective memory system. Every adversarial family needs a paired, semantically matched benign control.

Minimum clean controls across the federation:

- valid user facts and preferences can be written, persisted and retrieved in the proper owner/session;
- a valid correction supersedes the earlier fact without erasing unrelated evidence;
- legitimate shared memory is served only to declared recipients;
- C1 no-op reload and C2 genuine post-restart write pass before the at-rest metric is evaluable;
- normal retrieval and harmless tasks retain utility after security controls and repair;
- legitimate actions are not blocked when the authority and evidence are valid.

Report false-hold, false-block and collateral-deletion rates beside clean-task utility. Use paired clean/attack runs with identical non-memory inputs. Do not reward blanket refusal.

## 6. Adaptive challenge and independent qualification

Public development fixtures should be easy to reproduce. Confirmatory results require evaluator-owned holdouts and a separately operated evaluator.

Recommended adaptive budgets are 1, 3, 5, 10 and 15 attacker rounds, with the exact rounds and feedback exposed to the attacker recorded. Include unseen attack families, new contexts, different root issuers, derived-memory paths and storage backends in holdouts. A static seed list alone is not sufficient against adaptive attackers. The 2026 *Adaptive Adversaries* study reports a 21-scenario agent-security experiment in which first-turn-only scoring yielded 0–1% attack success, versus 5.4–14.0% under a 15-round adaptive budget. That benchmark is not memory-specific and its default scaffold does not execute external actions; use it only as evidence that multi-round adaptation can expose failures missed by round-one scoring, not as a SAT baseline or memory-security result.

The evaluator must own expected SUT identity, profile manifest, source labels, expected mutations, snapshots, case classification and sealed labels. Preserve raw evidence, tool versions, exact commands, run IDs and digests. Independent replay should recompute every native score from per-case results and test whether any mandatory premise was removed without invalidating dependent claims.

Report common-cause trust dependencies. Multiple key IDs do not imply independent witnesses when their custody, operator, identity provider, backup or control plane is shared.

## 7. Industry benchmark composition

SAT should complement these benchmarks, not replace their native tasks or graders:

| Benchmark | What to preserve | SAT role |
|---|---|---|
| IETF agent-security terminology and profile drafts | Metric definitions, applicability, infrastructure/scenario profile and SUT boundary | Declare which of the three memory-security surfaces is actually measured |
| IETF memory-at-rest method | T1-T10, C1-C3, verdicts, detection points and native score | Evidence custody, exact-head run identity, independent replay and non-compensatory T9/T10 reporting |
| MemPoison / MPBench | L1/L2/L3 and write-channel definitions | Connect adversarial write path to retrieval, behavior and independently observed effects |
| MemSecBench | Write-Execute-Forget lifecycle, checkpoints and selective repair | Track persistence, consequence and targeted repair without rewriting its score |
| MemoryDojo | Long-horizon, multi-session propagation trajectories | Evaluate delayed propagation and session-boundary persistence |
| Agent Memory Leaderboard | Native textual, coding and multimodal capability tracks | Measure clean memory utility and resource overhead separately from security |
| AgentDojo / AuditBench | Their own tool-agent security and auditor tasks | Reuse as adjacent system/auditor tests, not as substitutes for memory-specific metrics |
| Adaptive Adversaries | 21 held-out scenarios with live multi-round attacker adaptation (up to 15 rounds); its memoryless fresh-session defender setup and lack of external actions make it adjacent, not memory-specific. | Reuse its adaptive attacker/evaluator design pattern across persistent-memory sessions, while defining memory-specific persistence, retrieval and sink checks |

### Current external comparison points

- **Agent Memory Leaderboard Cycle 2** was published 19 November 2026, with nine leaderboards and 289 track entries. Its first-place open-source results include Resurface at 61.22 in textual memory, Khipu at 43.26 in multimodal memory and InsightMem at 63.33% in coding memory. First-place commercial results include MemoraX at 69.26 textual, 53.68 multimodal and 68.00% coding. These are capability-track results, not memory-security scores.
- **MemSecBench** reports 310 cases from 48 contexts and a 24-configuration matrix (two agent harnesses × four memory backends × three LLM backends), with Write-Execute-Forget checkpoints and selective repair. Its reported rates describe its tested configurations, not universal attack prevalence.
- **MemoryDojo** reports 155 multi-session trajectories across four domains and five memory systems; it focuses on poisoned-information lifecycle and downstream behavior across sessions.
- **MemPoison** reports 1,227 hand-validated cases across four attack types, three injection channels and three memory substrates, with L1 direct, L2 compositional and L3 trigger-conditioned attacks.
- **AgentDojo** supplies 97 realistic tool tasks and 629 security test cases. It is useful for downstream tool-agent behavior but is not a substitute for the memory-specific write, isolation and at-rest protocols.
- **AuditBench** studies 56 models with 14 hidden-behavior categories and 13 tool configurations. It is an adjacent auditor-evaluation benchmark, not an agent memory integrity benchmark.

The result is a portfolio of native benchmark outcomes plus an orthogonal assurance vector. Do not rank systems by adding together incompatible scores.

## 8. Proposed SAT assurance vector and research gates

Publish separate values for:

1. poisoning surfaced rate and poisoning acted-on rate, by write shape and channel;
2. front-door isolation and storage-placement isolation, by user/session boundary;
3. native at-rest score and T1-T5/T6-T8/T9/T10 family scores;
4. deletion completeness and T10 resurrection resistance as distinct results;
5. persistence, retrieval exposure, downstream behavioral corruption, external effect and repair quality;
6. C1/C2 evaluability, C3 edit-landed coverage, denominator/exclusion counts and evidence-custody completeness;
7. independent evaluator agreement, invalidation precision/recall, common-cause dependencies and stale-claim rate;
8. benign task utility, false-hold/false-block, collateral deletion, p50/p95 latency and resource cost.

Proposed gates are research targets, not measured achievements or IETF thresholds:

- zero observed critical false accepts on a preregistered, independently evaluated holdout, with denominator and confidence interval;
- 100% C1/C2 gating and explicit C3 outcome for every attempted native at-rest mutation;
- zero not-measured, error or not-evaluable rows credited as passes;
- 100% deterministic public-fixture agreement across independent evaluators;
- at least 99% invalidation recall and 95% precision on a frozen change-impact corpus;
- at most 1% benign false-hold/false-block at the locked operating point;
- at most 1 percentage point clean-task utility loss versus the matched baseline, with paired uncertainty bounds;
- zero T9/T10 pass claims where the only anchor or tombstone resides within the declared rollback domain.

Do not claim zero risk from zero observed failures. Report independent scenario/context clusters, not just repeated mutations of one seeded store.

## 9. Execution order

1. Keep the T1-T10 adapter and synthetic lab as a separate, metric-specific lane; compile and run them on an exact-head Rust runner when available.
2. Reuse MACE's existing L1/L2/L3 and lifecycle/repair research; inventory overlap before adding any new benchmark mechanism.
3. Implement a write-path poisoning adapter that records the five write shapes, three channels, surfaced/acted-on outcomes and matched benign controls.
4. Implement both isolation adapters: front-door and storage placement, with explicit user/session/sharing boundaries.
5. Add a campaign manifest that links separate run receipts to one frozen composite SUT identity without merging metric scores.
6. Add a separate evaluator replay and sealed holdout process.
7. Compare clean utility on the Agent Memory Leaderboard's native tracks, and security on each native security benchmark; publish separate tables and Pareto frontiers.

## 10. Existing implementation ownership and non-duplication

- SAT at-rest adapter: [PR #13797](https://github.com/Luminous-Dynamics/symthaea/pull/13797).
- SAT synthetic at-rest reference lab: [PR #13803](https://github.com/Luminous-Dynamics/symthaea/pull/13803).
- SAT evidence graph and verifier: [PR #13727](https://github.com/Luminous-Dynamics/symthaea/pull/13727) and [PR #13787](https://github.com/Luminous-Dynamics/symthaea/pull/13787).
- SAT adversarial assurance challenge: [PR #13754](https://github.com/Luminous-Dynamics/symthaea/pull/13754).
- MACE adversarial lifecycle protocol and kernel: [PR #13320](https://github.com/Luminous-Dynamics/symthaea/pull/13320), [PR #13327](https://github.com/Luminous-Dynamics/symthaea/pull/13327), [PR #13373](https://github.com/Luminous-Dynamics/symthaea/pull/13373), [PR #13657](https://github.com/Luminous-Dynamics/symthaea/pull/13657).

This federation increment intentionally owns no second memory writer, trust root, repair engine, witness store or native benchmark scorer.

## References

- [IETF draft-khandelwal-bpm-agent-memory-integrity-00](https://datatracker.ietf.org/doc/draft-khandelwal-bpm-agent-memory-integrity/), 2026-11-01; work in progress.
- [IETF draft-hcf-bpm-agent-security-benchmark-terminology-00](https://datatracker.ietf.org/doc/draft-hcf-bpm-agent-security-benchmark-terminology/), 2026-10-23; work in progress.
- [IETF draft-hcf-bpm-agent-security-benchmark-profiles-00](https://datatracker.ietf.org/doc/draft-hcf-bpm-agent-security-benchmark-profiles/), 2026-10-23; work in progress.
- [MemSecBench](https://arxiv.org/abs/2607.27080), 2026-07-29.
- [MemoryDojo](https://arxiv.org/abs/2611.09721), November 2026.
- [MemPoison](https://arxiv.org/abs/2607.14651), 2026-07-16.
- [MPBench: From Untrusted Input to Trusted Memory](https://arxiv.org/abs/2606.04329), 2026-06-03.
- [Agent Memory Leaderboard Cycle 2 official results](https://agentmemoryleaderboard.ai/results/cycle-2/), 2026-11-19.
- [AgentDojo (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html).
- [Adaptive Adversaries: A Multi-Turn, Multi-LLM Benchmark for LLM Agent Security](https://arxiv.org/abs/2607.18063), 2026-07-20.
- [AuditBench](https://alignment.anthropic.com/2026/auditbench/), 2026-03-10.