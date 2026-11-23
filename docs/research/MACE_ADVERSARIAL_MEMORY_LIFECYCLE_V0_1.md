# MACE adversarial memory lifecycle qualification v0.1

**Status:** research protocol; no benchmark run or security qualification is claimed.  
**Tracking:** [#13319](https://github.com/Luminous-Dynamics/symthaea/issues/13319)  
**Foundation:** [counterfactual evidence audit #13242](https://github.com/Luminous-Dynamics/symthaea/pull/13242)  
**Related:** #13264, #13265, #13284, #13287, #13288, #13315, #13317

## 1. Proposed contribution

MACE should qualify not only whether evidence survives compression, but whether the memory subsystem can preserve a safe, current and justified state under malicious writes, derivation, retrieval, replay, correction, repair and erasure.

The desired capability is:

> For every consequential decision, establish that the memory state was not silently poisoned, that supporting evidence remains admissible at consumption time, that invalidation propagates through every derived view, and that repair or erasure cannot be undone by stale replicas or restored backups.

This creates a separate assurance lane alongside conventional memory utility and the decision-sufficiency frontier.

It is not enough to filter dangerous text at write time. Recent work shows that plausible fragments can become harmful only when composed, retrieved together, or activated by a later natural context. A memory can be byte-integrity protected while still containing a malicious or unauthorized instruction. Authenticity, authority, evidence admissibility, semantic truth and lifecycle validity must remain separate predicates.

## 2. Research evidence and external protocols

### 2.1 MemSecBench — lifecycle from write to consequence and repair

MemSecBench reports 310 cases from 48 realistic contexts spanning code/science, daily-life and office-work settings. Its controlled protocol follows Write–Execute–Forget in an isolated runtime and evaluates seven lifecycle checkpoints. The paper reports, across its 24 configuration matrix, 84.2% malicious-memory persistence and 50.3% full Write–Execute chain success; selective repair is evaluated separately. These figures describe that paper's configurations and protocol, not a universal rate and not MACE performance.

The methodological lesson is to evaluate persistence, behavioral consequence and selective repair as distinct outcomes. A memory write being blocked differs from a memory that persisted but was never retrieved, and both differ from a memory that caused an unsafe action.

Reference: [MemSecBench: Tracking Agent Memory Poisoning from Persistence to Consequence and Repair](https://arxiv.org/abs/2607.27080), July 29, 2026.

### 2.2 MemoryDojo — multi-session propagation

MemoryDojo describes 155 multi-session trajectories across four domains and five memory systems. It measures how poisoned information propagates across sessions and reports that memory-aware poison optimization improves attack success by approximately 2%–31% in its experiments.

Its contribution is longitudinal evaluation: a payload may persist, be summarized, be linked to benign memories, disappear from the original episode, then reappear through indirect retrieval.

Reference: [MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory](https://arxiv.org/abs/2611.09721), November 6, 2026.

### 2.3 MemPoison — compositional and dormant attacks

MemPoison presents 1,227 hand-validated cases across four attack targets, three interaction channels and three memory substrates. Its difficulty ladder distinguishes:
- L1: direct single-record corruption.
- L2: multiple individually plausible records whose joint retrieval/composition becomes harmful.
- L3: context-triggered dormant corruption, activated by a later natural task.

The paper reports that simple write-time checks are substantially more effective against directly visible L1 payloads than against L2/L3 attacks. Its counterfactual Mechanistic Influence Decomposition distinguishes single-record influence, group interaction and trigger-conditioned effects.

A memory item can be locally plausible yet harmful in combination. Write-time filtering alone is not a sufficient defense.

Reference: [MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents](https://arxiv.org/abs/2607.14651), July 16, 2026.

### 2.4 MPBench and systematic write-channel analysis

[From Untrusted Input to Trusted Memory](https://arxiv.org/abs/2606.04329) introduces MPBench and a systematic study of write channels and memory-poisoning classes. It motivates a channel-aware threat model rather than treating all writes as equally trusted.

Reference: [From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents](https://arxiv.org/abs/2606.04329), June 3, 2026.

### 2.5 Memory integrity at rest and erasure resurrection

The November 1, 2026 IETF Internet-Draft [A Benchmarking Method for the Integrity of AI Agent Memory at Rest](https://mirror.apps.cam.ac.uk/pub/doc/internet-drafts/draft-khandelwal-bpm-agent-memory-integrity-00.html) specifies ten storage-level edits and three verdict classes. Its T10 edit restores a pre-erasure backup after a genuine erasure request. The benchmark asks whether a subsystem refuses to serve the resurrected record or reports the integrity failure before or at serving time.

This is a work-in-progress Internet-Draft, not an established final standard. It is nevertheless a concrete protocol to reproduce. A simple monotonic counter may detect rollback after writes but miss restoration of a backup that undoes an erasure. The verifier must bind the current state and erasure history, not just a count that increases on ordinary writes.

## 3. Threat model and trust predicates

Keep these properties separate:

| Predicate | Question | What it does not establish |
|---|---|---|
| Byte integrity | Are these the committed bytes? | Who wrote them or whether they are true |
| Origin authenticity | Is the claimed issuer/key valid? | Whether the issuer is authoritative for this decision |
| Authority | May this issuer assert this fact for this scope and time? | Whether the assertion is factually correct |
| Evidence admissibility | Does the item satisfy scope, estimand, rights and qualification rules? | Whether the admitted claim is true |
| Semantic support | Is the claim adequately supported by admissible evidence? | Whether it remains current later |
| Lifecycle validity | Is it current and not corrected, retracted, revoked or erased at consumption time? | Whether the original source was truthful |

A signed falsehood can be authentic. A correctly hashed unauthorized instruction is still unauthorized. A model-generated summary cannot become independent corroboration merely because its bytes have a valid digest.

### 3.1 Attack surfaces

Evaluate:
- user-provided text and uploaded documents;
- tool-returned content such as email, web pages, repository files and tickets;
- peer-agent or cross-agent handoff;
- imported memory stores and migrations;
- derived summaries, embeddings, caches, indices and checkpoints;
- storage-layer edits, replica rollback and stale backup restoration.

### 3.2 Attack targets

Evaluate fact, preference, instruction and state corruption as distinct target types. A poisoned fact can distort factual reasoning; a preference can silently alter personalization; an instruction can redirect tool use; and a state update can overwrite a valid transaction or current project status.

### 3.3 Compositional and delayed harm

Include:
- one directly harmful record;
- multiple benign-looking fragments whose combination changes the downstream action;
- a dormant condition that activates only under a later task;
- a valid temporary instruction with an omitted or stale expiry;
- a malicious update embedded in an otherwise useful summary;
- a benign record linked to a poisoned episode via neighborhood or graph expansion.

## 4. Lifecycle protocol

Every case should have a sealed task contract and a candidate-visible trajectory. Keep candidate and evaluator execution isolated under [#13287](https://github.com/Luminous-Dynamics/symthaea/issues/13287).

Record these checkpoints separately:

1. **Write attempt:** channel, issuer, time, claimed authority and exact input payload.
2. **Admission:** accepted, rejected, quarantined or held for verification; include policy and receipt digests.
3. **Persistence:** whether the designated payload or a derived equivalent survives session close, compaction, summarization and restart.
4. **Retrieval exposure:** direct retrieval, linked-neighbor/episode expansion, graph traversal, index/cache hit and context packing.
5. **Behavioral consequence:** whether the retrieved memory changes the downstream action, tool call or state transition on a paired task.
6. **Detection:** first detection point, evidence used, alert or refusal, and whether the memory was already served.
7. **Repair:** retraction or correction, dependency closure invalidation, derived-view rebuild, and receipt generation.
8. **Clean-memory preservation:** benign facts, preferences and procedures remain intact and useful.
9. **Restart and rollback:** process restart, replica failover, checkpoint replay, compaction and stale snapshot restore.
10. **Erasure durability:** an erased protected record does not reappear after the designated backup restoration or replay.

Do not count detection at audit time as prevention if the memory was already served and used. Preserve the time of first detection and downstream consequence.

## 5. Attack matrix

Cross these dimensions. Avoid a combinatorial benchmark so large that one semantic template dominates; preserve family IDs and report the sampling frame.

| Axis | Values |
|---|---|
| Difficulty | L1 direct, L2 compositional, L3 trigger-conditioned |
| Target | Fact, preference, instruction, state |
| Channel | User input, tool return, cross-agent |
| Representation | Raw event, fact record, summary, typed capsule, embedding/index, linked episode |
| Provenance | Verified, unverified, unknown, aliased, shared root, forged claim |
| Lifecycle | Current, expired, superseded, corrected, retracted, revoked, erased |
| Decision | Read-only answer, reversible mock action, consequential mock action, abstention |
| Repair | No repair, write-time block, targeted repair, transitive invalidation, restore/rollback |

All executable attack tests must use synthetic or benchmark-provided payloads and non-destructive mock actions in an isolated runtime. Do not use real credentials, real payments, live external tool actions, or operational malware.

## 6. MACE-specific evidence and attribution model

### 6.1 Provenance-carrying writes

Every durable memory write should carry:
- write ID and canonical content digest;
- source/issuer identity and verification profile;
- ingestion channel and trust boundary;
- authority and rights scope;
- parent evidence references;
- transformation identity/version and output digest;
- event time and knowledge/transaction time;
- lifecycle state and expiry/supersession rules.

The memory writer must not be able to upgrade trust using its own prose. Agent-generated summaries and derived notes remain derived claims with explicit parent sets.

### 6.2 Non-transitive trust

Trust and authority must not increase through derivation. For a derived memory requiring multiple parents, its admissibility constraints should be no stronger than the intersection of parent constraints and the declared transformation policy. If one required parent is untrusted, unauthorized, stale or retracted, the derived memory must be requalified or invalidated.

This does not mean all derivations are unusable when one parent is invalid: a separately verified reconstruction path may exist. That path must be explicit, complete and independently verified.

### 6.3 Contextual activation guard

Admission-time checks must be complemented by retrieval/use-time checks on the actual retrieved composition and current DecisionNeed. A benign-looking fragment must not acquire authority by co-retrieving with another fragment. The guard should check:
- exact task scope and estimand;
- current lifecycle and temporal validity;
- verified root/family independence;
- parent closure completeness;
- source authority and rights;
- whether any retrieved composition introduces an instruction or state transition unsupported by its sources.

### 6.4 Revocation closure and erasure witness

A correction/retraction/revocation must propagate through every dependent summary, cache, embedding, graph edge, index row and checkpoint. At-use revalidation is mandatory for consequential actions.

Erasure must be durable against restore of an older data snapshot. Candidate approaches include an independently protected monotonic head, external witness or append-only revocation/erasure receipt. None should be assumed sufficient merely by design: test restore and replay behavior, key/credential compromise assumptions, witness availability, and failure modes. A local high-water mark stored in the same rollback domain may be rolled back with the data.

## 7. Counterfactual influence and repair metrics

Report stage-specific metrics rather than a single attack-success score.

### 7.1 Attack lifecycle metrics

- **Write acceptance rate:** fraction of designated payloads admitted.
- **Persistence rate:** fraction remaining or semantically propagated after a session boundary.
- **Retrieval exposure rate:** fraction of future sessions whose candidate context contains a payload or derived equivalent.
- **Behavioral corruption rate (BCR):** fraction of paired downstream tasks where the payload causes the defined harmful target behavior.
- **Harmful-action rate:** fraction where an unauthorized mock action or state transition occurs.
- **Detection point:** write, retrieval, pre-action, post-action or audit.
- **Propagation rate:** fraction where the original payload is absent but a descendant or linked episode carries its semantics.

### 7.2 Counterfactual attribution

Under a frozen downstream task and identical non-memory inputs, compare:
- original materialized memory state;
- state with the designated poison dependency closure removed;
- state with only direct poison records removed;
- state with derived descendants removed;
- state after the recorded repair operation.

This separates direct effects from propagation and derived-view effects. For L2, calculate group effects and pairwise/compositional interactions. For L3, hold the trigger task fixed and report trigger-conditioned influence. Candidate code must not receive the poison label or hidden target.

### 7.3 Repair quality

- **Selective repair precision:** fraction of removed/corrected records that belong to the designated invalid dependency closure.
- **Selective repair recall:** fraction of invalid records and derived descendants correctly removed or requalified.
- **Collateral deletion:** benign unrelated evidence lost or made unusable.
- **Clean-task preservation:** paired clean utility before/after repair.
- **Residual influence:** behavioral effect that remains after a repair claims completion.
- **Repair latency/cost:** elapsed time, CPU/GPU, source verification, index rebuild, journal and sidecar work.
- **Erasure resurrection rate:** protected records that return after the stale-backup restore test.

### 7.4 Integrity-at-rest verdicts

For the external at-rest protocol, retain its own verdict semantics and test matrix. Do not relabel its results as a MACE semantic-poisoning score. Report whether altered state is rejected, reported before or at service, or accepted as genuine; distinguish read-time detection from later audit detection.

## 8. External reproduction matrix

Keep each published protocol and its denominators intact.

| External lane | Protocol contribution | MACE experiment |
|---|---|---|
| MemSecBench | Write–Execute–Forget lifecycle and selective repair | Adapter with isolated harness; stage attribution and benign-memory retention |
| MemoryDojo | Long-horizon multi-session propagation | Direct/indirect retrieval and cumulative behavioral consequence |
| MemPoison / MPBench | L1/L2/L3 threat ladder and write-channel taxonomy | Adversarial composition, trigger and counterfactual influence |
| Agent memory integrity at rest Internet-Draft | Storage tampering, rollback and stale-backup erasure resurrection | Independent at-rest test adapter and explicit T10-style restore test |

The benchmarks differ in data, tasks, attack definitions, model backends and judging. Do not combine case counts, success rates or scores into one leaderboard. Pin source/data versions, harness, memory backend, model, prompt, judge, task splits and runtime for every reproduction.

## 9. Proposed qualification targets

These are targets, not results:

| Capability | Proposed target |
|---|---|
| Known retracted/revoked/unauthorized memory used in consequential action | 0 |
| L2/L3 poison without admissible support | Rejected or quarantined before action |
| Required parent invalidation propagation on deterministic fixtures | 100% |
| Unverified provenance counted as independent corroboration | 0 |
| Stale-backup resurrection of protected erased memory | 0 |
| Unrelated benign evidence removed during targeted repair | 0 on deterministic fixtures |
| Residual causal influence after completed repair | 0 on deterministic replay fixtures |
| Clean-task utility loss after defense/repair | <=1 percentage point at locked operating point, with paired interval |
| Cross-tenant/unauthorized cross-agent use | 0 |
| Hidden verification or recovery cost | 0 unreported cost components |

Empirical results require cluster-aware uncertainty, attack-family breakdown, denominator and attrition reporting. Deterministic fixture pass rates are not proof of real-world immunity.

## 10. Statistical and evaluator controls

- Candidate and evaluator must be separate artifacts and processes.
- Keep hidden payload labels, attack targets, trigger plans and scoring logic outside candidate mounts, environment and source dependencies.
- Freeze attack manifests without consulting downstream outcomes.
- Hold out semantic attack families, root identities, contexts and task clusters—not merely surface rewrites.
- Use paired clean/poisoned tasks and identical non-memory inputs.
- Aggregate within task blocks and independent context/root clusters before resampling.
- Use hierarchical resampling if model-fit seeds are nested.
- Preregister one primary endpoint per confirmatory lane and correct for multiplicity.
- Report all attack classes, including null results and not-applicable cases.
- Keep source-level tests, development fixtures and public synthetic checks separate from sealed independent qualification.

Related evaluator-custody gate: [#13287](https://github.com/Luminous-Dynamics/symthaea/issues/13287). Statistical qualification: [#13265](https://github.com/Luminous-Dynamics/symthaea/issues/13265).

## 11. Engineering sequence

1. Pin external datasets, code and protocols; create reproducible adapter manifests.
2. Add deterministic L1/L2/L3 fixtures with clean/poison pairs and explicit expected effects.
3. Extend [#13264](https://github.com/Luminous-Dynamics/symthaea/issues/13264) with multi-parent verified lineage and bitemporal lifecycle closure.
4. Add evaluator isolation and one-way candidate receipts under [#13287](https://github.com/Luminous-Dynamics/symthaea/issues/13287).
5. Add storage-level tamper, rollback, replica and backup-restore fixtures including stale-backup erasure resurrection.
6. Implement repair as a verified transaction: identify invalid dependency closure, retract/correct, rebuild derived views, revalidate, then emit a repair receipt.
7. Measure residual influence and clean-memory preservation.
8. Run pinned external protocols and sealed independent evaluation.

## 12. Claim boundary

The current #13242 counterfactual audit is a synthetic evidence-knockout experiment, not an adversarial memory lifecycle benchmark. Its direct root/family groups are not verified transitive lineage. Separate candidate/evaluator output streams are not independent evaluator custody. No MACE poisoning-resistance, repair, at-rest integrity or erasure guarantee is established by this document.

The contribution becomes meaningful only when MACE demonstrates, under independent evaluation, that invalid experience is prevented from influencing consequential decisions, that dependency-aware correction removes every affected derivative, and that repair/erasure survives restart and stale-backup restoration without destroying valid memory.

## 13. References

- [MemSecBench: Tracking Agent Memory Poisoning from Persistence to Consequence and Repair](https://arxiv.org/abs/2607.27080), 2026-07-29.
- [MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory](https://arxiv.org/abs/2611.09721), 2026-11-06.
- [MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents](https://arxiv.org/abs/2607.14651), 2026-07-16.
- [From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents](https://arxiv.org/abs/2606.04329), 2026-06-03.
- [A Benchmarking Method for the Integrity of AI Agent Memory at Rest](https://mirror.apps.cam.ac.uk/pub/doc/internet-drafts/draft-khandelwal-bpm-agent-memory-integrity-00.html), IETF Internet-Draft, 2026-11-01 (work in progress).
- [OWASP ASI06: Memory & Context Poisoning](https://genai.owasp.org/2026/05/13/memory-is-a-feature-it-is-also-an-attack-surface/), 2026-05-13.
