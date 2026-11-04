# Proof-Carrying Cultural Skill Capsules v0.1

**Status:** research architecture proposal; not a production wire protocol.  
**Scope:** reusable skill/policy knowledge exchanged between cognitive agents.  
**Design rule:** an artifact may be discoverable before it is trusted; it must not become a trusted capability merely because it was transmitted, signed, popular, or stored.

## 1. Paradigm shift

The current cultural-memory path is content-centric: a content hash, domain label, timestamp and truncated vector can be announced, then indexed in a resonance graph. That is useful for discovery and transport, but it does not establish that a recipient acquired a skill, can execute it, understands its conditions, or can safely transfer it.

CivBench proposes **proof-carrying culture**: transmit a typed, testable, provenance-preserving skill capsule rather than treating a content packet as knowledge.

A capsule is not merely an explanation or a vector. It binds:

1. **Scope:** which task family, contexts, environments and operating envelope it applies to.
2. **Contract:** preconditions, allowed actions, postconditions, invariants and resource/safety bounds.
3. **Policy:** a constrained, versioned representation that can be evaluated in a sandbox.
4. **Evidence:** demonstrations, independent test results, baselines, uncertainty and known failure cases.
5. **Lineage:** authorship, sources, parent capsules, transformations, content digests and signed attestations.
6. **Contestability:** the recipient can reject, fork, modify, compare or revoke the capsule without losing access to the original evidence.

The key separation is:

- **content identity** — which exact bytes/normalized object are referenced;
- **publisher identity** — who signed or asserted it;
- **semantic retrieval** — what it might be about;
- **execution validity** — whether it behaves as declared;
- **epistemic validity** — whether the evidence supports the stated claims;
- **transfer validity** — where it continues to work;
- **authorization** — whether a particular agent/environment is allowed to execute it.

These are different properties with different verifiers. A hash is not a signature, a signature is not a proof of truth, a semantic embedding is not an executable policy, and a successful test is not universal validity.

## 2. Capsule structure

The draft schema lives beside this document at `skill-capsule-v0.1.schema.json`.

### Identity and versioning

- `capsule_id`: stable logical identity across revisions.
- `capsule_version`: immutable revision.
- `content_digest`: digest of canonicalized capsule bytes, with algorithm and canonicalization version.
- `parent_capsules`: exact prior revisions from which this version was derived.
- `status`: proposed, structurally_valid, independently_tested, qualified, deprecated, revoked.

Do not mutate a published capsule in place. A changed policy, precondition, dependency or evidence claim creates a new revision and an explicit lineage edge.

### Scope and contracts

The scope should identify task-family IDs, context variables, ranges, exclusions, observation/action space versions, units, environment version and temporal validity. Never represent an unbounded context as universally applicable.

Contracts contain:
- preconditions and required observations;
- action schema, rate limits and forbidden actions;
- postconditions and task-success predicates;
- safety invariants and resource budgets;
- timeout/termination conditions;
- explicit assumptions and unsupported contexts.

Unknown, absent, not measured, false and measured-zero must remain distinct.

### Policy representation

V0 should prefer a constrained declarative policy IR or a capability-limited bytecode/runtime over arbitrary native code. The capsule references a policy artifact by digest and implementation/runtime version. A policy must declare determinism, stochasticity, seed semantics, memory bounds and dependencies.

An embedding may support semantic search or clustering, but it is never an execution authority or proof of semantic equivalence. Different policies with similar embeddings remain distinct; a content hash fingerprint must not be used as a semantic vector.

### Evidence and assurance receipts

Evidence should be represented as typed receipts referencing existing run/evidence contracts rather than a parallel generic run envelope. A receipt binds:
- task and benchmark version;
- model/policy digest and runtime;
- input/source snapshot digests and transformations;
- evaluation split and source/time/site holdout;
- seeds, replications and compute budget;
- baseline identities and scores;
- uncertainty intervals and calibration diagnostics;
- failed tests, adversarial cases and known limitations;
- independent verifier identity/version and verifier output digest.

Producer-generated receipts are claims by the producer. Independent qualification requires a separate verifier and exact binding to the same artifact digest.

### Provenance and identity

- Preserve author/agent identity, signing-key version and revocation status.
- Preserve upstream human sources, demonstrations, licences, consent restrictions and source-family dependencies.
- Distinguish author, curator, demonstrator, verifier and current holder roles.
- Retain transformations and forks as a directed lineage graph.
- Use the shared Mycelix/Symthaea evidence plane for provenance and attestations. Do not introduce another general-purpose evidence ledger here.

A signed capsule proves only that the signer asserted those bytes under a key. It does not prove authorship of the underlying idea, scientific accuracy, beneficial intent, or right to reuse third-party data.

## 3. Lifecycle: from announcement to trusted capability

| State | Meaning | Permitted operation |
|---|---|---|
| Discovered | Announcement or reference received | Index and inspect metadata |
| Retrieved | Exact capsule bytes obtained | Parse; verify digest/signature |
| Structurally valid | Schema, version and internal links pass | Sandboxed tests may begin |
| Authenticated | Publisher/key chain and revocation checked | Attribute the assertion to a key |
| Locally tested | Recipient's declared smoke/contract tests pass | Trial under declared local limits |
| Independently qualified | Independent verifier passes declared test suite | Mark qualified only for tested scope |
| Transferred | Recipient passes held-out task/context tests | Record transfer receipt and limits |
| Deprecated/revoked | New evidence, expiry, key revocation or safety issue | Block new execution; retain history |

These states are not a single trust score. For example, a capsule can be authenticated but fail tests; tested but not independently qualified; qualified in one context and explicitly unqualified in another.

## 4. Learning must be measured separately

CivBench must not infer learning from the existence of a packet, hash, graph edge, demonstration view or positive social-resonance score.

For every learner, record separate outcomes:

1. **Transport:** bytes arrived intact.
2. **Authentication:** publisher identity and signature verified.
3. **Comprehension/teach-back:** learner can reconstruct the rule, preconditions and failure boundaries without replaying the original trajectory.
4. **Acquisition:** learner can execute the skill in the training context.
5. **Integration:** learner actually uses the skill when appropriate, while preserving alternatives and the ability to reject it.
6. **Retention:** skill remains available after time, memory pressure, restart or removal of the original teacher.
7. **Correction:** corrupted or obsolete skill is detected and repaired when new evidence arrives.
8. **Transfer:** skill succeeds on pre-registered held-out tasks, contexts, topologies, sources and operating conditions.
9. **Cumulative improvement:** descendants improve on the best individual-learning and prior-generation controls without losing provenance or safety.

A learner's teach-back should be a new representation or policy artifact, not an exact string/hash copy. Its lineage should point to the source capsule, but the verifier should score it independently.

## 5. Paradigm-shifting experiments

### 5.1 Demonstration compression

Compare raw trajectory replay, natural-language explanation, action-rule list, declarative policy IR and an HDC retrieval embedding. Hold the number of bits/tokens/demonstrations fixed. Measure execution, teach-back, transfer and failure-boundary recall—not compression ratio alone.

### 5.2 Proof-carrying demonstration

Provide preconditions, postconditions, invariants and counterexamples with a demonstration. Compare with demonstration-only and explanation-only conditions. Test whether proof obligations improve correct rejection and transfer rather than merely increasing confidence.

### 5.3 Error-correcting culture

Inject controlled corruption, contradictory teachers, poisoned demonstrations, version drift and delayed corrections. Measure false adoption, time-to-detection, recovery, descendant contamination and rollback/revocation propagation.

### 5.4 Plural knowledge and contextual specialization

Allow multiple valid capsules with different scope. Test whether agents learn when to select each one, rather than converging on one globally prestigious policy. Score conditional performance, boundary violations and diversity of retained strategies.

### 5.5 Memory and forgetting

Vary retention budget and query drift. Compare FIFO, recency, raw performance, proven reliability, diversity preservation, provenance-aware retention and explicit archival. Test whether a low-frequency but high-value skill survives and whether obsolete skills are retired safely.

### 5.6 Knowledge sovereignty

Blind source labels, then reveal provenance in controlled conditions. Test whether learners can challenge, refuse, fork, request more evidence or retain competing interpretations. Do not equate adoption rate with beneficial learning or autonomy.

## 6. Benchmark ladder and proposed targets

Targets below are **proposed engineering goals**, not achieved results or published universal standards.

| Gate | Benchmark | Proposed target |
|---|---|---|
| G0: identity | Digest/canonicalization round-trip | 100% exact digest match on valid fixtures; all tampered fixtures rejected |
| G1: authentication | Signature, key rotation and revocation adversarial suite | 0 revoked/invalid identities accepted in the test suite |
| G2: structural safety | Schema, scope, precondition and capability checks | 100% invalid-scope/unknown-action fixtures rejected before execution |
| G3: execution | Deterministic replay and contract tests | 100% replay equivalence for declared deterministic policies under pinned runtime |
| G4: teach-back | Learner reconstructs policy from bounded demonstration | Report exact rule recovery, boundary recall and executable success separately |
| G5: transmission | Teacher choice, observation, acquisition and integration | Estimate each transition probability with uncertainty; no collapsing to one adoption score |
| G6: retention | Multi-generation, memory-pressure and teacher-removal tests | Report survival curves and recovery, with matched individual-learning budgets |
| G7: transfer | Held-out networks/tasks/contexts and source-family splits | Beat strongest simple baseline and archived DQN on pre-registered held-out score with 95% CI excluding zero |
| G8: reliability | Noise, poisoning, contradictory evidence and delayed correction | Report false-adoption and recovery curves; proposed initial goal: ≥50% lower false adoption than unweighted imitation at matched acquisition rate |
| G9: cumulative improvement | End-to-end population experiment | Proposed initial goal: ≥5% relative held-out score gain over strongest reproduced baseline without higher safety-boundary violation rate |

A target is not a result. Thresholds should be frozen after pilot calibration and before the definitive test split is opened. Small synthetic suites can test implementation correctness, but cannot establish human or real-world generalization.

## 7. Fair industry comparison

Do not claim to replace or beat DeepMind's few-shot imitation/cultural-transmission work, the Reward Network III human study, or existing agent communication systems using a single leaderboard score.

Use complementary tracks:
- Reward Network III: population-level discovery, demonstrator selection, inter-generation retention and human/machine social learning.
- DeepMind MEDAL-ADR line: agent-to-agent few-shot demonstration transfer and generalization to held-out contexts.
- Symthaea/HDC/LTC: cognitive representation, memory and adaptive control contenders on the same task budgets.
- Transparent baselines: random, greedy, majority imitation, prestige selection, performance-softmax, Bayesian/evidence-weighted, no-social and matched individual-learning.
- Proof-carrying capsule track: correctness, provenance, safety, teach-back, correction and transfer independent of popularity.

A credible superiority claim is task-specific: identify the task split, source version, baselines, budget, uncertainty, exact artifact digest and independent verifier. If a complex cognitive system cannot outperform a simple baseline, publish that result and keep the mechanism optional.

## 8. Security and failure cases

The adversarial suite must include:
- forged publisher identity, revoked key and key-rotation mismatch;
- capsule digest tampering, canonicalization ambiguity and stale parent link;
- hidden dependency, capability escalation and resource exhaustion;
- policy that succeeds on training demonstrations but violates a postcondition on a held-out state;
- misleading prestige or artificially inflated demonstration score;
- colluding teachers and Sybil amplification;
- semantic embedding collision or adversarially similar content;
- poisoned, partial or out-of-context source lineage;
- version rollback, replay and stale-evidence reuse;
- evaluator compromise and producer/verifier implementation overlap;
- false confidence and poor calibration;
- loss of low-frequency/high-value knowledge under memory pressure.

Fail closed for execution authorization; preserve rejected artifacts and reasons for audit. Never erase a revoked artifact's lineage or negative evidence.

## 9. Integration boundaries

- **Symthaea:** local discovery, representation, memory, teach-back and cognitive execution.
- **Mycelix / shared evidence plane:** identity, provenance, attestations, revocation, source lineage and dispute records.
- **CivBench:** benchmark tasks, train/validation/test splits, baselines, scoring and independent qualification criteria.
- **Symtropy:** controlled generational/population simulation and mechanism experiments.
- **Sol-Atlas:** optional temporal/spatial visualization of evidence, scope, lineage and transfer; visualization is not a validator.

The capsule is a typed payload/claim bundle layered over existing run/evidence contracts, not a replacement for them.
