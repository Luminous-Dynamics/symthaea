# CogSec Cognitive Mutation Census v0

Status: initial static census against Symthaea `main` at `9748eeba5cca2e62fbdac0d677dee716488f7c9c`.

This is not a completeness claim. Its purpose is to turn CogSec integration into a coverage problem: every state write that can materially or persistently change cognition must be classified, assigned an owner, and either mediated or explicitly documented as ordinary internal dynamics.

## Classification

- **P0 — privileged persistent/authority mutation:** must be mediated before CogSec enforcement can claim coverage.
- **P1 — bounded active cognitive influence:** should be labelled/budgeted and mediated for hostile/remote inputs.
- **P2 — ordinary internal dynamics/telemetry:** normally not capability-gated, but its inputs may still require labels.

## Current known mutation sinks

| Sink | Current path | Class | Primary risk | CogSec migration |
|---|---|---:|---|---|
| Active goal creation | `ContinuousMind::process_inputs()` handles `InputType::Goal`; public `set_goal()` queues it; `AsyncMindHandle::set_goal()` exposes it across the actor boundary | P0 | untrusted instruction becomes active objective | `GoalProposal` -> monitor -> `GoalActivationPermit` -> protected goal store |
| Working-memory admission | `process_inputs()` pushes raw HDC content plus source/verified/metadata parallel arrays | P1/P0* | untrusted data gains active influence; metadata can drift from content | replace parallel arrays with one security-enveloped item; admission policy; persistent promotion remains P0 |
| Working-memory eviction/graduation | evicted items create `GraduationEvent`; coordinator may persist to episodic memory | P0 | observation silently becomes persistent cognition | transactional `MemoryCandidate` -> admission -> commit receipt |
| Verification/consolidation strength | `MemoryCoordinator::process_graduations()` boosts effective Phi for `is_verified`, especially ActionFeedback/WebResearch, and gives verified episodes positive valence | P1/P0 | truth/integrity, salience, consciousness, and affect reinforce one another | separate epistemic support, salience, control integrity, and affect; CogSec label is not a Phi/valence input by default |
| Semantic-memory storage | `SemanticMemory::store*()` accepts HDC/error/category without security/provenance envelope | P0 when used as trusted long-term knowledge | persistent semantic poisoning and loss of provenance | store `SecurityEnvelope<SemanticEntry>` or protected equivalent; promotion permit for trusted semantic tier |
| Feedback valence mutation | `process_inputs()` applies `InputType::Feedback` metadata directly to emotional valence | P1 | adversarial feedback steers affect | bounded `AffectProposal` + source/influence budget |
| Surprise injection | public `inject_surprise()` directly raises arousal, lowers valence, and raises mood temperature | P1 | external caller can steer attention/exploration | distinguish trusted local outcome signal from external input; bounded affect/attention permit or internal-only handle |
| Thermodynamic/affective actor update | `AsyncMindHandle::update_thermodynamics()` directly changes load and mood; Liquid-Mamba backend may receive affect | P1 | remote/API caller obtains active affect/model influence | authenticated sensor/control adapter + bounded `Physiology/Affect` influence envelope |
| Social thought integration | trusted-enough peer behavior is weighted and bundled directly into `current_thought` | P1 | source trust becomes active cognitive influence permission | separate trust from `InfluenceGrant`; per-domain cumulative influence budget |
| Social neuromodulator coupling | multi-agent social messages can couple peer bath state into local bath | P1 | remote peer gradually alters learning/affect | authenticated labelled observation -> bounded influence proposal; no ambient coupling authority |
| Mesh unauthenticated telemetry routing | non-critical packets may fail open as unauthenticated, then wisdom/affect/gradient payloads enter downstream cognition | P0/P1 depending payload | “may observe” silently becomes “may influence/learn” | preserve authentication/control-integrity label through routing; fail open only to observation, fail closed to influence |
| Federated gradient inbox | `receive_gradient()` and mesh gradient packets append peer gradients to federated inbox | P1 staging / P0 promotion | attacker supplies learning candidate | receipt is observation/quarantine only; sender cannot self-assert authoritative trust weight |
| Federated local-weight update | `process_federated()` aggregates and applies peer gradient to local weights | P0 | remote learning directly alters trusted model | prepare -> shadow model -> qualify -> `LearningPromotionPermit` -> model epoch N+1 |
| Liquid-Mamba aggregated update | `process_federated()` can apply aggregated peer weights to LLM backend | P0 | remote peers alter language/reasoning backend | inference handle separated from training-control handle; only CogSec learning controller owns mutation handle |
| Swarm LoRA delta | `SwarmMessage::LinguisticDelta` can call backend `apply_lora()` | P0 | remote linguistic adaptation installed directly | `LinguisticUpdateCandidate` -> provenance/auth -> shadow qualification -> learning permit |
| Swarm resuscitation | healthy/thymus or proof-verified packet can replace holocell state and reset consciousness | P0 | validity proof is treated as sufficient to mutate local cognitive state | verification establishes validity only; separate local authority/consent capability required for state replacement |
| Brain mutation proof path | ZK proof verifies mutation properties and is described as enabling application | P0 | mathematical validity conflated with installation authority | proof result becomes evidence attached to a `LearningMutationRequest`; never the permit itself |
| Web-research integration | sufficiently confident “verified” research claims are inserted into local knowledge graph | P0 for trusted knowledge | confidence threshold silently promotes web assertions into knowledge | observation graph first; `EpistemicAssessmentV2` derived separately; operational/semantic promotion capability-gated |
| Web-research Phi accounting | integration measures Phi before/after and accumulates Phi gain | P2 metric, dangerous if used for promotion | graph integration/reward can favor connectedness over truth | keep diagnostic only; prohibit Phi from security/epistemic authorization |
| Knowledge graph mutable access | integrator exposes mutable graph access | P0 if reachable from untrusted subsystem | bypasses promotion/admission path | narrow mutation API or protected graph owner; raw mutable access unavailable to untrusted cognition |
| Outbound social/mesh cognition | current thought, affect, gradients, and other state are emitted to peers | P0 for confidentiality | private-derived cognition can leak without egress policy | output candidate inherits confidentiality; egress/declassification monitor before public/peer sinks |

`P1/P0*`: working memory itself is normally transient, but an admitted item can influence active cognition and later graduate to persistent memory. Runtime policy may therefore classify particular working-memory admissions as P0 when they carry instruction/authority-bearing or highly sensitive content.

## Cross-cutting bypass classes

### 1. Direct mutable handles

Holding a mutable or capability-rich object can itself confer authority. Examples include an LLM backend handle that supports both inference and model mutation, mutable knowledge-graph access, and direct mind-state methods.

Migration rule: split read/inference/proposal interfaces from mutation-control interfaces. Ordinary cognition receives only the least-authority trait/object it needs.

### 2. Metadata detachment

Parallel vectors or side maps allow security metadata to drift from content. Security labels and provenance should be structurally attached to the object they govern.

Migration rule: use security envelopes/protected records rather than parallel content/source/verified arrays.

### 3. Validity-to-authority confusion

A signature, ZK proof, health check, source verification, consensus score, or model-quality score can establish useful evidence without authorizing installation or action.

Migration rule: validity/evidence becomes input to a mutation request; only an independent capability/policy path can produce a commit permit.

### 4. Fail-open influence

Availability-oriented network paths may intentionally accept unauthenticated telemetry. That is acceptable only if downstream code preserves its low-integrity label.

Migration rule: **fail open to observation; fail closed to influence**.

### 5. Slow-burn influence

Per-event limits are insufficient when many small changes accumulate.

Migration rule: maintain per-principal/domain/mutation-class influence budgets with windowed cumulative accounting and decay.

### 6. TOCTOU/replay

Authorization may be valid at evaluation time and stale by commit time.

Migration rule: one-use permits bind mutation digest, resource-state root, policy root, policy/authorization/revocation epochs, and logical sequence; protected state revalidates those bindings at commit.

## Existing transactional work to reuse

The experimental `agent/cognitive-interchange-transactional-state-ratchets-v14` lineage contains a useful correctness pattern: clone/stage state, perform strict backend work, accept/reject the result, then commit only accepted state. Its tests explicitly ratchet preservation from a non-zero pre-existing state.

CogSec should reuse the *invariant*, not merge the diverged branch wholesale:

- rejected privileged mutation leaves protected pre-existing state unchanged;
- allowed independent error/audit accounting is explicit and append-only;
- external side effects that cannot be rolled back are called out as outside the transaction boundary;
- future hot paths may replace clone-and-commit with explicit prepare/commit APIs once equivalence is proven.

## Required census expansion before enforcement

The next audit passes should cover:

1. all public `ContinuousMind` mutators and actor commands;
2. all writes to model/backend weights, adapters, LoRA state, projections, and CfC parameters;
3. all episodic, semantic, conversation, and knowledge-graph persistence paths;
4. all trust/reputation/quarantine/peer-policy mutation paths;
5. all tool, process, filesystem, network, actuator, and administrative execution paths;
6. all outbound sinks carrying generated text, cognitive vectors, memory, gradients, affect, telemetry, or evidence;
7. all feature-gated variants (`mesh`, `liquid-mamba`, `multi_agent`, `mycelix`, robotics/control features);
8. all FFI/plugin/dynamic dispatch surfaces capable of mutation;
9. restore/checkpoint/resuscitation/import paths;
10. test-only or legacy bypasses that may become reachable in production profiles.

## Enforcement gate

CogSec enforcement must not make a complete-coverage claim until:

- every P0 sink has a named protected-state owner;
- every P0 sink requires a monitor-minted one-use permit or an explicitly equivalent lower-level enforcement mechanism;
- every P1 remote/untrusted influence path carries security labels and bounded cumulative influence policy;
- every egress path has an explicit confidentiality policy;
- evidence-plane counters show `privileged_mutations_without_permit == 0` under qualification;
- red-team tests cover alternate/bypass paths, not only the primary API;
- unknown privileged mutation sinks are treated as a release blocker.
