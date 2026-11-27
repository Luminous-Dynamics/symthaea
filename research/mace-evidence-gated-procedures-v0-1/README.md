# MACE Evidence-Gated Procedural Memory

**Version:** 0.1.0  
**Status:** interchange and lifecycle contract authored; compiler/runtime not implemented  
**Tracking:** [issue #13922](https://github.com/Luminous-Dynamics/symthaea/issues/13922)

## Research position

MACE should learn from verified experience, but a successful trace is not automatically a reusable procedure.

The question is whether a system can turn a sequence of prior actions into a procedure that is:
- grounded in trace events rather than generated from an unverified narrative;
- explicit about preconditions, postconditions, permissions, effects and environment;
- independently replayed and checked before admission;
- retrieved only when its applicability and authorization requirements hold;
- versioned and revocable when its source evidence or environment changes; and
- measured by whether its use causally improves outcomes without increasing unsafe actions.

This is **evidence-gated procedural memory**. It is an extension of the DSM evidence lifecycle, not a claim to invent trace-to-skill compilation.

## Current prior art and industry comparison

Zep Agent Skills is important prior art (paper: https://arxiv.org/abs/2611.01386). It captures task traces with ordered events, outcomes and verification strength; uses Reflector and Curator roles to compile verified traces into versioned source-cited skills; applies admission scans/gates; and feeds verified use outcomes back into trust and retrieval ranking. The paper reports 4.5–18.8 percentage-point transfer gains for weaker executor models on selected tasks from tau3-bench banking, AppWorld and EnterpriseOps-Gym, plus results on ALFWorld. Those are the paper's results, not MACE results.

Zep's current product documentation also states that:
- a Skill version is immutable and can be compared/restored;
- each Skill retains history and supporting evidence;
- deleting a source Trace removes Skills that cite it;
- each invocation can record its outcome.
See https://help.getzep.com/v4/agent-skills and https://help.getzep.com/agent-skills/manage-skills.

Other adjacent capabilities:
- **Zep/Graphiti:** bi-temporal edges, source episodes, invalidation of old facts, and hybrid vector/full-text/graph retrieval: https://help.getzep.com/graph-overview
- **Mem0:** entity-aware multi-signal retrieval and an ADD-only extraction pipeline are described in its platform documentation; its own benchmark repository reports LongMemEval V1 94.4% Top-200 and LoCoMo 92.5%. These are vendor-maintained results and not directly comparable to LongMemEval-V2: https://github.com/mem0ai/memory-benchmarks
- **Letta:** Git-backed editable MemFS, persistent agent memory and background dreaming/consolidation: https://docs.letta.com/configuration/memory

MACE should not claim novelty for temporal graphs, source-linked skill versions, skill compilation or editable memory files. The research opportunity is the conjunction of explicit procedure contracts, independently resolved evidence, execution-authority separation, replay-based admission, counterfactual procedure utility and full lifecycle accounting.

## Three separate artifacts

### 1. Procedure proposal — untrusted candidate output

The schema is at schemas/procedure-proposal-v0.1.schema.json. A synthetic read-only release-gate example is at fixtures/procedure-proposal-release-gate-v0.1.json. It performs no deployment, approval or mutation; its trace identifiers and digests are illustrative fixtures, not verified real execution evidence.

A compiler or agent proposes a procedure with:
- intended task family and objective;
- applicability/environment predicates;
- required permissions and declared effects;
- ordered typed actions with per-step preconditions, expected observations and failure behaviour;
- postconditions, known failure modes and compensation/rollback suggestions;
- source Trace identifiers, source event identifiers and trace digests.

The proposal schema deliberately has no trusted, verified, approved, maturity or admitted field. A candidate cannot grant authority to its own output. Source references and proposed assertions are untrusted until independently resolved.

### 2. Admission receipt — independent authority output

The schema is at schemas/admission-receipt-v0.1.schema.json.

A separate evaluator/verifier issues one of:
- admitted;
- rejected;
- quarantined.

The receipt binds the proposal digest, verifier identity and version, source-evidence audit, isolated replay results, negative-control results, policy checks, environment digest, validity interval, predecessor/revocation lineage and an attestation signature.

The candidate/compiler does not possess the signing authority. A receipt is valid only when its signature is verified against an out-of-band trust root and the proposal digest matches exactly. The schema is a proposed contract; no signer, key service or verifier is implemented by this increment.

### 3. Runtime-use receipt — per-invocation evidence

The schema is at schemas/runtime-use-receipt-v0.1.schema.json.

Every invocation records:
- exact admitted skill version and attestation digest;
- executor, task and environment digests;
- applicability and permission decisions;
- precondition checks;
- executed step IDs and observed outcomes;
- postcondition verification;
- failure, compensation, rollback or escalation;
- independently verified terminal outcome.

A skill's past admission does not replace checking current applicability, authorization and source status at runtime.

## Lifecycle and trust boundaries

1. **Capture:** record ordered tool calls, inputs/outputs (or protected digests), observations, environment state, declared permissions and an externally verifiable terminal outcome.
2. **Reflect:** propose a procedure and map each claim/step to one or more source events. Reflection cannot approve its own proposal.
3. **Curate:** an independent authority resolves every source event and checks whether each step is supported by the trace. Unsupported steps are removed or cause rejection.
4. **Qualify:** static schema, action/effect, permission, policy and precondition checks run before isolated replay. Replay includes counterexamples, changed environments and negative controls.
5. **Admit:** issue an immutable, signed version receipt with explicit validity, source dependencies and rollback lineage.
6. **Apply:** the runtime verifies the signature, active version, source status, environment, permissions and preconditions before exposing or executing the procedure.
7. **Reconcile:** record the actual step sequence, effects, final outcome and independent verification.
8. **Revise/quarantine/revoke:** verified success can support a later version; verified failure, environment drift, policy change, source revocation or erasure can quarantine or revoke dependent versions. Old versions are auditable only while retention policy permits.

The core authority rule is: **the model that proposes a procedure cannot admit it, and the memory system cannot authorize the actions contained in it.**

## Paradigm shifts

- **From transcript retrieval to executable contracts:** a procedure is not merely text. It has preconditions, typed actions, effects, postconditions and failure behaviour.
- **From successful run to causal evidence:** replay and step ablations should identify which steps are necessary, safe and justified; avoid promoting incidental steps that happened during one successful run.
- **From generic retrieval to applicability checks:** select a skill only when task, environment, policy, permission and source validity match.
- **From static trust to outcome-conditioned qualification:** successful and failed uses influence the next admission decision, but ranking signals never override hard authorization and validity gates.
- **From single skill to compositional capabilities:** compose smaller procedures only when input/output contracts, effects, permissions and invariants are compatible.
- **From write-once skill to governed lifecycle:** corrections, source revocations, expiry, key rotation, replay and rollback must propagate to every derived version and cache.

## Benchmark plan

Compare the following arms on identical task seeds, executor versions, tools, permissions, environment fingerprints and downstream adjudication:
1. no procedural memory;
2. raw successful trace retrieval;
3. human-authored procedure/skill;
4. simple trace-to-skill compiler;
5. Zep Agent Skills, where the public benchmark adapter and reproducible configuration are available;
6. MACE evidence-gated procedure compilation;
7. fine-tuning or policy-training reference, where compute/data/license budgets permit.

Task suites:
- tau3-bench banking and authorization-sensitive workflows;
- AppWorld tool-use workflows;
- EnterpriseOps-Gym and other reproducible operations tasks;
- code repair and repository-specific build/test procedures;
- environment gotchas and multi-step application workflows;
- policy/environment drift;
- contradictory traces, partial failures and poisoned "successful" traces;
- revoked/erased source evidence;
- cross-model transfer and composition.

Report separately:
- verified task pass rate and success transfer by executor/model/effort;
- unsafe/unauthorized action rate;
- precondition precision and missed-applicability rate;
- required-step precision/recall and unsupported-step rate;
- source-event coverage and claim-to-evidence completeness;
- stale, revoked or expired skill invocation rate;
- counterexample detection and skill quarantine latency;
- version regression rate and rollback success;
- cross-model transfer delta;
- tokens, latency, compute and maintenance cost per independently verified success.

Do not compare Zep's reported transfer gains with MACE results unless task sets, executor models, prompts, tools, budgets and success adjudication are matched. Report the source paper's numbers as external reference only.

## Initial acceptance gates

1. Candidate proposal, independent admission receipt and runtime-use receipt have distinct schemas and trust boundaries.
2. Proposals cannot self-assert trust/admission; schemas reject unknown fields.
3. Every action has declared preconditions, effects, expected observation and failure behaviour.
4. Every admitted version binds trace/event digests, environment, policy, verifier and replay receipts.
5. A revoked source invalidates dependent versions and prevents their runtime use.
6. A procedure cannot execute unless current permissions and all applicable preconditions pass.
7. A runtime-use receipt records actual execution and independent outcome verification.
8. All Rust checks run on an exact head before performance or safety claims.

## Relationship to MACE

- DSM #13893 / PR #13895 focuses on episodic/semantic memory's counterfactual decision utility.
- This programme extends those evidence and lifecycle controls to procedural memory.
- AML Add/Search remains a separate participant interface. Skill execution, final-answer generation or authority grants do not belong in its Add/Search endpoints.
- The MACE Research, AML Academic/Open-source and AML Commercial profiles remain separate; model eligibility and benchmark scores must not be conflated.

## Current status and non-claims

This branch contains a research specification, three JSON Schemas and one synthetic candidate proposal fixture. The fixture was read back and its JSON structure, step/source-event references and absence of self-asserted trust fields were checked. The JSON Schemas have been parsed as JSON but have not yet been run through a full JSON Schema validator. No procedure compiler, admission authority, signer/key service, replay sandbox, runtime guard, benchmark adapter, test result, transfer improvement or production feature is implemented. Vendor results are not MACE results.
