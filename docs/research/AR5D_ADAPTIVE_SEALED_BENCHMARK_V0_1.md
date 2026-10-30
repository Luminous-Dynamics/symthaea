# AR-5D — adaptive scientific benchmarks with sealed oracle verification v0.1

**Status:** task/evidence contracts authored; not compiled, executed, or security-qualified.  
**Tracking issue:** [#9332](https://github.com/Luminous-Dynamics/symthaea/issues/9332)  
**Implementation:** crates/domains/symthaea-architecture-engineering/src/adaptive_benchmark.rs  
**Dependency:** AR-5B direction allocation and AR-5C typed primitive qualification.

## Motivation

A fixed benchmark eventually becomes a target. Solvers can memorize its examples, exploit metadata, overfit recipe families, or learn a shortcut that does not measure the scientific capability. Adaptive benchmark generation can respond to those attacks, but a benchmark generator that also exposes the answer key or rewrites its own acceptance threshold is not a trustworthy evaluator.

AR-5D therefore separates the task concept, task recipe, public solver packet, private oracle, recipe-refinement adjudication and evidence receipt. The solver and the search policy receive only the public packet. The oracle and locked qualification instances stay in an independently operated verifier service.

This follows the task-concept/recipe refinement direction of AutoSciBench, the stepwise verification and multi-domain orientation of SciAgentArena, and the complete plan → implement → execute → analyze research cycle of FIRE-Bench. It adds explicit process separation, immutable commitments, task-version retention and resource-bound evidence receipts.

## Implementation surface

The new Rust module currently provides:

- TaskConcept: stable scientific construct including domain, data modality, reasoning capability, intervention and task family.
- TaskRecipe: immutable generator, public-template, environment and verifier-profile digests plus resource ceilings.
- PublicTaskPacket: exact solver-visible request; no oracle, secret seed, split, answer key, hidden shortcut labels or rubric fields.
- SealedOracleCommitment: commitment to verifier-held answer material with a random nonce of at least 256 bits.
- RecipeRefinementReceipt: binds prior and revised recipe versions and requires independent construct-validity adjudication.
- VerifierReceipt: binds task packet, solver artifact, environment, verifier profile, evidence bundle, stage scores and measured resource use.

Seven unit tests are authored for canonical/versioned digests, public packet non-disclosure at the schema boundary, concept/recipe binding, oracle opening tampering, construct-preserving refinement, receipt/task binding, resource limits and stage evidence integrity. These tests have not been run yet.

## Critical security boundary

Private Rust fields, omitted serde fields and SHA-256 hashes do not provide security against a solver in the same process or against someone who can inspect memory. The SealedOracleCommitment type is a protocol/data structure only. The actual oracle payload, nonce, split assignment and grading logic must remain in a different process or service with separate access control and audit logs.

Similarly, VerifierReceipt.validate_shape checks syntax, bindings and budget consistency; it does not verify an Ed25519 signature. A separate verifier client must validate the signature over a canonical receipt payload against a trusted key registry, check key revocation/rotation, and bind signer identity to an approved verifier deployment. A digest or signature-shaped string is not proof of authentic verification.

The service must never expose oracle-bearing Rust values, seeds, stack traces, private exceptions, grader metrics or split identifiers to the solver. It must not include hidden labels in task IDs, filenames, log ordering or timing.

## Concept × recipe × instance model

A TaskConcept describes the scientific construct and should remain stable across revisions. A TaskRecipe describes how the task is instantiated and verified. A task instance binds a specific recipe version to a generated input artifact, but its split and private oracle data belong to the verifier-side record, not the public packet.

This separation lets a recipe be hardened against shortcuts without silently changing the meaning of the task. Any change to the reasoning construct, intervention semantics, data modality or target capability requires a new concept identity or explicit expert adjudication.

## Qualification partitions

Maintain four partitions outside the solver-visible packet:

1. Development: visible tasks for agent and policy iteration.
2. Calibration: used for score thresholds, task difficulty and variance estimation; excluded from final superiority claims.
3. Sealed qualification: hidden instances held by the verifier; never fed back into search or threshold selection.
4. Rotating challenge: generated from locked concept/recipe families with secret seeds, retained version-by-version to track capability drift.

A task is never silently moved between splits. A sealed task exposed to a solver is burned and replaced by a new instance from a locked recipe family. The old exposure remains in the audit record.

## Adaptive hardening loop

1. Select a concept from the preapproved scientific construct registry.
2. Generate candidate recipes with pinned builder and environment digests.
3. Validate data provenance, executability, oracle determinism, task answerability and resource bounds.
4. Run diverse solver baselines under matched total budgets.
5. Diagnose shortcuts from permitted solver-visible traces and controlled probes.
6. Propose recipe refinement that closes the shortcut while preserving the original concept.
7. Run an independent construct-validity adjudication against the original concept.
8. Freeze the revised recipe hash and publish a new benchmark version.
9. Retain old recipes and instances; report performance on both old and new versions.

The solver or direction allocator may propose hardening tests, but must not accept its own recipe, edit the verifier's key, or set the final qualification threshold.

## Threat and negative-control matrix

| Threat | Control |
|---|---|
| Answer leakage through IDs, filenames or ordering | Randomized opaque IDs and metadata-only shortcut baselines |
| Secret-seed reconstruction | Verifier-side generation; high-entropy nonce; access log and chosen-seed attack |
| Recipe duplication across splits | Concept/recipe similarity audit and family-level split |
| Published-answer contamination | Time/version control, fresh generated instances and contamination probes |
| Filename/metadata shortcut | Metadata-only, row-order and label-prior baselines |
| Hidden-label probing | Rate limits, query audit and non-informative errors |
| Grader probing | Separate verifier process, fixed API and query budgets |
| Reward/receipt manipulation | Signature verification, artifact-bound receipts and replay tests |
| Construct drift during hardening | Independent expert adjudication and versioned construct digest |
| Generator/verifier correlated bug | Independent reference implementation and differential oracle tests |
| Difficulty inflation that changes the construct | Pair old/new variants and perform construct-validity review |
| Malicious solver feedback to the curator | Solver feedback is untrusted; use independent shortcut probes and adjudication |

## Evaluation metrics

Keep dimensions separate:

- scientific construct validity and task determinacy;
- sealed qualification capability and full-cycle evidence correctness;
- planning, implementation, execution and analysis performance;
- shortcut exploit rate and oracle leakage rate;
- recipe-refinement yield without construct drift;
- verifier precision, signature validity and replay rejection;
- task difficulty, solver variance and worst-family results;
- wall-clock, CPU, memory, tool calls, candidate executions and energy;
- robustness across rotating recipes and previously unseen concept instances.

No weighted aggregate may compensate for oracle leakage, invalid task ground truth, or broken evidence integrity with higher solver scores.

## Proposed targets (not achieved)

- Zero oracle or qualification-label exposure in adversarial access tests.
- Zero false oracle assertions on exhaustive finite-domain task families.
- 100% authenticated receipt verification against a trusted key registry.
- At least 95% expert-adjudicated construct validity before qualification admission.
- No material task-meaning drift during shortcut remediation.
- Family-level confidence intervals, failed/censored runs and worst-family results reported.
- Rotating sealed results predict transfer to independently constructed architecture/operator tasks.

These targets require calibration and preregistration. The 95% construct-validity target is not a result from this implementation.

## Relationship to external benchmarks

- AutoSciBench (arXiv:2610.05140, 4 October 2026) proposes concept/recipe generation and iterative benchmark hardening from solver feedback.
- SciAgentArena (arXiv:2606.12736, June 2026) provides about 200 stepwise-verified scientific tasks across five domains and reports uneven capability on open-ended problems.
- FIRE-Bench (PMLR 306, 2026) measures full-cycle insight rediscovery with evidence-grounded claims.
- ScienceAgentBench provides 102 expert-validated tasks from peer-reviewed publications.
- SciConBench uses a clean-room harness for scientific conclusion synthesis over roughly 9,000 questions.
- Paper2Agent (Nature, 16 September 2026) shows how research papers and code can be converted into testable agent-native research objects, while explicitly distinguishing faithful execution from analytical validity on open-ended questions.

AR-5D is not a reproduction of these systems. It is the integrity and adaptive-evaluation layer for Symthaea's architecture and primitive discovery programme.

## Next implementation increments

1. Implement an out-of-process verifier protocol and canonical signed payload format with actual Ed25519 verification and key rotation.
2. Add a sealed split registry, access audit and burn-on-exposure state transitions.
3. Add recipe generation for finite-state systems and causal synthetic tasks with exhaustive oracles.
4. Add shortcut baseline harnesses and construct-validity adjudication workflow.
5. Add rotating challenge generation and version-drift reporting.
6. Connect AR-5B direction allocation and AR-5C operator discovery to this verifier through receipts only.

## Claim ceiling

v0.1 only defines contracts and shape validators. It does not implement process isolation, secure key management, signature verification, benchmark generation, or adaptive qualification. Even a passing future synthetic campaign would qualify the benchmark mechanism on specified families; it would not establish real scientific or architecture discovery without independent transfer and replication.
