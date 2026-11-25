# Lean trust-closure assurance capsule — protocol v1

**Issue:** [#13603 — LEAN-TRUST-CLOSURE-CAPSULE-001](https://github.com/Luminous-Dynamics/symthaea/issues/13603)  
**Programme:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Related audit:** [#13575 — LEAN-AXIOM-AUDIT-INDEPENDENCE-001](https://github.com/Luminous-Dynamics/symthaea/issues/13575)  
**Status:** research protocol and initial fixture design; no general capsule emitter, validator, independent-checker run, or repository-wide trust qualification is claimed.

## 1. Purpose: make trust multidimensional

A proof is not one indivisible trust event. At minimum, an auditable theorem result has several different questions:

1. **Statement identity:** did the checker validate the exact theorem statement intended by the trusted challenge?
2. **Logical dependency closure:** which axioms, admissions, declarations, opaque values and imported definitions does the root actually depend on?
3. **Kernel acceptance:** did the official Lean kernel accept the exact exported proof artifact?
4. **Independent replay:** did a separately implemented checker accept that same artifact under an explicit axiom policy?
5. **Construction containment:** what executable code ran while elaborating/building the proof, and was that activity confined?
6. **Environment reproducibility:** are source, toolchain, imports, manifests and exported artifacts bound to immutable identities?
7. **Residual uncertainty:** what could not be inspected, replayed, or independently established?

The output is a **vector of assurance claims**, not a scalar trust score and not a single green badge. A theorem can be kernel-checked while statement fidelity is unknown; its logical closure can be complete while the build was not isolated; or the official kernel and an external checker can disagree. Those states must remain distinguishable.

### Central rule

**An empty axiom list is one observation, not a complete proof-assurance argument.**

The current Symthaea fresh per-root DFS is an evidence producer for logical dependency closure. It is not a proof checker, a complete threat-model implementation, or a replacement for Lean's kernel.

## 2. Why this layer is necessary

Lean's validation reference explicitly distinguishes whether a theorem has a valid proof from what the theorem statement means. It describes progressively stronger checks, culminating in a trusted challenge and comparator-based proof validation. The comparator route isolates proof construction, exports a proof artifact, checks that artifact outside the sandbox, compares theorem statements against the challenge, and can use several independently written kernels.

The same reference lists residual assumptions: challenge correctness, sandbox safety, comparator plumbing, and the absence of a common bug across all configured checkers. Those assumptions should be visible in every Symthaea receipt.

The repository's current axiom-audit work addresses a narrower issue. Lean issue #15226 reports order-sensitive under-reporting in imported-inductive axiom summaries. That is a reason not to treat cached axiom summaries as sole authority; it is not evidence that the Lean kernel accepted an invalid theorem. The fresh per-root traversal is a bounded audit response and must retain its own limitations.

References:
- Lean Language Reference, “Validating a Lean Proof”: https://lean-lang.org/doc/reference/latest/ValidatingProofs/
- Lean issue #15226: https://github.com/leanprover/lean4/issues/15226
- Lean Kernel Arena / nanoda: https://arena.lean-lang.org/checker/nanoda/

## 3. Two planes, seven assurance axes

Keep the following planes separate.

### A. Logical and theorem plane

| Axis | Question | Minimum evidence |
|---|---|---|
| Statement fidelity | Is this exactly the trusted target? | Challenge digest, elaborated target identity, transitive definition/import binding, comparator statement match |
| Logical closure | What assumptions and declarations support the theorem? | Fresh per-root typed dependency graph, axiom/admission inventory, unresolved-node accounting |
| Kernel replay | Did the official kernel accept the exact exported proof? | Exact export digest, kernel/toolchain identity, raw replay receipt |
| Independent replay | Did a different implementation accept the same export? | Checker identity and source/version, same export digest, exact result and axiom policy |

### B. Construction and environment plane

| Axis | Question | Minimum evidence |
|---|---|---|
| Construction containment | What code ran to produce the proof, and where? | Build command, sandbox policy/result, executable-component inventory, network/filesystem/process observations where available |
| Environment reproducibility | Can the same inputs be identified and reconstructed? | Source revision/blob hashes, Lean/Mathlib revisions, Lake manifest, import and artifact digests |
| Computational trust | Did native execution, compiler replacement, or external computation support a proof step? | Version-specific marker inventory, dependency evidence, explicit policy disposition |

A successful value on one axis cannot upgrade another. In particular:
- kernel_replay = satisfied does not imply statement_fidelity = satisfied.
- logical_closure = complete does not imply construction_containment = satisfied.
- two runs of the same kernel do not imply independent replay.
- an unsafe tactic or plugin used during construction is not automatically a logical axiom in the final proof term; its use belongs in construction provenance.
- a declaration being labelled unsafe is not, by itself, a complete malicious-code detector. Record executed code and build containment rather than treating one modifier as a security boundary.

## 4. Capsule data model

The strict JSON contract is defined in [LEAN_TRUST_CLOSURE_ASSURANCE_CAPSULE_V1.schema.json](LEAN_TRUST_CLOSURE_ASSURANCE_CAPSULE_V1.schema.json). The required top-level concepts are:

### 4.1 Subject and challenge

Bind:
- repository owner/name and immutable commit;
- exact source file path, blob digest and source revision;
- root declaration's fully qualified name and declaration kind;
- trusted challenge artifact identity and digest;
- elaborated target-type digest, universe parameters and a versioned serialization format;
- parser, notation, scope and relevant typeclass-instance environment identity;
- all transitive definitions needed to establish what the target statement means.

A pretty-printed string is useful for review but is not a stable statement identity by itself. A theorem with an unchanged short name but a changed namespace, binder type, universe, local instance, notation expansion, or imported definition must not silently retain the prior identity.

### 4.2 Logical dependency closure

Preserve the distinction between graph edges and the trust-bearing leaves.

Each node records:
- fully qualified Lean name and declaration kind;
- source module and import provenance;
- type digest and value digest where available;
- value availability (available, opaque_available, unavailable, or not_applicable);
- outgoing typed edges (type_reference, proof_reference, constructor, recursor_rule, inductive_parent, or module_import);
- whether the node is an axiom, admission, opaque declaration, quotient primitive, or unresolved reference.

The closure summary records:
- one independent traversal per root;
- algorithm identity and version;
- whether the environment was configured to expose imported values;
- whether only a visited set was used to break cycles;
- raw native axiom output as a diagnostic channel;
- exact custom axioms, foundational axioms and admission markers;
- unresolved/inaccessible declarations;
- deterministic sorted closure digest and the root-order permutation result.

Do not infer that an unavailable value has no dependencies. An inaccessible declaration, unrecognized constant kind, failed lookup, or incomplete import is unknown and blocks a complete-closure claim.

The audit should preserve both:
1. **Observed closure** — every reference and trust-bearing leaf the extractor actually observed.
2. **Coverage statement** — which Lean declaration kinds, environment paths and serialization forms the extractor has been tested to cover.

A complete result is scoped to that explicit coverage statement and pinned environment. It is never a universal claim about every Lean version or repository.

### 4.3 Axiom and admission policy

Axiom inventories are not binary. Store each axiom separately with:
- fully qualified name;
- classification (foundational, custom, admission, native_computation, or unknown);
- policy disposition (allow, reject, review, or unknown);
- rationale and policy version.

The policy must not silently allow all names merely because they are present in a configured baseline. A foundational allowlist is a versioned policy decision, not a fact discovered by the dependency walker. sorryAx and other admissions are explicit and rejected by default.

Lean's reference identifies propext, Classical.choice, and Quot.sound as standard axioms, and describes sorryAx as evidence of an incomplete proof. The benchmark may define stricter policies; every exception must be explicit and visible.

### 4.4 Version-aware native computation

Native computation is a trust boundary and must be interpreted against the exact Lean version.

The reference documents that native evaluation can enlarge the trusted computing base through the Lean compiler and declarations marked implemented_by; it also describes version changes to the native-computation axiom mechanism. In particular, the historical Lean.trustCompiler account is labelled for Lean versions up to 4.28, while Lean 4.29 and later use dedicated per-computation axioms for decide +native and related native computation.

Therefore:
- never hard-code one native marker as the only possible representation;
- capture the exact Lean version and all observed native markers;
- inspect implemented_by / extern metadata and relevant replacement implementations when the pinned API exposes them;
- distinguish kernel reduction from native execution;
- require an explicit policy disposition for every native-computation dependency;
- do not claim an external checker validated the compiler's calculation merely because it accepted the exported proof under an axiom policy.

The #print axioms output may help detect native markers, but the capsule records the version-aware audit and replay evidence separately.

### 4.5 Construction provenance

Record executable components involved in proof construction:
- macros, commands, elaborators and tactics;
- plugins and dynamically loaded modules;
- external processes and custom build commands;
- native evaluators and compiler invocations;
- environment variables and relevant tool versions;
- sandbox identity, policy digest and outcome;
- network, filesystem and process restrictions where observable.

Classify a component by when and how it runs (elaboration, build, export, replay, or runtime). Do not copy every transitive source dependency into this list: it is an execution inventory, not the logical proof graph.

A tactic may execute arbitrary code during construction. The assurance question is whether the resulting proof artifact was independently validated outside that code's influence, and whether the theorem statement was compared with a trusted challenge.

### 4.6 Kernel and independent replay

Every replay record binds the **same exact exported artifact digest** and records:
- checker implementation and version/source revision;
- checker role (official_kernel or independent_kernel);
- exact Lean/export format compatibility;
- allowed axiom policy and actual observed axiom profile;
- result (accepted, rejected, error, or not_run);
- raw log/receipt digest and exit status.

An independent replay may be marked satisfied only when the checker implementation is materially distinct from the official Lean kernel and checks the same export artifact. A second invocation of the official kernel is a repeated replay, not independent replay.

Use the current Lean comparator path as the primary high-assurance integration target. The reference describes lake comparator with --paranoid as checking through multiple independently written kernels. nanoda is one suitable independent implementation and is written in Rust. Other checkers can be added when supported by the pinned export format and policy.

The legacy lean4checker path can be an additional check, but it is not interchangeable with comparator: the reference notes that it trusts .olean structural correctness and does not itself validate the exported format. Record exactly which checker ran and what it protects against.

The capsule also carries a required `checker_diversity` reference with the registry digest, exact run-profile digest and profile ID. This avoids treating a list of external checker names as a diversity argument. The referenced profile records the six comparator execution paths, known shared algorithm lineages, unknown pairwise relationships, checker-specific outcomes and same-export digest binding. The independent-replay assurance axis may be `satisfied` only when the referenced checker-diversity profile is itself `qualified_under_policy`. A synthetic, inconclusive, rejected or not-assessed profile cannot upgrade that axis. See [CHECKER-DIVERSITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/13637).

### 4.7 Statement fidelity

The trusted challenge must be authored or reviewed in a trusted environment. The capsule should bind:
- challenge source and digest;
- all challenge imports and their digests;
- target declaration identities;
- elaborated type identities and relevant definition identities;
- comparator's statement-match result;
- a reviewer-facing rendering.

The challenge and solution may use different proof terms, tactic scripts or proof routes. They must not silently prove different statements. Statement matching is a separate axis from logical closure and kernel replay.

## 5. Assurance state semantics

Each assurance axis has exactly one of these states:

- satisfied: required evidence exists and all axis-specific checks passed;
- failed: evidence demonstrates a policy violation or invalid result;
- unknown: required evidence is missing, ambiguous, inaccessible or untested;
- not_applicable: the pinned policy explicitly declares this axis not required for this cohort, with a rationale.

The capsule disposition is:

| Disposition | Meaning |
|---|---|
| qualified_under_policy | Every required axis is satisfied; no required axis is unknown or failed; all policy exceptions are explicit |
| rejected | At least one required check demonstrates a violation or invalid proof/artifact |
| inconclusive | No definitive rejection is established, but one or more required axes are unknown |
| draft | The capsule has not yet been evaluated |

Do not compute a weighted average. A 100% score in one dimension cannot compensate for an unknown target statement or missing checker replay. A policy that makes an axis not applicable must state why and must not relabel it as satisfied.

## 6. Failure and mutation semantics

The validator must fail closed on:
- a missing source, manifest, import or artifact digest;
- a root declaration that cannot be resolved;
- a changed target type, universe, binder, instance or challenge import;
- an unknown constant kind or incomplete dependency walk;
- a missing opaque value where the selected policy requires its closure;
- an unresolved import or inaccessible dependency;
- an unapproved custom axiom, admission or native-computation marker;
- a proof export that differs from the artifact replayed;
- an independent checker that checked a different export;
- an independent-replay axis marked satisfied without a qualified, digest-bound checker-diversity profile;
- a checker-diversity profile whose registry or profile digest is missing or malformed;
- a checker error represented as acceptance;
- a build that escaped the declared sandbox constraints;
- a root-order change that changes a supposedly deterministic closure result.

A failure to compile the test harness is not evidence that a theorem is false. Record error or unknown where appropriate; do not relabel infrastructure errors as mathematical counterexamples.

## 7. Initial adversarial fixture matrix

The fixture design is stored in [LEAN_TRUST_CLOSURE_ASSURANCE_FIXTURES_V1.json](LEAN_TRUST_CLOSURE_ASSURANCE_FIXTURES_V1.json). It includes:
- clean proof and explicitly allowed foundational-axiom controls;
- direct/transitive custom axioms and sorryAx;
- imported-inductive and mutually recursive cycles;
- unresolved dependency and unavailable opaque value;
- theorem target substitution and environment digest drift;
- native computation with a version-aware expected classification;
- build-time metaprogram/plugin provenance separate from proof dependencies;
- malformed export rejection and kernel/independent-checker disagreement;
- root-order determinism.

Fixture outcomes distinguish reject from inconclusive. A missing dependency is never represented as an empty dependency set.

## 8. Evaluation metrics

Report each metric by Lean version, theorem family, checker, axiom policy and artifact cohort.

| Metric | Definition |
|---|---|
| Dependency recall | Seeded trust-bearing edges/leaves recovered by the fresh audit |
| False-negative count | Seeded trust dependencies missing from the audit |
| Unresolved detection | Fraction of missing/inaccessible declarations marked unknown or rejected |
| Root-order invariance | Identical closure and receipt digests across root-order permutations |
| Target substitution detection | Wrong-target mutations rejected before qualification |
| Environment binding accuracy | Source/toolchain/import/artifact mutations invalidate the capsule |
| Native classification accuracy | Version-specific native-computation cases classified correctly |
| Replay acceptance | Exact exported artifacts accepted/rejected by the official kernel |
| Independent replay agreement | Same-export agreement across distinct checker implementations |
| Isolation violations | Forbidden build-time actions detected or prevented |
| Inconclusive rate | Cases correctly withheld because evidence is insufficient |
| Accepted-capsule cost | Build, export, replay and audit cost per qualified result |

Do not optimize only for the proportion qualified: an unsafe validator could increase coverage by turning unknowns into passes. Report coverage, false negatives, false acceptances and abstentions together.

## 9. Implementation order

1. Freeze this assurance-vector protocol and strict schema.
2. Validate the fixture manifest and add the expected classifications to the research workflow.
3. Connect the current fresh per-root DFS as the logical-closure evidence producer.
4. Add the native-computation and opaque-value observations against the exact pinned Lean API.
5. Add a challenge/solution pair and comparator receipt for one small theorem.
6. Bind the same export to an independent checker where the pinned comparator setup supports it.
7. Add build-time component and sandbox provenance.
8. Run mutation controls and preserve every raw log, exported artifact and receipt digest.
9. Obtain an independent review of the threat model before claiming a qualified profile.

This order allows schema, fixtures, and deterministic validation to advance without waiting for a new broad CI runner. Kernel replay and checker claims remain pending until actual receipts exist.

## 10. Proposed first cohort

Start with a deliberately small cohort:
- And.comm and the current local action pilot;
- the imported-inductive reproduction from issue #13575;
- one direct custom axiom and one transitive custom axiom;
- one sorryAx control;
- one version-pinned native-computation example;
- one build-time tactic/plugin provenance example;
- one target-substitution challenge/solution pair.

Do not begin with Mathlib-wide trust closure. Prove the capsule contract and mutation semantics on small, independently inspectable cases first.

## 11. Claim ceiling

This protocol does not:
- replace Lean's kernel or prove the soundness of Lean's logic;
- prove a fresh DFS complete for all Lean versions and declaration forms;
- certify the informal mathematical meaning of a theorem without a trusted challenge/review;
- remove common-mode bugs across checkers or the export/comparator pipeline;
- prove a sandbox secure merely because it has a name or a passing job;
- certify arbitrary compiler, plugin, OS, filesystem, network or host behavior;
- establish that a proof's mathematical explanation or conceptual-role annotation is correct.

The deliverable is narrower and more useful: **a theorem-specific, replayable, multi-axis statement of what was checked, by whom, against which artifacts, under which policy, and what remains trusted or unknown.**
