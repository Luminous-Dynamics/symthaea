# Proof-scope equivalence and paired comparability — protocol v1

**Issue:** [#13537 — MATH-SCOPE-EQUIVALENCE-001](https://github.com/Luminous-Dynamics/symthaea/issues/13537)  
**Programme:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Paired receipt:** [#13345 — MATH-LOCAL-GLOBAL-RECEIPT-001](https://github.com/Luminous-Dynamics/symthaea/issues/13345)  
**Status:** proposed protocol. No equivalence-tier implementation, model run, or Symthaea score is claimed.

## 1. Research question

Can local and global mathematical judgments be compared when their textual representations differ, without weakening the exact-scope guarantee that makes the local-to-global consistency experiment valid?

The existing LocalGlobalPairV1 contract uses strict digest equality. That remains the primary, safest cohort. This protocol adds a separately reported secondary track for representationally different contexts whose equivalence is certified by deterministic typed normalization or Lean-kernel-checked transport proofs.

The scientific object is not string similarity. It is a proof that the compared contexts and target propositions denote the same mathematical problem under a pinned formal environment.

## 2. Why a second comparability tier is useful

Strict byte/digest identity prevents accidental context drift, but it can exclude semantically harmless differences such as:
- alpha-renaming of bound variables;
- reordering independent hypotheses;
- equivalent syntax or notation;
- definitional unfolding and folding under the same environment.

At the other extreme, embeddings, fuzzy matching and LLM judgments can make genuinely different scopes look similar. They must never establish comparability.

The protocol therefore separates **identity** from **equivalence** and keeps their results apart.

## 3. Comparability tiers

### Tier 0 — exact identity (existing primary result)

Require the current local/global source, theorem, goal, hypothesis, premise, conclusion and warrant digests to match exactly. Existing CIRR eligibility and results are not modified by this protocol.

### Tier 1 — deterministic typed alpha/structural equivalence

For the bounded propositional seed, normalize only typed AST structure:
- assign bound variables canonical positional identities;
- preserve free variable identities and types;
- preserve connective and argument structure;
- sort only collections explicitly declared unordered, such as a set of independent hypotheses;
- preserve premise multiplicity and all domain/scope information;
- version and hash the normalization implementation.

Do not perform textual substitution on variable names. Do not sort or rewrite a sequence whose order has mathematical meaning. An alpha-normalized match establishes only the equivalence covered by the normalization definition.

### Tier 2 — kernel-certified equivalence

For supported non-dependent propositions in a frozen Lean environment, require proof artifacts for both directions:
- local context entails global context;
- global context entails local context;
- local target is equivalent to the global target under the mapped binders;
- each required premise/warrant correspondence is discharged or explicitly excluded from the matched claim.

All proof terms must be checked by Lean's kernel against the exact source/toolchain/import closure. Record declaration identities, type digests, toolchain, Lake manifest, axiom/admission profile and checker output.

The initial v1 implementation must not attempt arbitrary dependent-context transport. Dependent binders, local instances, typeclass parameters, universe changes or unsupported telescope transformations remain unknown/ineligible unless a later protocol defines and validates them.

### Tier 3 — unknown / ineligible

Missing proof direction, incomplete source identity, unsupported context shape, changed domain or unresolved translation means the pair remains ineligible. One-way entailment is not equivalence.

## 4. Formal comparison object

A proposed ScopeEquivalenceV1 receipt binds:

- pair ID and protocol/schema/normalizer versions;
- local and global source repository, commit, path and declaration IDs;
- Lean/Mathlib toolchains, Lake manifest and import closure;
- exact elaborated target and context type identities;
- binder map, binder kinds/types, universe parameters and local-instance context;
- ordered premises and explicitly unordered hypotheses;
- original and normalized AST/type digests;
- comparability tier and exclusion reason;
- alpha/structural normalization evidence, where used;
- forward and reverse Lean proof artifact references and digests, where used;
- kernel replay receipt for every transport proof;
- axiom/admission profile and unresolved obligations.

The evaluator may use the receipt to decide whether the pair is comparable. The candidate model must not receive evaluator-only equivalence proofs if those proofs reveal the task's answer or gold disposition.

## 5. Paired experiment invariants

1. Tier 0 remains byte-for-byte unchanged.
2. Tier 1 is restricted to transformations explicitly listed in the versioned normalizer.
3. Tier 2 requires both directions of every equivalence obligation.
4. Any premise, hypothesis, binder type, domain, universe or target change not covered by a checked transport proof is excluded from the equivalence cohort.
5. No fuzzy similarity, model confidence, graph centrality or agreement vote can set comparability.
6. Every local/global paired result retains the same frozen candidate task, warrant, model snapshot and manipulation condition.
7. Global-context pressure may vary only in the presentation dimension defined by the paired experiment; it may not add a mathematically useful premise.
8. Hidden evaluator labels, witnesses and equivalence receipts remain outside candidate-visible input.
9. Unknown and unsupported cases remain visible in the exclusion ledger.
10. A false comparability finding invalidates that affected cohort pending adjudication.

## 6. Seed fixtures

Positive:
- exact-identity control;
- alpha-renaming of bound propositional variables;
- permutation of independent hypotheses;
- equivalent implication/disjunction form, with a precise AST-level translation;
- definitional rewrite under a pinned environment.

Negative:
- add a strengthening assumption;
- remove a necessary assumption;
- change a binder type or domain;
- introduce free-variable capture during alpha-renaming;
- collide fully qualified names while changing namespaces;
- provide only one direction of entailment;
- change the target while retaining a superficially similar expression.

Unknown:
- dependent binder/telescope transformation outside the v1 supported fragment;
- unresolved local instance or typeclass context;
- opaque definition whose required equivalence is not established;
- incomplete source/environment identity.

Every fixture must state whether it tests syntax, definitional equality, logical equivalence or scope mutation. Do not conflate these authorities.

## 7. Metrics and analysis

Report separate tables and separate denominators for:
- exact-scope CIRR (Tier 0);
- alpha/structural-scope CIRR (Tier 1);
- kernel-certified-equivalence CIRR (Tier 2);
- candidate-pair count, eligible count and exclusions by tier;
- false-comparability rate on non-equivalent controls;
- proof-transport replay rate and axiom/admission profile;
- unknown/abstention rate by scope feature;
- local rejection coverage and false-rejection rates;
- global invalid-warrant reuse over all independently invalid warrants;
- per-theorem-family results and cluster-aware uncertainty.

Do not pool tiers into one headline. A larger denominator is not a quality improvement unless comparability remains correct. A model that rejects every warrant must not appear stable simply because it rarely creates eligible local-rejection cases.

## 8. Relation to neighboring benchmarks

- **ProofGrid:** use its fine-grained proof-checking and first-failure diagnostic ideas; the proposed CIRR remains a distinct paired local-to-global measure.
- **FormalMATH:** use its semantically validated formal theorem tasks for matched prover-side evaluation, not as a proof of explanation quality.
- **TheoremGraph / LeanGraph:** reuse formal dependencies and external graph provenance. Graph matches are retrieval candidates, not scope-equivalence authority.
- **LeanGraphSearch:** retain its retrieval results as an upstream baseline. Better premise retrieval does not certify that the retrieved theorem statement is the same problem.
- **Human transfer:** use a separately governed delayed-transfer study; scope equivalence does not establish learning.

## 9. Execution sequence

1. Freeze the Tier 0 regression fixture and digest outputs.
2. Define the typed propositional alpha-normalizer with deterministic serialization.
3. Add Tier 1 positive/negative metamorphic tests.
4. Build a small Lean proposition-transport proof harness for Tier 2.
5. Add a versioned receipt schema and validator tests without weakening LocalGlobalPairV1.
6. Have two reviewers independently audit the fixture labels and transport obligations.
7. Compare exact and equivalence-tier eligibility coverage on a held-out theorem-family split.
8. Only then run local/global model trials and publish tier-specific CIRR with uncertainty.

## 10. Prospective acceptance gates

- 100% reproducible Tier 1 normalization outputs on frozen fixtures.
- 100% rejection or unknown classification for seeded non-equivalent contexts.
- Both kernel-checked directions for every Tier 2 equivalence claim.
- Zero observed false comparability in the frozen negative controls.
- Existing Tier 0 fixture digests and CIRR remain unchanged.
- Every ineligible pair has a machine-readable reason.
- Results include per-tier numerators, denominators, exclusions and family-aware intervals.

These are prospective gates, not measured Symthaea performance.

## 11. Claim ceiling

- Alpha-equivalence is not arbitrary mathematical equivalence.
- Definitional equality depends on the pinned Lean environment.
- One-way entailment is not logical equivalence.
- Finite propositional normalization does not solve dependent-context equivalence.
- A valid scope-equivalence receipt does not certify the candidate's proof plan.
- A small eligible cohort cannot support broad reliability claims.
