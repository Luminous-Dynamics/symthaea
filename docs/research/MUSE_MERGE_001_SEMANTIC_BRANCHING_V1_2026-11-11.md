# MUSE-MERGE-001 — Semantic Branching and Conflict-Aware Musical Merges

**Date:** 11 November 2026  
**Status:** Research contract / reference-model proposal; not a production implementation  
**Tracking issue:** [symthaea #11669](https://github.com/Luminous-Dynamics/symthaea/issues/11669)  
**Branch:** `research/muse-semantic-branch-merge-v1`

## 1. Thesis

Melothaea should treat a musical composition as a persistent, versioned semantic object that can be explored on independent branches and merged under explicit musical, provenance and authority constraints.

The key distinction is:

```
shared editing
  !=
safe semantic merge

same track or disjoint time range
  !=
independent musical change

valid symbolic merge
  !=
qualified audio realization
```

A version-control layer that merely stores snapshots, records edits or applies last-writer-wins does not solve the musical problem. A harmony change may invalidate a melody on another track; a tempo-map change may invalidate absolute-time event projections; an articulation edit may alter the acoustic tail while preserving the score event.

The new research question is whether the system can prove that two changes are independent before automatically combining them—and preserve both branches when that proof is unavailable.

This is a capability proposal. No current Melothaea implementation or competitor is claimed to meet or fail this contract.

## 2. Scope and reuse boundaries

This contract owns only the semantics and deterministic evaluation of branching, three-way musical merge and conflict classification.

Reuse rather than duplicate:

| Existing programme | Boundary to reuse |
|---|---|
| [#10051 — MEL-013A](https://github.com/Luminous-Dynamics/symthaea/issues/10051) | Durable Studio transactions, artifact identities, ancestry, multiple contributors, consent verification boundary |
| [#11319 — creative agency](https://github.com/Luminous-Dynamics/symthaea/issues/11319) | Protected material, proposal/approval/commit states, autonomy levels, human agency |
| [#11356 — VocalEditPatch](https://github.com/Luminous-Dynamics/symthaea/issues/11356) | Typed local edits, target IDs, dependency cones and rerender invalidation |
| [#9044 — renderer output lineage](https://github.com/Luminous-Dynamics/symthaea/issues/9044) | Actual renderer input/output evidence and receipt status |
| [#11389 — score-to-audio conformance](https://github.com/Luminous-Dynamics/symthaea/issues/11389) | Independent score/audio qualification |
| [#9184 — human-calibrated edit metrics](https://github.com/Luminous-Dynamics/symthaea/issues/9184) | Blinded musician judgments and unseen-renderer calibration |
| [#10417 — external benchmark ledger](https://github.com/Luminous-Dynamics/symthaea/issues/10417) | Version-pinned comparison protocols |

There must be no parallel artifact registry, authorship database, identity provider, consent ledger, score ontology, renderer or generic music-edit benchmark created by this work.

## 3. Industry context

Public product documentation reviewed on 11 November 2026 establishes that collaboration itself is not a differentiator:

- [Suno Shareable Workspaces](https://help.suno.com/en/articles/14616961) provide invited shared workspaces, with owner controls and collaborator creation/remix/extension.
- [Moises Studio](https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/) announced a browser-based collaborative workspace with simultaneous recording, MIDI production, arranging and editing.
- [Veena collaboration](https://www.veena.studio/help/collaboration) documents live shared editing, per-person undo, and latest-change-wins for simultaneous edits to the same control.

These are comparisons of documented product behavior, not audits of proprietary internals. They do not establish that competitors lack unpublicized merge mechanisms.

The opening is narrower and more technically defensible: musician-readable, score-semantic conflict detection; explicit branch ancestry; authority-aware merges; deterministic replay; and evidence invalidation when edits change rendered inputs.

## 4. Semantic object model

### 4.1 Immutable composition version

A merge subject is a canonical composition/event graph, not a rendered WAV or opaque DAW project container.

A version resolves to:

- canonical graph digest and schema identifier;
- exact parent version/artifact identities in the existing lineage system;
- stable PartId, event, motif, section and relationship IDs;
- time-domain and tempo-map identity;
- declared musical constraints;
- protected-content declarations and authorization references;
- provenance for source and edited events;
- separately referenced performance/render artifacts and their evidence status.

A version is immutable after commit. A new edit creates a new version. A version identifier or branch name is not a substitute for hashing canonical graph bytes.

### 4.2 Typed patch

A patch is a preconditioned transaction against one exact base version. At minimum it binds:

- transaction ID and idempotency key;
- base-version digest and schema;
- actor identity reference and authorization evidence;
- operation type and stable target IDs;
- expected prior values or target digest;
- proposed values;
- operation read-set and write-set;
- declared musical intent and protected predicates;
- dependency/impact evidence;
- preconditions and postconditions;
- exact result graph digest and replay receipt.

A patch based on a stale digest is not applied. Rebase is a new evaluation against the new base; it is never a blind rewrite of the old transaction's base identifier.

### 4.3 Operation semantics

Operations should be typed by musical meaning, not arbitrary JSON paths. Initial examples:

- change pitch, velocity, duration or articulation of selected score events;
- alter a named instrument/performer assignment;
- replace a motif occurrence while preserving or explicitly transforming identity;
- change a chord/harmonic function;
- change a tempo-map segment or meter declaration;
- alter a phrase, section or formal boundary;
- change a part's orchestration role.

An operation must declare which facts it reads and writes. For example, a pitch edit may read the event identity, part, register constraints, harmonic context and voice-leading relationships—not just the pitch field.

The time representation must remain explicit. Beat-relative and absolute-time edits must not be treated as interchangeable. The merge layer must preserve the source representation, including irregular/asymmetric rhythmic structures, non-equal-tempered pitch information and unresolved analyses. It must not force every composition into a single Western meter, tuning or beat-grid assumption.

## 5. Musical dependency model

Let (G) be a composition graph and (P) a typed patch. Its dependency footprint consists of:

```
Footprint(P, G) = {
  read facts,
  write facts,
  related constraints,
  time intervals,
  derived projections,
  authority/protection dependencies
}
```

The footprint is not simply a track ID or time interval. It may include cross-part relations and downstream projections.

Dependency families:

1. **Identity:** event, motif, section, part and instrument identities.
2. **Temporal:** beat/frame mapping, tempo map, meter, onset, duration and event ordering.
3. **Harmonic/contrapuntal:** chord membership, harmonic function, voice-leading and dissonance constraints.
4. **Rhythmic:** pulse/cycle, subdivisions, polyrhythm, groove, accent and phrase relations.
5. **Orchestration:** register, role, doubling, balance and instrument assignment.
6. **Formal:** motif recurrence, phrase/section boundaries and long-range structural obligations.
7. **Performance:** articulation, expression, note tails, coarticulation, acoustic overlap and renderer control dependencies.
8. **Authority:** contributor, protected material, purpose, scope and revocation state.

Unknown dependency is a first-class result. An operation cannot be classified as independent merely because the current graph lacks a recognized relationship edge.

## 6. Conflict and commutativity semantics

### 6.1 Conservative conflict classes

| Class | Example | Default handling |
|---|---|---|
| Direct write overlap | Both branches change one event's pitch | Conflict / human decision |
| Same event, distinct properties | Pitch vs articulation | Operation-specific proof required |
| Cross-part relation | Chord change vs dependent melody note | Revalidate or conflict |
| Time-domain interaction | Tempo edit vs absolute-time onset edit | Conflict or explicit rebase |
| Global invalidation | Meter/form/tempo change | Revalidate affected projections |
| Render invalidation | Score changed after WAV render | Mark old render stale |
| Authority conflict | Protected motif edited without scope | Reject |
| Unknown dependency | Semantics insufficient to prove independence | Unknown; no automatic merge |

### 6.2 Commutativity condition

For candidate patches (P_a) and (P_b), evaluated against the same immutable base (G_0), automatic commutativity requires more than disjoint field writes.

The strict reference condition is:

```
apply(apply(G0, Pa), Pb)
  ==
apply(apply(G0, Pb), Pa)
```

where equality is equality of canonical graph content and the resulting operation/provenance semantics, not merely equality of selected numeric fields.

Both application orders must also satisfy all declared hard musical invariants, protected-content predicates, authorization checks and schema constraints. If this proof is not available, the pair is not automatically mergeable.

A later optimization may establish domain-specific semantic equivalence weaker than byte-identical result equality, but each equivalence must be explicit, versioned and independently tested. Do not assume all musical operations are associative or commutative.

### 6.3 Merge outcome type

The result must be one of:

- `SafeAutomaticMerge`: operation order is proven irrelevant and all hard invariants hold.
- `MergeWithRevalidation`: graph edits are jointly applicable, but declared constraints or derived artifacts require recalculation.
- `HumanDecisionRequired`: multiple valid outcomes exist or creative intent conflicts.
- `Conflict`: no jointly admissible result exists under current constraints.
- `Unknown`: dependency, evidence, authority or durable state is insufficient.

Unknown is never coerced to SafeAutomaticMerge. A human choice resolves an aesthetic conflict; it does not waive hard protection or authority requirements.

## 7. Three-way merge protocol

Given common base (G_0), branch A tip (G_A), and branch B tip (G_B):

1. Resolve and verify all three immutable versions and their ancestry.
2. Confirm that both branches actually descend from the declared base. If the common base is ambiguous or lineage is incomplete, return Unknown.
3. Extract typed patch sequences (Delta_A = G_A - G_0) and (Delta_B = G_B - G_0), retaining exact operation IDs and provenance.
4. Revalidate current actor authority, protected-content scope and any relevant revocation.
5. Compute the direct and transitive musical dependency footprints.
6. Build a conflict matrix, including pairwise interactions and higher-order shared constraints.
7. If SafeAutomaticMerge, apply a deterministic canonical order and verify that the reverse order produces the same canonical graph.
8. If MergeWithRevalidation, create an uncommitted candidate and discharge its specified constraints before commit.
9. If HumanDecisionRequired, present the base, both branch deltas, alternatives and conflict explanations. Do not silently prefer one branch.
10. If Conflict or Unknown, preserve both branch tips and emit a non-committable report.
11. Commit only after compare-and-set of the expected branch heads and durable transaction preparation.
12. Independently reload and replay the result, verify ancestry/patch IDs and graph invariants, then commit the new immutable version.

The merged version has both branch tips as ancestry parents in the existing artifact DAG. A merge does not erase either source history.

A rollback is a new verified transaction that creates a successor version. Historical provenance is not deleted.

## 8. Transaction and failure model

The reference lifecycle is:

```
PREPARED
  -> BASES_VERIFIED
  -> CONFLICTS_ANALYZED
  -> MERGE_READY | AWAITING_DECISION
  -> APPLIED
  -> INDEPENDENTLY_VERIFIED
  -> COMMITTED
```

Failure states include Rejected, Conflict, Unknown and InDoubt.

Required properties:

- expected-base compare-and-set;
- idempotent transaction identity and retry;
- deterministic canonical serialization;
- independently reproducible conflict analysis;
- no branch-head update before durable preparation;
- in-doubt reconciliation after uncertain commit acknowledgement;
- no blind retry after an ambiguous durable/external effect;
- preservation of both branch references and the full patch-to-result mapping.

A successful symbolic merge does not update the authority of an actor, grant new rights, or make all source contributions jointly reusable for every purpose.

## 9. Renderer and evidence boundary

The merge layer works on the symbolic composition graph. A valid graph merge is not evidence of a valid performance.

If the merged graph changes any source input to an earlier performance artifact, the old rendered audio is not the realization of the new version. Mark derived artifacts stale or invalidated according to their exact dependency set.

The new version proceeds through the existing qualified renderer and independent observation path:

```
merged score
  -> qualified performance schedule
  -> observed renderer invocation
  -> exact output bytes / PCM
  -> independent score-to-audio conformance
  -> evidence-qualified performance
```

Reuse #9044 and #11389. Do not duplicate their renderer receipt, timing calibration or acoustic observation systems here.

A merge receipt proves only the declared graph transformation, its authority/precondition checks and the invariants actually revalidated. It does not prove musical quality, listener preference, acoustic fidelity or authorship in law.

## 10. Deterministic reference benchmark

Create an isolated finite reference evaluator before any Studio integration. Its first corpus is synthetic and known-answer; it does not require a model checkpoint, audio renderer or human panel.

Minimum fixtures:

| ID | Branch A | Branch B | Expected result |
|---|---|---|---|
| M01 | Change event E1 on part A | Change event E2 on independent part B | SafeAutomaticMerge |
| M02 | Change E1.pitch to 60 | Change E1.pitch to 64 | Conflict |
| M03 | Change E1.pitch | Change E1.articulation | Operation-specific proof; otherwise Unknown |
| M04 | Change tempo-map segment | Change an absolute-time onset | Conflict or explicit rebase |
| M05 | Change chord function | Change dependent melody note | Revalidate or conflict |
| M06 | Change separate bars | Both depend on one phrase constraint | Detect relational conflict |
| M07 | Change global meter | Reuse old frame projection | Revalidation required |
| M08 | Edit protected motif | Actor lacks scope | Reject |
| M09 | Patch uses stale parent digest | Branch tip changed after preview | Reject stale base |
| M10 | Exact same idempotency key retried | Same prepared merge | No duplicate result |
| M11 | Distinct valid aesthetic alternatives | Incompatible intentions | HumanDecisionRequired |
| M12 | Renderer/patch identity changes after preview | Commit stale evidence | Revalidation required |
| M13 | Commit acknowledgement is lost | State cannot yet be resolved | InDoubt |
| M14 | Parent bytes substituted or scope revoked | Re-evaluation before commit | Reject / Unknown |
| M15 | Irregular rhythm or non-equal-tempered field | Unrelated edit elsewhere | Preserve native representation |

Add property-based tests for:

- commutativity only for explicitly qualified operation pairs;
- counterexamples to false commutativity;
- idempotent replay;
- canonical serialization determinism;
- stable identity and unaffected event preservation;
- stale-base rejection;
- provenance graph acyclicity;
- rollback as a new transaction;
- invalidation propagation to dependent performance artifacts.

Do not claim arbitrary merge associativity: musical constraints, merge decisions and approval order can make outcomes order-sensitive.

## 11. Qualification scorecard

### Deterministic targets (prospective)

- 100% exact expected result on frozen compatible known-answer fixtures.
- 100% rejection or escalation for direct and relational conflict controls.
- Zero false-safe automatic merges on the frozen adversarial suite.
- 100% stale-base and unauthorized-operation rejection.
- 100% deterministic replay and idempotent retry for supported transactions.
- 100% preservation of unaffected stable event IDs, PartIds and canonical event content.
- No Unknown or InDoubt state may become committable without reconciliation.

For every metric report counts, denominators, exact fixture manifest, version, confidence bounds where sampling applies, and all failure/Unknown outcomes. Deterministic fixture pass rates are not human collaboration results.

### Later human qualification

Compare the merge preview against manual branch comparison and a shared-session/latest-change-wins control. Measure separately:

- false-safe conflict acceptance;
- unnecessary conflict escalation;
- musician time to understand and resolve a conflict;
- correctness of conflict explanations;
- ability to preserve or select multiple alternatives;
- perceived creative control and ownership;
- quality and usefulness of the final selected composition.

Use randomized, counterbalanced tasks and composition/participant-aware statistics. Determine sample size from pilot variance and power analysis. Keep musician cohorts, traditions and task families disaggregated where they have distinct semantics.

## 12. Relationship to public benchmarks

This is a separate capability from music-editing benchmarks:

- [SMEBench](https://arxiv.org/abs/2610.36715) provides 3,600 paired synthetic music-edit examples across twelve operations and a smaller real-audio complement. Reuse its source/target pairs where licensing permits; do not recreate its benchmark.
- [MMAE / ICASSP 2027 Audio Editing Challenge](https://audio-editing-challenge.github.io/) evaluates broad audio instruction following, consistency and exact match. It measures the result of an edit, not semantic multi-branch merge correctness.
- [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench) measures generation quality under its published evaluation protocol. It is not a merge or preservation benchmark.
- [RMI-Bench #10088](https://github.com/Luminous-Dynamics/symthaea/issues/10088) measures reciprocal interaction, not branch-merge correctness.

Keep all scores separate. A system can be an excellent generator and a poor editor, or a correct merge engine that does not create preferred music.

## 13. Execution plan

**F0 — finite reference contract**
- Freeze version, patch, dependency, conflict, outcome and receipt types.
- Implement a pure Rust reference evaluator outside production paths.
- Add M01–M15 and exact canonical expected outputs.
- Run formatting, tests and mutation tests on the exact head.

**F1 — dependency semantics**
- Add read/write sets and musical dependency edges.
- Test operation-pair commutativity against an independently implemented oracle.
- Add higher-order shared-constraint counterexamples.
- Keep unrecognized dependency classes non-committable.

**F2 — transaction integration**
- Reuse MEL-013A identity, ancestry, persistence and consent boundaries.
- Add compare-and-set branch heads, idempotency and in-doubt reconciliation.
- Independent replay must reload exact source and result graph bytes.

**F3 — Studio preview**
- Present a musician-readable three-way view: base, branch A, branch B, candidate result, conflict class and invalidated render spans.
- Support accept, reject, keep-both-as-alternatives and manual revision.
- Never auto-resolve aesthetic disagreements.

**F4 — realization and human studies**
- Re-render only approved result versions using qualified backends.
- Qualify output through #9044 / #11389.
- Calibrate merge explanations and preservation reports with blinded musicians under #9184.

## 13.1 Initial reference evaluator authored (12 November 2026)

A bounded standard-library-only Rust reference preflight evaluator is now authored at:

- `tools/mel-muse-merge/Cargo.toml`
- `tools/mel-muse-merge/src/lib.rs`
- `tools/mel-muse-merge/README.md`
- `.github/workflows/muse-merge-reference.yml`

The crate declares its own nested Cargo workspace so it can be built independently of the large parent workspace. It exposes explicit enums for authority, artifact integrity, intent, transaction state, merge outcome and stable reason codes. Its input is a declared patch footprint: stable targets, reads, writes, musical dependency keys, authorization state, integrity state and revalidation requirements.

The initial implementation covers:
- exact same-key idempotent replay as reuse of an existing result, not a second mutation;
- stale base/branch-head rejection;
- missing/revoked authority rejection and unknown authority abstention;
- artifact digest mismatch and unknown-evidence abstention;
- direct write/write conflicts;
- read/write dependency conflicts;
- cross-part musical dependency conflicts;
- human escalation for two valid alternative proposals;
- revalidation-required candidate outcomes for stale performance receipts or changed projections;
- an explicit registry for operation pairs qualified by a separate proof/test suite;
- no commit authority in any preflight result.

**20 Rust unit tests are authored**, including tests corresponding to the 15 frozen fixture IDs, symmetry checks, qualified-operation-pair handling, manifest ID/outcome consistency, unknown authority and unavailable-revalidation controls.

The 15-case JSON fixture manifest now carries the evaluator's decision inputs, and its Draft 2020-12 schema has been extended accordingly.

### Validation boundary

- The five crate/workflow files and the updated fixture/schema have been committed and read back from the dedicated research branch.
- The Rust crate has no external dependencies; its Cargo lockfile contains only the package itself.
- No local Cargo/rustc/rustfmt is available in the current authoring environment.
- The exact-head workflow is queued; compilation, formatting, tests and Clippy are **not yet claimed as passed**.
- The test harness currently checks manifest case IDs and declared outcomes by text inclusion; it does not deserialize or execute the JSON cases. The Rust unit tests construct their typed counterparts directly.
- A SafeAutomaticMerge result means only that the declared footprint preflight found no conflict. It never authorizes a commit. Actual patch application, both-order result equivalence, invariant checking, durable compare-and-set, commit and independent replay remain future stages.

## 14. Claim ceiling

This programme may establish that a declared set of musical edit operations can be merged with deterministic, replayable preservation and conflict guarantees under stated assumptions.

It does not establish:

- universal merge correctness for arbitrary music representations;
- aesthetic or perceptual superiority;
- safe live performance;
- human-equivalent musical understanding;
- legal authorship or copyright ownership;
- competitor inferiority;
- SOTA superiority.

The desired achievement is narrower and more valuable: a musician can explore without losing alternatives, combine compatible contributions without silent overwrites, and understand exactly which musical decisions were preserved, changed, invalidated or left unresolved.
