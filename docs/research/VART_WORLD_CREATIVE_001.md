# VART-WORLD-CREATIVE-001 — Embodied Experience-Conditioned World Improvement

Status: preregistration scaffold only. Runtime integration is intentionally not claimed by this document.

## Research question

Can Symthaea use embodied experience in a persistent authored world to propose and select bounded revisions that improve independently declared world properties, while preserving physical validity, provenance, counterfactual separation, and edit-authority boundaries?

## Scope

This experiment evaluates the closed loop:

1. enter a committed world;
2. explore under grounded support/obstacle gates;
3. acquire before/after observations;
4. form an immutable experience episode;
5. state a prospective revision hypothesis before outcome inspection;
6. generate bounded counterfactual revisions;
7. evaluate only physically admitted or explicitly evaluable candidates;
8. bind the selected candidate to the exact proposal;
9. independently replay the proposal and authority checks;
10. apply the typed world edit;
11. revisit the edited world;
12. measure consequences without collapsing them into a single quality score.

The experiment does **not** claim consciousness, sentience, general intelligence, artistic genius, or universally good world design.

## Required provenance boundary

The following identities must be frozen before execution:

- source HEAD and TREE;
- World Forge qualified-parent receipt;
- Symthaea/Reality Ledger qualified-parent receipt where used;
- world fixture digest;
- seed/randomness policy;
- rendering backend and GPU identity where visual evidence is used;
- physics/support/obstacle backend identity;
- experiment plan digest;
- measurement implementation digest;
- baseline-policy implementation digests.

A later source correction creates a new evidence lineage. Qualification must not be repaired in place.

## WC-001 — RevisionHypothesis / RevisionOutcome

Before a selected world revision is applied, record a `RevisionHypothesis` with at least:

- trial_id;
- experience_episode_digest;
- source_world_state_digest;
- observed_problem declarations;
- proposed mechanism declarations;
- predicted outcome vector;
- predicted non-effects / protected invariants;
- uncertainty/confidence representation;
- evidence references available at decision time;
- selected counterfactual proposal digest.

After revisiting the changed world, record a `RevisionOutcome` with:

- trial_id;
- resulting_world_state_digest;
- measured outcome vector;
- protected-invariant results;
- side effects;
- prediction error by dimension;
- whether the declared mechanism remained consistent with available evidence.

Outcome construction must not rewrite the prior hypothesis.

## WC-002 — Multi-objective outcome vector

No overall `world_quality`, `beauty`, `cinematic_quality`, `intelligence`, or equivalent scalar is admissible as the primary scientific result.

The initial outcome vector should keep at least these channels separate:

### Physical validity

- reachable-area delta;
- support violations;
- obstacle/collision violations;
- path feasibility;
- route length or effort where preregistered;
- edit-induced physical regressions.

### Declared-goal consequence

Measures tied to the prospectively stated revision hypothesis only.

### Perceptual consequence

Examples, when justified by the qualified perception surface:

- landmark visibility;
- occlusion change;
- depth-structure change;
- viewpoint coverage;
- spatial diversity;
- composition descriptors.

### Side effects

Any preregistered protected property that changes despite being declared as a non-target.

### Counterfactual selection quality

Compare the selected candidate against the other candidates that were available at selection time.

### Human evaluation

Optional and separate. Human judgments must be blinded to agent identity, condition, and provenance wherever practical.

## WC-003 — CreativeTrial identity

Each trial must bind:

- committed source world version;
- presence/embodiment session;
- experience episode;
- all generated candidate world versions;
- selected candidate;
- applied revision;
- committed resulting world version;
- revisit observation;
- hypothesis and outcome records.

Counterfactual candidates that are not selected remain counterfactual forever and must not be relabeled as committed history.

## WC-004 — Baselines

Every confirmatory run must include, at minimum:

1. **Full Symthaea** — the complete qualified decision path under test.
2. **Random-valid** — choose randomly among the exact same physically admissible candidate set.
3. **Heuristic** — a preregistered deterministic policy using only declared measurements available to Full Symthaea.

Where practical, add:

- external model policy supplied the same evidence;
- human selection supplied the same candidate set.

Safety/authority gates remain identical for every policy.

## WC-005 — Ablations

The framework must support independent removal of at least:

- embodied experience;
- persistent memory;
- depth evidence;
- counterfactual evaluation;
- independent proposal replay;
- Reality Ledger provenance context;
- learned/native judgment, replaced by random-valid selection.

An ablation must change only its declared factor. It must not silently receive different candidate sets, easier worlds, or weaker physical gates.

## WC-006 — Adversarial world fixtures

Initial fixture classes:

### Pretty Trap

A visually attractive revision degrades the only safe route.

### Local Optimum

Repeated locally favorable edits degrade a global property.

### Hidden Dependency

A localized edit changes a distant physical or navigational property.

### Delayed Consequence

A revision appears favorable on immediate inspection but fails after extended traversal/revisit.

### Counterfactual Decoy

A candidate looks favorable from the evaluation viewpoint but poorly from ordinary embodied viewpoints.

### Memory Trap

A previously visited region changes; the agent must distinguish historical committed observations from the current committed state and from rejected counterfactuals.

Fixture generation and expected invariants must be frozen before confirmatory outcomes are inspected.

## WC-007 — Calibration ledger

For every predicted dimension, preserve:

- prediction;
- observed outcome;
- signed error;
- absolute error;
- uncertainty/confidence representation;
- calibration bucket when applicable.

Primary long-horizon question: does prospective prediction error improve across experience without degrading protected physical properties?

Calibration is measured independently from whether the edit happened to receive a favorable outcome.

## WC-008 — Long-horizon campaigns

Two distinct campaign families are required:

### Longitudinal coherence

Small number of worlds with many sequential revisions. Initial target: at least one 100-revision campaign after pilot qualification.

Tests:

- accumulated physical regressions;
- provenance corruption;
- memory/world-version confusion;
- aesthetic or policy drift;
- prediction calibration through time;
- rollback/revisit integrity;
- cumulative world-state reproducibility.

### Generalization

Many unfamiliar worlds with few revisions each.

Tests whether any measured advantage transfers beyond one authored fixture.

These campaign families must be reported separately.

## WC-009 — Blinded human evaluation

When human aesthetic/spatial preference is evaluated:

- remove agent/policy names;
- randomize before/after presentation order where compatible with the task;
- preserve a separate linkage key outside the rater surface;
- preregister exclusion criteria;
- report inter-rater agreement;
- never use human preference to override physical invalidity.

## WC-010 — Confirmatory plan

Confirmatory execution is unauthorized until the machine-readable plan has non-null values for:

- qualified parent receipts;
- exact source identities;
- fixture set and digests;
- seed set;
- policy set;
- ablation set;
- outcome definitions;
- calibration definitions;
- failure thresholds;
- protected invariants;
- stopping rule;
- evidence root;
- environment identity.

The plan must be sealed before confirmatory outcome inspection.

## WC-011 — Independent verifier

The evidence verifier must be logically independent of the runtime that produced the evidence.

At minimum it must reject:

- rewritten hypothesis after outcome observation;
- mismatched source/result world lineage;
- candidate substitution;
- proposal/decision/application digest mismatch;
- counterfactual relabeling as committed history;
- missing rejected-candidate evidence;
- baseline receiving a different candidate set;
- aggregate-score-only reporting;
- missing protected-invariant measurement;
- stale world/body state at selection or execution;
- evidence path reuse;
- partial campaign publication as complete.

## WC-012 — Reproduction capsule

A qualified campaign should seal enough material for an independent party to reconstruct the scientific claim surface:

- source identities;
- qualification prerequisites;
- experiment plan;
- environment capsule;
- fixture definitions and digests;
- raw observations where licensing/privacy permits;
- candidate sets;
- policy decisions;
- physical admission evidence;
- hypotheses;
- applied revisions;
- revisit observations;
- outcome vectors;
- calibration ledger;
- baseline and ablation results;
- verifier output;
- claim text and claim ceiling;
- full manifest and checksums.

## Initial confirmatory success criteria

The first study should **not** require Full Symthaea to dominate every baseline on every dimension.

A defensible first positive result would require all of the following:

1. zero authority/provenance violations;
2. zero unreported physical-safety violations;
3. selected revisions are measurably better than random-valid selection on at least one prospectively declared target family;
4. the advantage is not explained solely by receiving different candidate sets;
5. protected side effects remain within preregistered bounds;
6. prediction calibration is nontrivial and does not worsen catastrophically over repeated revisions;
7. the independent verifier reproduces every admitted trial/result relationship.

Failure is scientifically useful and must be preserved as such.

## Stop/go gates

### Gate A — Exact integration parent

Do not wire WC runtime code against an approximate or stale World Forge tree. The qualified v0.5-A source and parent receipts must be available first.

### Gate B — Pilot only

Before confirmatory execution, run a small noncanonical pilot to validate evidence plumbing, not hypotheses or thresholds.

### Gate C — Freeze

Freeze thresholds, fixtures, policies, and seeds after the pilot and before confirmatory outcomes.

### Gate D — Confirmatory execution

Run Full Symthaea, baselines, and ablations under the same admission/authority surface.

### Gate E — Independent closeout

Only independent verification may admit the final bounded claim.

## Proposed bounded claim

A successful first experiment may support a claim no stronger than:

`EvidenceBoundExperienceConditionedWorldImprovementQualified`

This means only that, under the exact preregistered fixtures, policies, measurements, environment, and qualified World Forge lineage, the tested experience-conditioned policy demonstrated the admitted bounded improvement behavior.

It does not imply general creativity, general intelligence, consciousness, universal aesthetic competence, or transfer to physical-world autonomy.

## Integration note

The v0.5-A handoff discussed on 2026-09-01 reports a frozen static source identity and a later live integration identity, but those commits are not currently resolvable from the connected `Luminous-Dynamics/symthaea` GitHub repository. Therefore this branch intentionally adds only host-neutral preregistration material. WC runtime patches must be rebased onto the exact qualified World Forge v0.5-A lineage once that lineage is pushed or otherwise supplied.