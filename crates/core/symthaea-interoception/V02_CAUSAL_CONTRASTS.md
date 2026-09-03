# Affective Emergence v0.2 — Causal Contrast and Intervention Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents paired deterministic scenarios from being described as causal tests merely because many fields are numerically matched.

In a deterministic artificial system, we can often state the structural intervention more explicitly than in observational biological data. v0.2 should use that advantage.

## 1. Principle

Every mechanistic comparison should distinguish:

- what was deliberately manipulated;
- what was held fixed by construction;
- what is allowed to change as a mediator of that manipulation;
- what must remain unavailable to the candidate;
- what downstream candidate contrast is being tested.

Do not “control away” a mediator and then claim to have measured the total effect of the upstream manipulation.

## 2. Causal variable roles

A future causal-contrast manifest should classify variables/artifacts into roles such as:

### `ExogenousExperimentalInput`

Examples:

- declared drive schedule;
- explicit intervention schedule;
- synthetic fixture manipulation of precision/importance/channel geometry;
- scenario generator parameter chosen before execution.

These are controlled by the experiment, not inferred as agent actions.

### `NativeParameter`

Examples:

- preferred/viable ranges;
- precision;
- importance;
- dynamics configuration.

A scenario may intervene on a native parameter prospectively, but the contrast must say so explicitly.

### `NativeState`

Examples:

- current channel values;
- velocities;
- cycle index.

Native state is generally downstream of earlier drives/interventions/parameters and may act as a mediator.

### `DerivedNativeReport`

Examples:

- raw channel deviations;
- legacy weighted homeostatic deviation;
- allostatic forecast trajectory/report;
- breach timing/breadth.

These are deterministic functions of native state/configuration/forecast policy and should not be treated as independent experimental causes unless the experiment explicitly constructs synthetic reference fixtures for a mathematical test.

### `ObservatoryDerived`

Examples:

- CandidatePayload;
- candidate fingerprint;
- blinded comparison result.

These are read-only descendants of allowed prefix artifacts in v0.2.

### `SemanticPostHoc`

Examples:

- semantic arm labels;
- affective interpretation;
- unblinded hypothesis labels.

These must never be ancestors of primary candidate computation.

### `ForbiddenFeedback`

Any edge from observatory/candidate/semantic output back into:

- native drive;
- native intervention;
- native state update;
- neuromodulation;
- attention/memory/action-selection control.

Such an edge is outside v0.2 and invalidates the observational architecture.

## 3. Required DAG invariants

The design/evidence graph must respect at least:

`exogenous inputs / native parameters`
→ `native state trajectory`
→ `native reports / prefix-causal forecasts`
→ `candidate payload`
→ `blinded comparison`
→ `unblinding / semantic interpretation`.

Forbidden primary paths include:

- candidate payload → native state/drive/intervention;
- semantic arm identity → candidate payload;
- future schedule/suffix → prefix-causal candidate payload;
- post-run exclusion disposition → candidate payload;
- unblinded outcome → exclusion decision unless prospectively allowed by a mechanical criterion, which should normally be forbidden.

The trusted replay/evidence harness may hold broader provenance authority, but that authority must not become an input edge into candidate computation.

## 4. Prospective CausalContrastManifest

For every paired/factorial discriminator used for mechanistic interpretation, freeze a canonical manifest containing:

- contrast schema/version;
- stable contrast ID;
- exact scenario/cut-point identities for all arms;
- manipulated field(s);
- manipulation type: `Set`, `Add`, `ScheduleChange`, `ParameterChange`, or other prospectively typed operation;
- fields required equal across arms;
- fields intentionally not matched because they are expected mediators;
- allowed downstream path(s);
- forbidden changes/path(s);
- candidate/baseline pair(s) the contrast is intended to discriminate;
- whether the contrast estimates total-path, direct-path, mediator-specific, or purely algebraic/diagnostic behavior;
- expected structural invariances/differences when mathematically implied;
- nuisance fields to audit;
- canonical SHA-256.

A contrast cannot be defined after observing candidate outputs inside the same confirmatory lineage.

## 5. Total vs direct effects

Be explicit about the estimand-like question even though the system is deterministic.

### Total-path contrast

Example:

> What changes downstream when the external drive differs while initial state/configuration are held fixed?

Do **not** subsequently require current native state to remain equal if current state is a mediator of the drive effect.

### Direct-path diagnostic

Example:

> Holding current native state equal by construction, does a forecast policy respond differently to two already-observed drive histories?

This is a different question. The matched current state is intentional and should be described as a direct/history-sensitive diagnostic rather than the total effect of drive.

### Parameter intervention

Example:

> Holding state/geometry fixed, change only precision.

This isolates the mathematical role of precision in a candidate; it does not imply the agent experienced an endogenous confidence update unless a later model provides such dynamics.

## 6. Mediator discipline

Potential mediators include:

- native state after an intervention;
- state velocity;
- homeostatic deviation;
- forecast trajectory;
- breach timing;
- recovery trajectory.

If a contrast is intended to measure the effect of an upstream manipulation through regulation, these variables should generally be allowed to change.

If they are forcibly matched, the contrast instead measures a different direct/conditional question and must be labeled accordingly.

Do not use nuisance matching to erase the mechanism under test.

## 7. Post-treatment / outcome-dependent selection firewall

Exclusion and scenario selection can themselves create a distorted causal comparison if they depend on downstream candidate outcomes.

Therefore:

- exclusion criteria are prospectively mechanical;
- candidate/hypothesis outcomes cannot determine scenario inclusion;
- unblinded semantic labels cannot determine inclusion;
- a surprising mediator trajectory is not excludable merely because it breaks an expected story;
- missing/invalid execution evidence may exclude under locked rules, but the evidence remains preserved.

This complements the exclusion-evidence registry design in issue #263 / PR #267.

## 8. Required causal contrast families

The exploratory program should include at least:

### C1 — drive intervention

Same validated initial state/configuration, different declared drive schedule.

Use for total-path response through native state/regulation.

### C2 — intervention pulse

Same initial conditions, explicit intervention differs at one locked step.

Use for causal sensitivity/recovery-path tests.

### C3 — precision-only parameter contrast

Same values/geometry/importance/history, precision differs.

Use to identify epistemic/legacy weighting effects without claiming endogenous precision dynamics.

### C4 — importance-only parameter contrast

Same values/geometry/precision/history, importance differs.

Use to identify normative weighting effects.

### C5 — healthy-denominator contrast

Same deviated channel and its own weight, manipulate only an unrelated healthy channel's weight/precision.

Use to identify denominator dilution in scalar means.

### C6 — history-at-matched-state contrast

Construct matched current state/burden with different already-observed histories/velocities.

Use to test history/forecast sensitivity while explicitly acknowledging that the contrast is conditional on matched current state, not the total effect of history.

### C7 — forecast-policy contrast

Same immutable prefix, different prospectively locked forecast policy.

Use to attribute candidate differences to forecast assumptions rather than native execution.

### C8 — future-suffix diagnostic

Same prefix, mutate unseen future only.

Expected candidate payload invariance. This is an information-flow diagnostic, not a causal claim about future events.

## 9. Causal graph vs candidate factorization

Candidate coordinates and causal contrasts answer different questions.

- candidate coordinate: **what measurement is computed?**
- causal contrast: **what experimental change makes candidate/baseline responses informative?**

The `CandidateDiscriminationManifest` should reference the causal-contrast IDs that satisfy each pairwise discrimination obligation where the discriminator is mechanistic rather than purely algebraic.

This prevents a candidate from being called mechanistically distinctive based only on uncontrolled scenario differences.

## 10. Structural equation transparency

Where practical, v0.2 design/review should document deterministic structural relationships explicitly, for example:

- native transition law consumes prior state + declared drive + dynamics config;
- homeostatic report consumes current native state;
- forecast trajectory consumes allowed prefix state/history + locked forecast policy;
- candidate consumes prefix/trajectory artifacts under its coordinate;
- semantic evaluation consumes frozen blinded artifacts only after unblinding.

The point is not to claim that this simple DAG exhausts cognition. It is to make the experimental subsystem's causal dependencies auditable.

## 11. Manipulation-check evidence

Every causal contrast should produce a blinded manipulation-check artifact proving:

- intended manipulated fields actually differ as declared;
- required-equal pre-treatment fields match;
- forbidden fields did not change;
- mediator fields are reported rather than silently forced equal unless the contrast explicitly requires matching;
- the contrast references the correct source trace/prefix identities.

A failed manipulation check is an integrity/design failure for that contrast, not evidence against the scientific hypothesis.

## 12. Confirmatory causal-claim gate

A confirmatory report may use language like “responds to manipulation X beyond baseline Y” only when:

- X is defined by a locked causal contrast;
- the manipulation check passes;
- the primary/baseline pair is identifiable under the locked discrimination manifest;
- nuisance matching does not condition away a declared mediator for the claimed total-path effect;
- candidate computation remains prefix-causal/read-only;
- no outcome-dependent scenario selection occurred.

Otherwise use descriptive language such as “differs across these scenarios” rather than causal language.

## 13. Evidence-root consequences

The prospective root should bind:

- causal-contrast contract/version;
- ordered causal-contrast manifest digests required by primary hypotheses;
- mapping from pairwise discrimination obligations to contrast IDs;
- manipulation-check contract/version.

The realized package should bind:

- manipulation-check artifact digests;
- causal-contrast execution/accounting report;
- any contrast invalidation/indeterminate status.

## 14. Claim boundary

A causal-contrast contract can make mechanistic comparisons in the deterministic subsystem more precise and can prevent inappropriate conditioning or semantic overstatement.

It does not establish that any downstream regulatory candidate is emotion, subjective valence, feeling, mood, suffering, sentience, or consciousness.