# Active Appliance Diagnostics Protocol v0.1

**Status:** synthetic design and evaluator contract only  
**Tracker:** [Symthaea APPLIANCE-DIAG-001](https://github.com/Luminous-Dynamics/symthaea/issues/7819)  
**Related system benchmark:** [Symtropy APPLIANCE-QUAL-001](https://github.com/Luminous-Dynamics/symtropy/issues/1805) and [substrate PR #1806](https://github.com/Luminous-Dynamics/symtropy/pull/1806)  
**Physical status:** no fan sensor data, field failures, diagnosis accuracy, safety qualification, or repair benefit is claimed

## Paradigm shift

A conventional diagnostic assistant ranks the fault it thinks is most likely. Active diagnosis instead asks:

> Which next measurement is most likely to distinguish the live hypotheses for the least cost, without breaching the declared safety/operator profile?

This changes the objective from maximizing static label accuracy to improving the quality and efficiency of the *diagnostic process*. Correct abstention, discovering out-of-scope faults, detecting sensor/data quality problems and avoiding false reassurance count as primary outcomes.

## Scope boundary

The reference device is a proposed guarded low-voltage DC fan supplied by an appropriate external certified power supply. All records in `fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json` are synthetic probability tables and contract cases. They are not derived from measured fans, reliability studies, repair-event frequencies, or calibrated likelihoods.

The first slice is deliberately a **math/policy evaluator**, not a trained Symthaea model, physical sensor adapter, or appliance controller. It computes whether the input contract is coherent and demonstrates which measurement would be selected under declared toy priors/likelihoods/costs. The input assumptions remain visible and should be replaceable by future versioned, instrument-supported estimates.

The low-voltage fan is not exempt from finished-product safety review. [IEC 60335-2-80:2024](https://webstore.iec.ch/en/publication/70375) covers household and similar electric fans, including DC-supplied and battery-operated examples. Applicable IEC 60335-1 requirements, national adoption, intended use, hazards and qualified test scope must be confirmed before any physical pilot or sale. Diagnostic policy fixtures cannot qualify the appliance.

## Decision policy v0.1

1. Validate the contract first: unique hypothesis/measurement/outcome identities; prior probabilities sum to one; each measurement gives a normalized outcome distribution for every hypothesis; costs are finite and positive. Never silently normalize bad input.
2. Protection state has precedence. A tripped protection yields `abstain_and_review/protection_tripped`; unknown protection state yields `abstain_and_review/protection_unknown`. No information-gain score can override either state.
3. Missing, stale, conflicting, or unit-invalid observations yield `abstain_and_review/observations_unreliable`. The contract refuses to rank fault hypotheses from unusable evidence.
4. Only available measurements marked `low` risk under the declared profile are eligible in v0.1. `qualified_person_required` and `prohibited` remain ineligible; v0.1 does not attempt to infer whether a person is qualified.
5. Compute prior entropy `H(H) = -Σ p(h) log₂ p(h)`. For a candidate test `m`, compute the predicted outcome probability, Bayesian posterior per outcome, expected posterior entropy, and information gain `IG(m) = H(H) - E[H(H|outcome,m)]`.
6. Rank eligible candidates using `IG(m) / cost_units(m)`. Cost is a within-fixture synthetic unit, not dollars, seconds, or a universal physical risk scale. Ties resolve lexicographically by stable measurement ID.
7. If no eligible measurement exists, return `no_safe_candidate`. If eligible tests have no material discrimination value under the supplied model, return `no_discriminating_measurement`. Do not invent a fallback or silently select a prohibited test.
8. This selection is an advisory test plan only. It never commands hardware or declares a device safe/unsafe. A physical operator follows separately approved procedure and any protection/lockout requirements.

## What this protocol tests

- Bayesian arithmetic and normalized probability contracts.
- Cost-sensitive selection rather than information gain in isolation.
- Stable tie-breaking and deterministic repeatability.
- Protection-state precedence and fail-closed handling of stale/missing/conflicting/unit-invalid input.
- No implicit repair of malformed prior or likelihood distributions.
- No use of unavailable, qualified-person-only, or prohibited tests as if they were eligible.
- Explicit `unknown_fault` probability mass; no forced closed-world classification.
- Trace/corpus digest binding for reports.

## Corpus census

The v0.1 synthetic corpus contains 21 scenarios: 4 selected-measurement examples, 7 abstain-and-review controls, 3 no-safe-candidate examples, 2 no-discriminating-measurement examples and 5 invalid-contract controls.

A case passing proves only that this evaluator reproduces the declared synthetic decision contract. It does not demonstrate accuracy on physical faults. These fixture probabilities are deliberately transparent and published; this corpus is not a hidden test set and must not be used to report generalization performance.

## Evaluation ladder

### A0 — contract integrity

Validate the JSON Schema; then enforce semantic checks that JSON Schema cannot express reliably here (probability sums, exact identity coverage, positive finite costs, safety precedence and expected decision matching). Emit SHA-256 commitments for exact schema/corpus bytes. Mutate priors, likelihoods, identities, costs, observation states and protection state; each mutation must fail or abstain as specified.

### A1 — independent synthetic evaluator comparison

Implement the equations in an independently written reference evaluator (for example a tiny separate test oracle) and compare action + next-test score/ordering on generated probability tables. Include asymmetric priors, zero-probability outcomes, equal scores, near-ties, unknown hypotheses, incomplete likelihoods, repeated IDs and floating-point tolerance boundaries. Do not use the same helper function as both oracle and implementation.

### A2 — calibrated model inputs

Only after named sensor hardware, calibration procedures and a physical sampling protocol exist, estimate fault priors/likelihoods from traceable observations. Bind them to exact device/component/procedure revisions, environment and dataset split. Preserve cohort, failure-mode, sensor-failure and missingness coverage. Priors must be scope-specific and not inferred from an event-selected community repair dataset as if it were a random field-failure sample.

### A3 — measurement-selection utility

Compare active policy against preregistered baselines:

- fixed checklist / most-common-next-test;
- one-shot classifier then technician-selected next test;
- random eligible test (paired seeds where appropriate);
- active information gain per cost;
- technician-selected next test in a consented protocol.

Report number of tests and time/cost to resolve a case, unresolved/abstained rate, diagnosis accuracy at stop, false reassurance, unsafe recommendation, calibration, and total restore time including setup/wait/rework. Evaluate by fault family and condition; one average must not conceal a critical-fault miss.

### A4 — physical, blinded evaluation

Use held-out physical articles/repair episodes with exact sensor calibration and independent outcome adjudication. Preregister task definitions, stopping conditions, randomization, exclusions, safety stop rules and analysis. Report sensitivity and false-negative rate for each critical fault class, uncertainty intervals, out-of-distribution behavior, abstention and adverse events. Do not inject dangerous faults into powered household devices; physical fault introduction needs a separate lab hazard review and appropriate qualified supervision.

## Proposed targets after valid data exist

These are not measured results and are not all applicable to synthetic-only A0/A1:

- 100% detection/rejection of malformed probability and identity contracts in the finite contract corpus.
- 100% of stale/missing/conflicting/invalid-unit states abstain from fault ranking in v0.1.
- 100% of tripped/unknown protection cases prevent next-measurement selection in v0.1.
- 0 prohibited or qualified-person-required measurements selected without an explicit later qualified authorization profile.
- Identical input bytes produce identical decision/output bytes under the pinned toolchain.
- For a future physical fault set, preregister minimum class-specific sensitivity and false-reassurance limits with the relevant safety reviewer; do not take a synthetic 90% top-3 target as a physical safety threshold.
- Active testing must improve a preregistered utility measure (e.g. tests-to-resolution or total diagnostic cost) over a frozen baseline without worsening critical-class misses or safety stops, using confidence intervals and adequate sample size.

## Integration boundary

- Symthaea owns the diagnostic evidence/hypothesis/decision analysis and future calibrated model.
- Symtropy can later supply explicitly synthetic counterfactual traces or measurement-cost trade-offs, but its outputs remain synthetic.
- Mycelix can later bind actual repair episodes, source parts, procedure versions and post-repair outcomes through existing repair/manufacturing provenance. A work-order close is not a diagnosis label or restored-capability proof.
- The appliance hardware's local controls/protection remain deterministic and independent. This crate has no GPIO, network command endpoint, actuator, or hardware authority.

Reuse existing engineering observation/evidence and repair/provenance contracts after a source crosswalk. Do not create a second repair ledger or duplicate the current manufacturing record model.

## Nonclaims

Nothing in v0.1 proves an appliance is safe, identifies a real root cause, predicts population failure frequencies, or validates a physical diagnostic model. No sensor hardware, empirical data, or technician study exists in this deliverable. Every output should prominently identify itself as synthetic contract evaluation until the physical evidence gates are independently completed.
