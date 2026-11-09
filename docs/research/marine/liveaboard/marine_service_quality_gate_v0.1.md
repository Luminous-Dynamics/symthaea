# Marine service-quality qualification gate v0.1

**Status:** dependency-free software contract and synthetic conformance tests. No physical water-quality, electrical-quality, vessel-safety or certification claim is made.

## Purpose

The marine service-continuity evaluator measures delivery against a declared quantity floor. Quantity is necessary but not sufficient to establish usable service.

This adapter evaluates source-owned quantity, quality, evidence-currentness and authorization results against an exact mission service contract. It does not create water chemistry, microbiology, electrical power-quality or calibration physics.

## Ownership

- Watermaker and accepted product-water boundary: Symthaea #10994.
- Electrical power-transient and protection evidence: Symthaea #11028.
- Instrument identity, raw records, clocks, calibration and uncertainty: Symthaea #11064, composing ROB-INSTR #4866, ROB-EXP #4800, SE-OBS #3695, calibration #4620 and quantity semantics #4861.
- Mission/configuration identity: #10988.
- Maintenance and accepted restoration: #11049 and Symtropy #4056.
- Marine service-continuity profile: #11115 and evaluator PR #11126.

This module is only a thin contract evaluator. Domain owners determine applicable criteria, measurement methods, validity, uncertainty and evidence authenticity.

## Contract

A ServiceQualityContract freezes:

- service identity;
- approved quality-profile digest;
- exact configuration digest;
- quantity unit and minimum delivery rate;
- explicit owner-declared quality predicate inventory;
- whether measured evidence requires current calibration.

A QualifiedServiceObservation binds:

- service/profile/configuration identity;
- quantity and exact unit;
- observation/admission time and half-open validity interval [valid_from, valid_until);
- evidence class: synthetic, assumed, model estimate, external source or measured;
- source reference and raw content digest;
- instrument and calibration identity/currentness where required;
- authorization state;
- owner-supplied predicate results, each bound to the quality profile and its own evidence reference/content digest.

The evaluator checks digest format only. It does not fetch source records, authenticate a signer, validate laboratory results, recalculate instrument uncertainty or establish calibration validity. Those remain owner-verification responsibilities.

## Result semantics

- ContractQualified: every declared contract predicate and applicable evidence/authority gate passes for the decision time and exact configuration.
- Unavailable: known quantity-floor or quality predicate failure.
- Unknown: an observation is not yet available, a required predicate is missing/unknown, or authorization is unknown.
- InvalidEvidence: wrong service/profile/configuration/unit, stale validity, invalid predicate evidence, missing required calibration or expired calibration.
- UnsafeNotAuthorized: an applicable authority gate explicitly denies release.

A report retains its evidence class. A synthetic contract-qualified result remains synthetic and must never be described as measured physical qualification.

## Frozen conformance cases

1. Adequate quantity + every required quality predicate passes + current measured calibration + authorized release -> ContractQualified.
2. Adequate quantity + water-quality failure -> Unavailable.
3. Adequate quantity + missing microbiology result -> Unknown.
4. Adequate quantity + expired calibration -> InvalidEvidence.
5. Wrong configuration or quality-profile revision -> InvalidEvidence.
6. Late-arriving quality observation -> excluded from earlier decision-time qualification.
7. Unknown authorization -> Unknown.
8. Denied authorization -> UnsafeNotAuthorized.
9. Synthetic observation where calibrated measured evidence is required -> InvalidEvidence, preserving Synthetic evidence class.
10. Observation at valid_until is outside the half-open validity interval.
11. Low quantity with all quality checks passing -> Unavailable.
12. Unexpected or duplicate quality predicates -> InvalidEvidence or structural rejection.
13. Unit mismatch is invalid; no implicit conversion or silent assumption.

## Benchmark targets

- 100% exact result agreement with an independently authored table-driven oracle on the frozen corpus.
- Zero rate-only false positives.
- Zero missing, wrong-profile, wrong-configuration, stale or late quality evidence accepted.
- Zero authorization denial overridden by quantity or quality surplus.
- 100% of supplied required predicate results accounted for.
- 100% deterministic output for identical typed inputs/toolchain.
- Synthetic, model-estimate, external and measured evidence remain visibly distinct.
- Report predicate-level reason and evidence class; no single weighted quality score.

## Industry comparison

Do not claim to invent water-quality analysis, RO monitoring, electrical power-quality analysis or condition monitoring. These are existing equipment/vendor/measurement capabilities. A future comparison must use the same permissioned episodes, configuration, source coverage, calibration policy, acceptance criteria, sampling regime and observation horizon. An unexposed field is Unknown—not presumed pass or fail.

## Claim boundary

A passing software test proves only that this adapter follows the declared typed contract on its inputs. It does not prove potability, electrical safety, instrument authenticity, calibration correctness, vessel seaworthiness, flag/class compliance or authorization to operate.
