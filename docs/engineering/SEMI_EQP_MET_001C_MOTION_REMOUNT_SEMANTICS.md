# SEMI-EQP-MET-001C — Motion Discrepancy, Revisit, Remount, and Calibration-Discontinuity Semantics

Parent: SEMI-EQP-MET-001 #5912  
Issue: #5922  
Semantic parent: SEMI-EQP-MET-001B #5918 / draft PR #5919

## Purpose

Freeze the inspection-bench motion/remount evidence semantics before any production motion adapter or claim-bearing physical stage campaign.

This tranche is documentation/data only. It introduces no hardware-control path, vendor selection, motion parameters, construction instructions, semiconductor process parameters, or physical execution authority.

## Core theorem

A motion command is not a physical position observation.

Controller/readback agreement is not automatically an independent physical reference.

Repeatable revisit is not absolute accuracy.

A correction does not erase the original discrepancy.

A remount does not silently preserve the old transform or calibration context.

A fresh calibration does not rewrite historical observations.

## Canonical ownership

This profile composes existing owners:

- SE-OBS #3695 — physical observations, raw/derived distinction, source/receive timestamps, quantity/frame/calibration refs;
- EXEC-ID #4620 — canonical calibration identity, epoch, and currentness semantics;
- ROB-INSTR #4866 — commanded/latched/observed separation and evidence-grade timing discipline;
- ROB-REALIZE #4859 / SE configuration owners — physical/as-built/remount configuration generations;
- SE-VV #3697 — discrepancy/anomaly lifecycle semantics;
- SEMI-MET #5891 — semiconductor metrology specialization;
- SEMI-QUAL #5892 — later cross-layer qualification composition.

SEMI-EQP-MET-001C does not create another coordinate-frame ontology, calibration engine, generic discrepancy object, motion planner/controller, observation store, or execution-authority system.

## Evidence planes

Keep independently attributable:

1. commanded target reference;
2. controller/encoder/readback reference where available;
3. independent physical/reference observation where available;
4. exact coordinate/reference-frame identity;
5. source and receive timing context;
6. calibration/transform identity and epoch;
7. residual/discrepancy artifact;
8. revisit/repeatability campaign identity;
9. absolute-reference evidence where applicable;
10. remount/reconfiguration generation;
11. corrected/registered derived artifact identity.

No plane silently establishes another.

## Revisit semantics

Repeated visits under one exact configuration may support candidate repeatability evidence.

Without an independent absolute reference and suitable calibration/currentness, the strongest allowed result remains repeatability-only.

A small cluster around the wrong location is still repeatable and inaccurate.

## Remount semantics

A mechanically relevant remount or configuration change creates a new configuration context.

Prior raw observations remain historical and retain their original calibration/transform context.

Transfer of a previous calibration after remount is a proposition requiring explicit evidence. It is never inferred from friendly names, nominally identical hardware, or visually similar images.

## Calibration-discontinuity semantics

Calibration identity and currentness affect which measurement claims can be made from an observation. They do not mutate the original raw observation.

A new calibration epoch creates a new measurement-admission context.

Historical evidence remains bound to the calibration/currentness that applied when it was used.

## Correction semantics

Registration, coordinate correction, image alignment, or any equivalent transformation is derived evidence.

The lineage must retain:

- original raw observation;
- original discrepancy where present;
- correction/transform identity;
- corrected/derived artifact.

A successful correction does not retroactively make the original command or observation exact.

## Timing semantics

Where source and receive clocks/timestamps exist, preserve them separately.

Same sample index is not simultaneity.

Host receipt time is not automatically physical event time.

This tranche defines no timing-correction algorithm.

## Coverage semantics

A local reference feature establishes only the scope actually observed.

Local position evidence cannot silently become whole-sample positioning accuracy or field-wide calibration.

## Frozen synthetic corpus

Path:

`docs/release/evidence/semi-eqp-met-001c-motion-corpus-v1.json`

Canonical SHA-256:

`915225ff10857ce51e0d7a9e2dbd8c591d62d360e621b15a2f6d43a078b5686f`

The corpus contains exactly 16 benign synthetic cases:

1. command without physical/reference observation;
2. command and controller readback agree without independent reference;
3. command differs from independent observed position;
4. clustered revisits under one exact configuration;
5. repeatability without absolute reference;
6. current absolute reference + current calibration;
7. stale calibration blocking stronger absolute-position claim;
8. remount changing transform identity;
9. fresh post-remount calibration creating a new epoch while preserving history;
10. correction producing a derived coordinate while retaining original discrepancy;
11. same friendly stage name with different physical/configuration identity;
12. source and receive timestamps/clocks preserved separately;
13. replayed log producing no fresh physical repeatability evidence;
14. local reference supporting only a local/profile-relative claim;
15. multiple corrected derivations from one physical observation remaining one physical witness;
16. motion/calibration evidence minting zero physical execution authority.

## Required future independent validator

A future stdlib-only reference oracle must:

- hard-bind the exact corpus digest;
- reject schema/authority/count/case-ID drift;
- derive outcomes rather than echoing `expected_*` fields;
- preserve command/readback/reference distinctions;
- preserve repeatability-vs-accuracy distinction;
- preserve remount/configuration/calibration discontinuity;
- preserve original discrepancy after correction;
- preserve source/receive timing distinction;
- classify replay as no fresh physical observation;
- preserve one physical witness across multiple derivations;
- mint no physical execution authority.

## Production gate

Production Rust and claim-bearing physical motion qualification remain blocked until:

1. #5917 and #5921 dedicated reference workflows pass on their exact heads;
2. canonical observation/calibration/configuration public surfaces are sufficiently stable;
3. a production specialization can differentially reproduce the frozen A/B/C corpora without duplicating canonical owners;
4. the physical reference-target campaign is separately preregistered.

## Prohibited content

This tranche contains no:

- hardware dimensions or tolerances;
- travel ranges;
- speeds or accelerations;
- forces or loads;
- currents, voltages, control gains, or actuator commands;
- vendor BOM;
- semiconductor fabrication/process parameters;
- hazardous materials/process instructions;
- autonomous physical execution path.

## Claim ceiling

Even a complete future software PASS establishes only faithful reproduction of the frozen synthetic motion/remount semantics.

It does not establish real stage accuracy or repeatability, encoder correctness, traceable calibration, optical metrology performance, commissioned hardware, semiconductor inspection capability, semiconductor process capability, yield, fabrication capability, or physical execution authority.
