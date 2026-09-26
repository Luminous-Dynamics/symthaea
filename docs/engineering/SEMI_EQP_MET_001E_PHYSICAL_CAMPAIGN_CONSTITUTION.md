# SEMI-EQP-MET-001E — Preregistered Benign Physical Reference-Target Campaign Constitution

Parent: SEMI-EQP-MET-001 #5912
Issue: #5934

## Purpose

Freeze the first claim-bearing physical-campaign constitution for the bounded semiconductor inspection/positioning/calibration bench before any hardware selection, device control, or physical acquisition.

This tranche is representation/data only. It does not select vendors or components, drive motion, energize optics, acquire a real image, or inspect a semiconductor article.

## Core theorem

exploratory observation != preregistered qualification evidence

reference artifact present != reference artifact current

current calibration != current metrological chain when the reference artifact is stale

partial or aborted run != completed qualified run

later success != deletion of earlier failure

same physical target across sessions != independent target replication

software replay != fresh physical acquisition

post-reveal acceptance change != preregistered confirmation

## Ownership

This campaign composes existing owners:

- SE-OBS #3695 — physical observations, raw/derived lineage, timestamps, frames, calibration refs;
- EXEC-ID #4620 — calibration identity, epoch, currentness;
- ROB-REALIZE / SE configuration — exact as-built/configuration and remount generations;
- SEMI-MET #5891 — semiconductor metrology specialization;
- SEMI-QUAL #5892 — machine/process/wafer qualification composition;
- SENSE #5858/#5859 — evidence custody and anti-self-evidence;
- A/B/C/D SEMI-EQP-MET reference layers — benchmark, lineage, motion/remount, optical metrology.

No new observation store, calibration ontology, hardware-control API, motion planner, image pipeline, or execution-authority path is introduced here.

## Campaign classes

### Exploratory

Used to learn, debug, or characterize the developing bench.

Exploratory evidence may motivate:
- a new benchmark profile;
- a new calibration procedure;
- a changed configuration;
- a future preregistered campaign.

It may not be retroactively reclassified as preregistered confirmatory evidence.

### Claim-bearing candidate

A future run can only enter this class when the campaign constitution is frozen before acquisition and all required identities/currentness/held-out conditions are satisfied.

Even then it is only a candidate run until its exact evidence is evaluated under the frozen profile.

## Required bindings

A claim-bearing candidate run must bind at least:

1. exact campaign constitution/version;
2. exact instrument/as-built/configuration subject;
3. exact reference target/article identity;
4. target/reference currentness and custody;
5. exact calibration identity/epoch/currentness;
6. exact session identity;
7. task/region/coverage profile;
8. held-out region/feature identity where applicable;
9. raw physical observation root(s);
10. derived-artifact lineage;
11. environment/context refs material to applicability;
12. preregistered acceptance axes/tolerances;
13. run state;
14. abort/failure evidence where present;
15. explicit zero physical execution authority.

## Held-out discipline

Held-out regions/features/targets intended for confirmatory evaluation must not be used to tune:
- optics;
- alignment;
- motion;
- calibration;
- correction/registration transforms;
- filters;
- thresholds;
- acceptance criteria.

Leakage invalidates the confirmatory interpretation for that campaign lineage.

## Remount and recalibration

A materially relevant remount/reconfiguration creates a new evidence context.

Prior calibration transfer is unsupported unless explicit evidence supports it.

Fresh calibration creates a new calibration context while preserving:
- pre-remount observations;
- prior calibration identities;
- stale/failed calibration history;
- discrepancies and aborts.

## Abort/failure retention

Aborted, partial, failed, invalid, or incomplete runs remain evidence.

A later successful run does not overwrite them.

## Repetition and replication

Keep distinct:

- repeated acquisition within one session;
- repeated session on the same physical target;
- remounted repeated session;
- new physical target/article;
- software replay.

Only the exact declared relationship may be claimed.

## Frozen synthetic corpus

Path:

`docs/release/evidence/semi-eqp-met-001e-physical-campaign-corpus-v1.json`

Canonical SHA-256:

`b5bc685d53fef2afd6435f2ef53ff5c4348f9a56de822f01f7d03ec35cb13341`

The corpus contains exactly 16 benign synthetic cases:

1. preregistered/current candidate claim-bearing run;
2. exploratory run cannot be retroactively promoted;
3. held-out leakage;
4. target identity mismatch;
5. stale reference artifact;
6. stale calibration;
7. remount without calibration-transfer evidence;
8. remount with fresh calibration and preserved history;
9. aborted run retained, not passed;
10. incomplete coverage;
11. repeated session on same target is not independent-target replication;
12. software replay is not a fresh physical run;
13. derived outputs do not multiply physical witnesses;
14. missing session/custody evidence;
15. post-reveal acceptance-criteria change;
16. zero physical execution authority.

## Qualification rules

An independent reference validator must:

1. hard-bind the exact corpus SHA-256;
2. reject schema/authority/count/fixture-ID drift;
3. derive outcomes from fixture relationships rather than echo `expected_*`;
4. preserve exploratory-vs-claim-bearing separation;
5. reject held-out leakage;
6. preserve target/calibration/currentness identity;
7. preserve remount/recalibration history;
8. retain abort/failure evidence;
9. preserve repeat-session vs independent-replication semantics;
10. preserve raw/derived witness counts;
11. reject post-reveal criteria changes;
12. mint no physical execution authority.

## Physical campaign gate

No claim-bearing physical campaign may begin until:

- #5917, #5921, #5925 and #5929 dedicated reference workflows PASS on exact heads;
- an independent 001E validator PASSes on its exact head;
- canonical observation/calibration/configuration surfaces are stable enough to consume directly;
- exact physical reference artifacts and their custody/currentness are identified;
- the actual hardware/configuration subject is frozen separately;
- acceptance criteria are frozen before confirmatory acquisition.

Exploratory prototyping may occur earlier but remains exploratory.

## Prohibited content

This V1 contract contains no:

- vendor/BOM selection;
- hardware dimensions;
- travel ranges, speeds, accelerations, forces or control gains;
- focal lengths or optical powers;
- voltages, currents, frequencies or device commands;
- semiconductor process parameters;
- hazardous materials/process instructions;
- autonomous physical authority.

## Claim ceiling

Even a complete software PASS establishes only faithful reproduction of the frozen synthetic campaign semantics.

It does not establish:

- real instrument accuracy/repeatability;
- traceable calibration;
- valid physical reference targets;
- commissioned hardware;
- wafer inspection performance;
- semiconductor process capability;
- fabrication capability;
- yield;
- productive closure;
- physical execution authority.
