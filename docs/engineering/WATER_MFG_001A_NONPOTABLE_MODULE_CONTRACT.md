# WATER-MFG-001A — Non-potable circulation/filter module evidence contract

Issue: #5997  
Parent: WATER-MFG-000 #5966  
Scope: source/data only

## Purpose

Freeze a deterministic reference contract for a synthetic non-potable, low-head
circulation + filter-housing + flow-observation module before any WATER-MFG
production adapter is written.

This fixture is deliberately not a potable-water benchmark. It is an evidence
and authority-separation benchmark.

## Core separation

A source being present does not establish feedstream characterization. An
as-built pump/filter module does not establish hydraulic duty. A treatment
effect observation does not establish water quality or safety. One service
episode does not establish continuity. A locally supported enclosure or
manifold does not establish productive closure when key components remain
externally dependent.

## Canonical source

- schema: `water-mfg-001a-nonpotable-module-reference-v1`
- canonical JSON: `docs/release/evidence/water-mfg-001a-nonpotable-module-reference-v1.json`
- SHA-256: `75c74323afb879f0af4eb5917d74e620a01905beba7af50bbbb6301c757438c1`
- evidence planes: 15
- known-answer cases: 16

Canonical encoding is compact JSON with sorted keys and exactly one final
newline.

## Ownership

WATER-MFG owns only water-infrastructure-specific composition. The corpus binds
rather than duplicates:

- IND-COMP #5965 for component function;
- MFG-PROC #5686 for manufacturing-process semantics;
- FIELD/SE-OBS for physical observations/currentness;
- SENSE #5820 for sensing-system semantics;
- CIV-SERVICE #5949 for service projection;
- MFG-LIFE #5705 for lifecycle/repair;
- CIV-BOOT #5774 for productive/reproductive closure;
- Mycelix operations for operational facts.

## Evidence planes

The frozen corpus preserves source/feedstream identity, feedstream
characterization, component/configuration identity, as-built lineage,
hydraulic function, media identity/currentness, calibrated observations,
treatment-effect observation, external water-quality/safety evidence,
commissioning, capacity/continuity, renewal, common-mode dependencies,
residual/recovery routes, and productive closure as independent planes.

No one plane upgrades another.

## Known-answer intent

The corpus includes negative controls for uncharacterized feedstreams,
unmeasured pump duty, unknown filter media, visual-only output judgment,
external water-quality authority, missing commissioning, shared failure roots,
missing renewal, unqualified replacement, feedstream drift, partial local
closure, profile-bounded function, Mycelix↔engineering non-laundering,
model↔FIELD non-laundering, a fully represented synthetic route, and zero
physical-operation authority.

## Authority boundary

The reference contract cannot authorize:

- dosing or disinfection;
- public-water operation;
- pump/valve actuation;
- procurement or resource allocation;
- potable-water, pathogen-removal, sanitation or discharge-compliance claims.

A future independent validator PASS establishes only faithful representation of
this exact synthetic contract.
