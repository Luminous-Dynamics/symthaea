# SEMI-EQP-MET-001E2 — Exact Bench Physical / As-Built / Configuration Subject Manifest

Parent: SEMI-EQP-MET-001 #5912
Campaign parent: SEMI-EQP-MET-001E #5934
Issue: #5941

## Purpose

Freeze the campaign-facing manifest that identifies the exact physical inspection-bench subject and exact as-built/acquisition/calibration context a future claim-bearing run refers to.

This tranche is documentation/data only. It selects no vendor or hardware and introduces no control surface.

## Ownership

This profile composes existing owners:

- ROB-REALIZE #4859 owns physical article identity, as-built generations, installed substitutions, repair/rework lineage, and the design->physical distinction.
- SEMI-EQP #5889/#5910/#5911 owns semiconductor-tool role composition and equipment capability/configuration evidence.
- SE-OBS #3695 owns physical observations and raw/derived observation provenance.
- EXEC-ID calibration #4620 owns calibration identity, epoch, qualification and currentness.
- EXEC-ID commissioning #4623 owns commissioned physical-subject relations where applicable.
- SENSE provenance/custody #5858/#5859 owns replay/derived anti-self-evidence constraints.
- SEMI-EQP-MET-001E2 binds canonical refs for this inspection-bench campaign profile only.

No new universal `BenchId`, `PhysicalArticleId`, `AsBuiltConfigurationId`, device-attestation system, calibration ontology, commissioning authority, or observation store is created.

## Core theorem

A design or BOM is not a physical bench article.

A friendly bench/camera/stage name is not physical instance identity.

The same design instantiated twice yields two physical subjects.

Replacing an installed subsystem creates a new as-built/configuration generation where the change is material.

The same physical hardware under a changed acquisition parser/firmware/profile may remain the same physical article while becoming a different acquisition context.

A commissioning record does not mint execution authority.

A complete subject manifest does not establish physical capability.

## Manifest roles

A future campaign subject manifest should reference, rather than duplicate:

1. physical article / exact equipment subject;
2. exact as-built configuration generation;
3. installed subsystem instance refs by semantic role;
4. optics/mount/fixture configuration refs where material;
5. acquisition firmware/driver/parser/profile refs;
6. calibration identity, epoch and currentness;
7. commissioning relation ref where the profile requires one;
8. campaign constitution ref;
9. exact reference-target binding;
10. custody/session profile refs;
11. local/import dependency refs;
12. explicit execution authority = none.

## Identity decomposition

Do not collapse all concerns into one opaque ID.

Preserve separately:

- design identity;
- physical article identity;
- as-built configuration generation;
- installed subsystem identities;
- acquisition profile identity;
- calibration identity/epoch/currentness;
- commissioning evidence/relation;
- campaign constitution;
- reference-target identity/currentness;
- run/session/custody identity.

A campaign evidence capsule may commit all of these together, but they remain independently attributable.

## Configuration-generation law

A materially changed installed configuration creates a new context/generation rather than rewriting the old one.

Examples include, where material to the evidence profile:

- imager replacement;
- stage/reference-sensor replacement;
- changed optical path;
- changed mount/fixture;
- remount;
- acquisition firmware/parser change;
- calibration epoch transition requiring a changed measurement context.

Historical observations remain attached to the configuration under which they were acquired.

## Provenance strength

Friendly names, filenames, product-family labels, catalog entries and intended BOM positions do not establish exact installed physical identity.

Weak identity may remain useful for exploratory development. It cannot silently satisfy a stronger claim-bearing profile.

## Frozen synthetic corpus

Path:

`docs/release/evidence/semi-eqp-met-001e2-bench-subject-corpus-v1.json`

Canonical SHA-256:

`496f5686df93d5995c82f982f9c164d7a2d775b2657395eca62235c9271ba384`

The corpus contains exactly 16 benign synthetic cases:

1. exact physical/as-built/current-calibration/campaign/target binding -> manifest complete;
2. design without physical article -> physical subject missing;
3. friendly names only -> weak identity insufficient;
4. one design instantiated as two articles -> distinct physical subjects;
5. camera replacement -> new as-built generation;
6. changed optics -> new configuration context;
7. parser/firmware change -> same article, changed acquisition context;
8. stale calibration -> campaign subject not current;
9. manifest/campaign target mismatch -> reject;
10. explicitly unordered subsystem refs canonicalize;
11. duplicate subsystem ref -> reject;
12. imported critical subsystems -> operational but import-dependent;
13. missing campaign/custody binding -> incomplete;
14. remount without calibration-transfer evidence -> transfer required;
15. exploratory prototype manifest -> exploratory only;
16. manifest -> zero physical execution authority.

## Qualification rules

A future independent validator must:

1. hard-bind the exact corpus digest, schema, authority, count and fixture identities;
2. derive results from relationships rather than echoing `expected_*` values;
3. preserve design-vs-article and article-vs-configuration distinctions;
4. preserve physical-article continuity separately from acquisition-context changes;
5. reject weak friendly-name identity as a strong physical join;
6. canonicalize only fields explicitly declared unordered;
7. reject duplicate subsystem refs;
8. preserve import-dependent productive-closure state independently of operational status;
9. require explicit calibration-transfer evidence after relevant remount;
10. mint no physical execution authority.

## Promotion gate

Selecting/fixing the actual claim-bearing bench subject remains blocked until:

- A/B/C/D/E independent reference workflows pass on their exact heads;
- this 001E2 corpus has an independent validator;
- canonical ROB-REALIZE/SEMI-EQP/SE-OBS/calibration surfaces are sufficiently stable to reference directly;
- actual installed-subsystem identities, as-built generation and reference-target identity/currentness are recorded separately from design intent.

Exploratory hardware may be assembled earlier but remains exploratory.

## Prohibited content

No vendor/BOM selection, dimensions, travel ranges, speeds, forces, focal lengths, optical powers, electrical operating values, controller commands, semiconductor process parameters, hazardous materials/process instructions, or autonomous execution.

## Claim ceiling

A software PASS establishes only faithful physical/as-built/configuration subject-binding semantics.

It does not establish:

- commissioned hardware;
- instrument accuracy or repeatability;
- traceable calibration;
- valid reference-target metrology;
- wafer inspection performance;
- semiconductor process capability;
- safety admission;
- productive closure;
- physical execution authority.
