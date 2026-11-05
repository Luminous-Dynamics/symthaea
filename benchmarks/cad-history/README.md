# CAD-HISTORY-001 public fixture registry

This directory contains the public v0.1 blueprint catalog for edit-preserving parametric history reconstruction and metrology-aware scan-to-CAD.

- manifest.json — 11 named cases across parametric edit, scan-metrology and adversarial/ambiguity families.
- manifest.schema.json — JSON Schema 2020-12 for the public blueprint catalog.

## Status

All cases remain blueprint-only, gold labels not yet adjudicated and unscored. No native CAD files or point clouds have been generated; no scanner/calibration artefact or commercial scan-to-CAD product was run; no external CAD V3 task data was downloaded. Execution and physical testing remain unauthorized.

The V3 comparator fields in the manifest pin the public run contract (Harbor 0.23.0, FreeCAD 1.1.0, validator 0.6.0, schema 1.3, one attempt, four-way concurrency, zero retries, 9000 seconds). No official dataset digest has been captured and redistribution/training terms are not verified.

Compose FAB-GEOM #4927, ENG-TOL #4883, ROB-REALIZE #4859 and ENG-DESIGN-005 #6117. No new geometry identity, scanner measurement authority, as-built lineage, CAD kernel, manufacturing approval or physical authority is created.

## Static checks performed

The exact branch readback was parsed as JSON and checked for:
- 11 unique case IDs across 3 families (6 parametric-edit, 1 scan-metrology, 4 ambiguity/adversarial);
- family membership covers every case exactly once;
- every case remains not_yet_adjudicated, not_scored and blueprint_only;
- no CAD files, point clouds or external data were materialized/downloaded;
- no candidate execution, commercial scan-to-CAD run or physical testing is authorized.

No JSON Schema 2020-12 engine, CAD kernel, scanner, external V3 task or model was run.

See [CAD-HISTORY-001 #10510](https://github.com/Luminous-Dynamics/symthaea/issues/10510), its [fixture specification PR #10529](https://github.com/Luminous-Dynamics/symthaea/pull/10529), and the [external benchmark registry PR #10458](https://github.com/Luminous-Dynamics/symthaea/pull/10458).
