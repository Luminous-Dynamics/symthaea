# ENG-DESIGN-CHANGE-QUAL-001 public blueprint registry

This directory contains the public v0.1 blueprint registry for engineering design-change impact and evidence applicability.

- manifest.json — 24 named task blueprints across six change families.
- manifest.schema.json — JSON Schema 2020-12 for the public blueprint manifest.

## Status

The registry is public authoring metadata only:
- all 24 cases remain blueprints, not materialized design/evidence graphs;
- all gold labels are not yet independently adjudicated;
- no candidate scores exist;
- no sealed holdout has been created;
- no untrusted execution, external MBSE/PLM evaluation, ETK action or physical testing is authorized.

The corpus is distinct from the repository/code impact gold set #10173. It composes ENG-DESIGN-005 #6117, Systems Engineering #3677 and ETK. It does not implement another semantic graph, delta model, dependency engine, evidence-currentness authority or release workflow.

## Static checks performed

The exact branch readback was parsed as JSON and checked for:
- 24 unique case IDs;
- six family IDs with four cases each;
- every case referenced exactly once by the family census;
- all cases gold_status=not_yet_adjudicated and score_status=not_scored;
- execution_authorized, untrusted_execution_authorized and physical_testing_authorized all false.

No JSON Schema 2020-12 validator, benchmark scorer, candidate agent, external product or physical record was executed.

See [issue #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543) for the full protocol and [ENG-DESIGN-CHANGE-QUAL-001 fixture bank PR #10546](https://github.com/Luminous-Dynamics/symthaea/pull/10546) for the public scenario definitions.
