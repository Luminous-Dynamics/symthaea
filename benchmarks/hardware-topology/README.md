# HARDWARE-TOPOLOGY-001 public fixture registry

This directory contains the public v0.1 blueprint registry for mechanical assembly pose/mating, ECAD pin-to-net reconstruction, and manufacturing-feasibility checks.

- manifest.json — 20 named blueprint cases across three families.
- manifest.schema.json — JSON Schema 2020-12 for the public blueprint registry.

## Status

The registry is public authoring metadata only:
- all 20 cases are blueprint_only;
- gold labels are not yet independently adjudicated;
- no candidate scores exist;
- no sealed holdout has been created;
- no BenchCAD public/Core case, drawing, model, reference artifact or canary has been downloaded;
- no CAD/ECAD, KiCad, FreeCAD, model, benchmark or physical run is authorized.

Compose existing owners: BUILT-ROB-PILOT #6040, ROB-CATALOG #4877, ROB-REALIZE #4859, ENG-TOL #4883, COMPUTE-MFG-001C #8223, ENG-EMC #6179, ENG-SPICE #5633/#5661, FAB-GEOM #4927 and MFG-VALUE #7608. This directory adds no new geometry, BOM, netlist, manufacturing or physical authority.

## Static checks

The exact branch readback is expected to parse as JSON and satisfy:
- 20 unique case IDs;
- 3 task families (8 mechanical assembly, 6 ECAD, 6 manufacturing feasibility);
- family membership accounts for every case exactly once;
- every case is not_yet_adjudicated, not_scored and blueprint_only;
- external access and physical-execution gates remain false.

No JSON Schema engine, CAD/ECAD parser, benchmark runner, data download, machine, inspection or physical test has been executed.

See [HARDWARE-TOPOLOGY-001 #10530](https://github.com/Luminous-Dynamics/symthaea/issues/10530) and the [fixture specification PR #10532](https://github.com/Luminous-Dynamics/symthaea/pull/10532).
