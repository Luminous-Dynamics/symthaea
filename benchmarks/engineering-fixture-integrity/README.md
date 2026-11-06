# Engineering fixture integrity extensions

Companion source for the optional OpenSCAD / CGAL audit in the synthetic
engineering-fixtures-v0.4 package.

## Run

Install the package's pinned Python mesh-audit requirements and OpenSCAD
2021.01. From the repository root, run:

```bash
python -m unittest discover -s benchmarks/engineering-fixture-integrity -p 'test_manifest_binding.py' -v
python benchmarks/engineering-fixture-integrity/audit_openscad_cgal_mesh.py \
  --fixture-root /path/to/unpacked/engineering-fixtures-v0.4
```

The preflight checks the package manifest before importing geometry. Every CAD
source/edit STEP, the assembly STEP, and each assembly part-definition STEP
must match its manifest SHA-256. It also checks case census, source/edited
variant pairing, six unique part definitions, seven assembly instances, and
the pinned Python/CadQuery/Open CASCADE/Trimesh/OpenSCAD toolchain.

## What this checks

CadQuery/OCP imports the AP214 STEP B-reps and tessellates 12 CAD source/edit
solids plus seven assembly instances. Each STL is processed through OpenSCAD's
CGAL Boolean path using a disjoint subtractor, then re-imported with Trimesh.
The audit requires one CGAL polyhedron conversion, CGAL's `Simple: yes`,
single connected watertight/winding-consistent meshes, bounded volume
deviation, and bounded bounding-box deviation.

The receipt binds the manifest SHA-256, each original source artifact SHA-256,
the derived STL SHA-256 values, and the exact toolchain. For assembly instances,
the parent assembly digest is the source identity; temporary per-instance STEP
files are recorded as derived artifacts.

## Development-run observation

A 2026-11-05 development run reported:
- 19/19 mesh subjects passed.
- Maximum OCP-input-STL to CGAL-output-STL volume deviation: 1.122e-6 relative.
- Maximum original B-rep to CGAL-output-STL volume deviation: 4.133e-4 relative.
- Maximum B-rep to CGAL-output bounding-box deviation: 0.00470 mm.

Those are observations from the prior v0.4 development run. The current
manifest-binding verifier revision has not been rerun against the v0.4 ZIP in
this execution environment. Nine manifest preflight/negative-control unit test
cases have been authored, but they have not yet been executed here. The draft
PR's GitHub workflows were skipped, not passed. Both the unit-test run and the
full 19-subject replay remain open validation tasks.

## Claim boundary

This is a **second-engine check of tessellated meshes**, not a second STEP
B-rep kernel. STEP parsing and tessellation still use Open CASCADE through
CadQuery OCP. It does not establish AP242 conformance, native FreeCAD history,
manufacturability, engineering approval, physical conformity, safety, or code
compliance. The report is tied to synthetic development fixtures and must not
be represented as a CAD agent benchmark score.
