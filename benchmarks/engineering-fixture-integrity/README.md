# Engineering fixture integrity extensions

Companion source for the optional OpenSCAD / CGAL audit in the synthetic
engineering-fixtures-v0.4 package.

## Run

1. Unpack `engineering-fixtures-v0.4.zip`.
2. Install the package's pinned Python mesh-audit requirements.
3. Install OpenSCAD 2021.01 and make `openscad` available on PATH.
4. Run from the unpacked package directory:

```bash
python audit_openscad_cgal_mesh.py
```

Or supply `--fixture-root` to point at the unpacked package.

## What this checks

CadQuery/OCP imports the AP214 STEP B-reps and tessellates 12 CAD source/edit
solids plus seven assembly instances. Each STL is processed through OpenSCAD's
CGAL Boolean path using a disjoint subtractor, then re-imported with Trimesh.
The audit requires CGAL's `Simple: yes`, a watertight and winding-consistent
re-exported mesh, bounded volume deviation, and bounded bounding-box deviation.

Observed in the 2026-11-05 development run:
- 19/19 mesh subjects passed.
- Maximum OCP-input-STL to CGAL-output-STL volume deviation: 1.122e-6 relative.
- Maximum original B-rep to CGAL-output-STL volume deviation: 4.133e-4 relative.
- Maximum B-rep to CGAL-output bounding-box deviation: 0.00470 mm.

## Claim boundary

This is a **second-engine check of tessellated meshes**, not a second STEP
B-rep kernel. STEP parsing and tessellation still use Open CASCADE through
CadQuery OCP. It does not establish AP242 conformance, native FreeCAD history,
manufacturability, engineering approval, physical conformity, safety, or code
compliance. The report is tied to synthetic development fixtures and must not
be represented as a CAD agent benchmark score.
