# Marine CAD Backend Evaluation (G3b)

Status: research comparison / capability selection. No backend is qualified merely because it advertises STEP, NURBS or B-rep support.

## 1. Candidate matrix (15 Nov 2026)

| Candidate | Geometry/kernel scope | Documented STEP capability | Current interpretation |
|---|---|---|---|
| Open CASCADE 8.0.1 | Mature C++ CAD kernel; B-rep, NURBS, topology, modelling and XDE | STEP import/export; AP242 must be selected explicitly; default writer schema is AP214 | First industrial/open-source baseline candidate; AP242:2025 Edition 4 must be proven from actual output and CAx-IF cases |
| Rust cad-kernel 0.6.3 | Exact analytic/NURBS B-rep, Booleans, sewing, validity, exact measures and meshing | Documented output AP214 | Valuable Rust-native geometry cross-check candidate; not an AP242:2025 writer; project describes itself as extracted/not yet general-use |
| Rust brep-to-step 0.2.1 | Kernel-neutral shape-to-STEP writer, curves/surfaces up to NURBS | Minimal shape-only AP242 Edition 2 product/unit/context | Possible writer component experiment; not evidence of Edition 4 PMI/PDM/validation-property support |
| Rust vcad_kernel_step 0.10.0 | vcad kernel STEP import/export | Documented AP214 | Conversion diagnostic only, not AP242:2025 profile qualification |
| Commercial CAD kernel | Mature native B-rep/STEP interoperability | Vendor/release-specific | Needed as an independent import/round-trip comparison where licensed access exists |

Version facts and capability claims must be rechecked before a backend is pinned. Record source URL, release/tag, binary/build digest, compile flags, STEP settings and all test artifacts in the capability manifest.

## 2. Recommended architecture

- **Rust orchestration and evidence plane:** Symthaea owns candidate identities, semantic scope, tolerance profiles, capability manifests and receipts.
- **OCCT adapter:** a small isolated C++17 process or C ABI wrapper, pinned to OCCT 8.0.1 initially. Do not let C++ library types leak into the Rust public API.
- **Independent comparison:** use a separate kernel family for import-back; Rust cad-kernel may become one candidate after its general-use and test-corpus limitations are characterized.
- **STEP writing:** AP242:2025 Edition 4 must be explicitly configured and checked in the actual Part 21 FILE_SCHEMA declaration. Do not assume that an AP242DIS or a generic “AP242” setting means Edition 4.
- **Capability evidence:** every PASS or FAIL capability result references both a frozen fixture-manifest digest and test-report digest. The capability manifest digest binds backend build/configuration and its evidence.
- **No self-certification:** same-kernel export/import is a diagnostic control only.

## 3. CAx-IF test resources

Use current CAx-IF/MBx implementation practices for:
- geometric and assembly validation properties;
- PMI representation and presentation;
- persistent IDs;
- tessellated 3D geometry;
- cross-domain STEP/QIF traceability (including Test Cases 7 and 8).

The published CAx-IF sample assets have their own redistribution/reuse terms. Preserve their official URL, version and provenance; do not mirror them into this repository until rights are confirmed. Track current CAx-IF versions separately from ISO edition numbers.

## 4. Required bake-off

For each candidate/backend build, record:
- capability manifest and backend identity;
- import/export schema and actual FILE_SCHEMA;
- exact B-rep and tessellated representation support separately;
- analytic, NURBS, periodic, trimmed and p-curve/seam behavior;
- shell/solid counts, topology checks and self-intersection capabilities;
- validation properties, PMI/GD&T, persistent IDs, product structure, materials and revision handling;
- geometry deltas (dimensions, volume, area, centroid, inertia, symmetric surface distance and edge gap);
- repair events, tolerance widening and expert intervention;
- wall-clock, peak memory and total validation cost.

Test the same source geometry through:
1. same-kernel round trip (diagnostic control);
2. OCCT export -> independent kernel import;
3. independent kernel export -> OCCT import;
4. neutral tessellation -> independent comparison;
5. native/STEP/round-trip comparisons of semantic metadata and validation properties.

## 5. Selection rule

Do not choose the backend with the longest feature list. Choose the smallest trustworthy toolchain that passes the frozen corpus and has the lowest total cost to an independently verified artifact.

For AP242:2025 Edition 4, a backend remains INCOMPLETE until schema edition, geometric representation, required metadata and independent import evidence have all been checked. A successful AP214 conversion, AP242 Edition 2 output or STEP parse is not a pass for the Edition 4 profile.

The current G3b implementation provides a typed backend capability manifest and evaluator. It does not yet invoke a CAD kernel or export/import STEP bytes.
