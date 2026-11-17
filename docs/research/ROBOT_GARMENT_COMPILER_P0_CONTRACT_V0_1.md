# Robot Garment Compiler P0 — geometry admission contract v0.1

**Issue:** Symthaea [ROBOT-GARMENT-COMPILER-001 #12710](https://github.com/Luminous-Dynamics/symthaea/issues/12710)  
**Parent:** [ROBOT-LIGHTSHELL-001 #12688](https://github.com/Luminous-Dynamics/symthaea/issues/12688)  
**Implementation:** `symthaea_engineering::robot_garment_compiler`  
**Status:** synthetic geometry admission prototype; Rust tests authored but not run  
**Claim ceiling:** candidate geometric fit-screen only; no physical fit, protection, safety or manufacturability qualification

## 1. Why P0 starts with geometry admission

A robot-specific garment must respect articulated motion, cable routes, sensor fields, cooling and service access. The most consequential early mistake is generating an attractive panel that crosses a moving envelope or blocks a required service/sensor region.

Before attempting surface flattening or automatic pattern generation, P0 creates a deterministic, reviewable admission contract. It accepts only explicit robot-local geometry references and checks whether candidate panel bounds stay within a declared zone and outside protected volumes.

This is a deliberately conservative first step. Axis-aligned bounding boxes (AABBs) can overestimate occupied space and reject a candidate that a more precise mesh/OBB calculation might later accept. They are suitable only where the supplied bounds conservatively enclose the full panel/keep-out geometry in the same declared coordinate frame.

## 2. Files

- `docs/research/fixtures/robot_garment_compiler_v0_1/robot_garment_compiler.schema.json`
- `docs/research/fixtures/robot_garment_compiler_v0_1/single_elbow.example.json`
- `crates/domains/symthaea-engineering/src/robot_garment_compiler.rs`
- `crates/domains/symthaea-engineering/src/lib.rs` exports the module.

The fixture is synthetic and uses one robot-root frame in millimetres. It has one forearm shell zone, one candidate textile panel and four keep-outs:
- elbow swept-motion envelope;
- forearm cable route;
- wrist sensor aperture;
- service connector access.

The candidate is placed outside the expanded keep-out envelopes and remains within the declared forearm zone. Its material evidence is deliberately `unknown`, so the geometry can enter fit-screen planning but not material evidence admission.

## 3. Rust API

```rust
use symthaea_engineering::robot_garment_compiler::parse_and_evaluate;

let report = parse_and_evaluate(fixture_json)?;
if !report.eligible_for_fit_screen {
    for finding in report.findings.iter().filter(|finding| finding.blocking) {
        eprintln!("{} {}: {}", finding.code, finding.path, finding.message);
    }
    return Err("robot garment candidate rejected".into());
}
```

`eligible_for_fit_screen` means only that the provided AABB candidate passed this conservative synthetic/geometry admission contract. It does not establish physical fit or safe motion.

`material_evidence_admitted` is separately true only when every panel references a measured-coupon or qualified material profile. The geometry and material states cannot be conflated.

## 4. P0 invariants

### Source identity
- Explicit robot ID, revision, geometry state and optional geometry digest.
- A CAD-bound geometry requires a non-empty revision and valid SHA-256 digest.
- Synthetic fixtures remain explicitly synthetic.
- No implicit CAD repair, scale correction, axis flip or frame transform.

### Units and frames
- P0 accepts millimetres only.
- Every zone, keep-out and panel names a declared coordinate frame.
- A panel must use its zone's frame.
- A keep-out affecting a panel must use the same frame, unless a future contract supplies an explicit verified transform.
- Different frame IDs do not imply an identity transform.

### Geometric bounds
- Every bound must be finite and have strict min < max on all three axes.
- Every candidate panel must be contained by its declared zone.
- Keep-out clearance is expanded by `required_clearance_mm + uncertainty_mm`.
- Boundary contact counts as an intersection.
- A hard keep-out intersection blocks fit-screen admission.
- Advisory intersections are reported but do not block admission.

### Keep-out classes
The current contract supports:
- swept motion;
- pinch/shear;
- cable route;
- sensor aperture and optical field;
- thermal vent;
- service and emergency access;
- charging interface.

Unknown classes fail closed. The exact semantics and source geometry for each envelope still require owner review.

### Material evidence
- Unknown/supplier-declared material does not block geometric fit-screening.
- It does prevent `material_evidence_admitted`.
- Measured-coupon/qualified states require an exact material-profile reference.
- Supplier statements and model predictions cannot be promoted to measured textile properties.

## 5. Known-answer synthetic fixture

The fixture models a simple forearm region and elbow boundary:
- robot root coordinate frame, mm;
- forearm zone AABB;
- candidate textile-panel AABB;
- elbow sweep/cable/sensor/service keep-out AABBs;
- clearance and geometric uncertainty added before intersection testing.

The test suite contains eleven cases:
1. known-answer elbow fixture is eligible for fit-screen only;
2. panel entering joint swept envelope is rejected;
3. unknown coordinate frame fails closed;
4. panel outside declared zone is rejected;
5. clearance uncertainty contributes to the expanded envelope;
6. mismatched panel/keep-out frames require an explicit transform;
7. CAD-bound geometry without revision/digest is rejected;
8. unsupported units are not implicitly converted;
9. real candidate geometry cannot be admitted from a synthetic geometry state;
10. advisory keep-out intersections are reported without becoming hard blockers;
11. supplier-declared material remains distinct from measured-coupon/qualified material evidence.

These unit tests are authored but **not yet executed**. Run:
`cargo test -p symthaea-engineering robot_garment_compiler -- --nocapture`

No test pass is claimed until it runs against an exact source head.

## 6. What this prototype does not do

- It does not generate actual 2D patterns or flatten 3D surfaces.
- It does not solve textile stretch, drape, wrinkles, seam deformation or cloth/robot contact.
- It does not detect mesh collision or exact pinching; AABB checks are conservative broad-phase only.
- It does not infer thermal limits, cable bend radius or sensor FOV from CAD; these are supplied as owner-qualified envelopes.
- It does not choose fabrics or prove their mechanical properties.
- It does not make sewing/knitting machine programs.
- It does not qualify physical fit, comfort, impact performance, IP/IK, fire, ESD, hygienic suitability or human-contact safety.
- It does not authorize a robot, actuator or manufacturing machine.

## 7. Next geometry/compiler increments

1. Add transform records with source/target frame, revision, units, covariance/uncertainty and exact transform convention; reject cycles/inconsistent transforms.
2. Replace single AABB panel geometry with conservative triangle/mesh bounds and a qualified broad-phase/narrow-phase collision adapter, preserving geometry owner references.
3. Consume source-owned joint pose sets and swept envelopes, not invented representative poses.
4. Add panelization candidate generation on low-curvature synthetic surfaces.
5. Add 2D/3D distortion and developability estimates; pattern semantics remain delegated to #11750.
6. Add explicit seam/attachment graphs and test each path against keep-outs.
7. Bind physical fit/clearance measurements and invalidation to exact panel, material lot and robot article.
8. Compare against qualified manual patterning and manual CAD on the same robot/zone/ODD.

## 8. Benchmark design

Do not rank a candidate by global mesh similarity alone. The benchmark must report:
- local maximum surface deviation;
- joint swept-envelope false-negative rate (a hard gate);
- sensor aperture, vent, service and cable keep-out violations;
- required-pose coverage;
- AABB conservative false-positive rate against later mesh/physical references;
- seam and attachment topology preservation;
- material orientation preservation;
- human edits and correction time;
- fit-screen rejection rate and false admission rate.

The first safety-relevant target is zero accepted candidates with known keep-out intersections on the frozen fixture corpus. That is a test objective, not a universal proof of no collision.

## 9. Research frontier and strategic interpretation

The current research supports building this as an evidence-bearing compiler rather than a generative CAD demo:
- RGBench / GarmentDynamics is a useful cloth manipulation/simulator comparator but not a robot collision or protection benchmark.
- Recent humanoid textile-handling research highlights throughput, grasp stability, deformable perception, whole-body balance and hand reliability as unresolved deployment constraints.
- Smart textile and electronic-skin research is a separate signal/observation layer; its readings must not be treated as geometry or safety authority.
- Symtropy's cloth physics and Sol-Atlas geometry remain their own specialist owners.

The revolution is not "AI draws a robot jacket." It is a traceable, constraint-driven pipeline that can explain why each panel exists, which hazards it addresses, what it must avoid, how it can be manufactured, and what exact tests still remain.
