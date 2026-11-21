# REBCO-ANISO-3D-001 — local tape-frame anisotropy benchmark

**Tracking issue:** [Symthaea #13158](https://github.com/Luminous-Dynamics/symthaea/issues/13158)  
**Implementation branch:** `research/sc-aniso-3d-v0.1`  
**Initial slice:** synthetic global-to-local field transform auditor; no material surface or coil qualification is claimed.

## Research motivation

The 2026 commercial-tape study by Babouche et al., *Practical scaling parameters for critical current anisotropy in commercial REBCO coated conductors for high field applications*, Superconductor Science and Technology 39 (2026) 085021, DOI [10.1088/1361-6668/ae940e](https://doi.org/10.1088/1361-6668/ae940e), compares three commercial 4 mm tapes from SST, SuperPower and Faraday Factory Japan / SuperOx. It reports full-width measurements at UNIGE and same-spool patterned microbridge measurements at HFLSM, at different temperature / field ranges and angle sampling. The reported surfaces include tape-specific anisotropy and secondary pinning peaks; the paper also reports nonzero full-width / microbridge discrepancies. The data are stated to be available from the corresponding author on reasonable request, so the paper does not automatically grant access to a redistributable raw dataset.

A June 2026 open-access torque-magnetometry paper, *Method for three-dimensional Ic(theta,phi,B,T) anisotropy characterization of REBCO tapes using advanced torque magnetometry; experimental results at principal field anisotropies up to 45 T*, Superconductivity 18 (2026) 100257, DOI [10.1016/j.supcon.2026.100257](https://doi.org/10.1016/j.supcon.2026.100257), demonstrates a method for measuring a second angular coordinate `phi` as well as `theta`. This is a methodological comparator; it is not a complete spool-to-coil performance dataset.

The practical engineering issue is that a wound REBCO tape experiences a global magnetic-field vector whose direction relative to the tape changes along the as-built path. A scalar `Ic(B,T,theta)` fit with one nominal angle does not fully specify the local conductor condition when both in-plane and out-of-plane orientation matter.

## First executable slice

Files:
- Schema: `data/schemas/sc_rebco_local_tape_frame_v0_1.schema.json`
- Synthetic fixture: `tools/superconductivity-discovery-harness/fixtures/rebco_local_tape_frame_smoke_v0_1.json`
- Rust auditor: `tools/superconductivity-discovery-harness/src/bin/rebco_local_tape_frame_audit.rs`

The contract explicitly fixes:
- right-handed local basis `(longitudinal, width, normal)`;
- matrix columns are local unit axes expressed in global coordinates;
- `B_local = R^T B_global`;
- `theta` is measured from local tape normal;
- `phi` is in-plane azimuth from longitudinal toward width;
- angles are radians.

The auditor validates:
- 3x3 matrix shape and finite values;
- orthonormal basis within a fixed numerical tolerance;
- determinant +1 (rejects reflected / left-handed frames);
- inverse-transform consistency;
- unique station IDs and reel / segment / coordinate-frame lineage fields;
- nonnegative station coordinate and transform uncertainty;
- canonical lowercase SHA-256-shaped source-artifact digest;
- undefined orientation at zero / near-zero field, reported as null rather than a fabricated angle.

The synthetic fixture contains one identity frame, one rotated frame, and a zero-field case. The same global vector therefore resolves to different local tape angles at different stations, demonstrating why coil-local orientation must not be replaced by one global angle.

## What v0.1 does not do

- It does not solve the coil magnetic field.
- It does not reconstruct as-built winding geometry from manufacturing records.
- It does not estimate or propagate uncertainty through an anisotropy surface.
- It does not fit or interpolate `Ic(B,T,theta,phi)`.
- It does not convert full-width transport to patterned microbridge current.
- It does not infer performance beyond the measured condition envelope.
- It does not compute coil current margin, quench, or engineering acceptance.

The `transform_uncertainty_rad` field is retained as evidence, but uncertainty propagation into the angle / surface evaluation is intentionally deferred. This must be implemented before the output is used to make a physical margin claim.

## Next benchmark ladder

1. **Geometry/transform contract:** prove correct vector transforms for straight, curved, twisted, reversed and handedness-negative synthetic paths; fail closed on nonorthogonal or uncertain frames.
2. **Surface reconstruction:** source-pinned (Ic(B,T,theta,phi)) observations; compare scalar-angle interpolation, two-angle periodic interpolation, mechanism-informed pinning models, uncertainty-aware surfaces and conservative lower envelopes.
3. **Metrology transfer:** keep full-width transport, microbridge transport and torque-inversion estimates separate; use same-spool paired observations to quantify method discrepancy.
4. **Spool/lab transfer:** hold out complete spool, tape architecture, batch, laboratory and measurement geometry; no random angular-point splits.
5. **Coil-local margin:** transform a source-pinned field map into the as-built tape frame at each station and calculate the lower-bound (Ic) margin only inside the admitted ((B,T,theta,phi,strain)) domain.
6. **Adversarial qualification:** drop orientation, swap width/normal, reverse handedness, perturb winding registration, use unsupported field extrapolation, leak same-spool angular points, or mix microbridge and full-width data without a transfer model.

## Benchmark metrics

- (Ic) error on held-out angles / temperatures / fields;
- coverage and width of predictive intervals by tape, spool, lab and measurement geometry;
- angular error in the location of the anisotropy minimum and secondary peaks;
- local-frame rotation / vector residual;
- uncertainty in worst-station lower-bound (Ic);
- false-safe probability for coil-local current margin;
- abstention / out-of-domain rate;
- runtime and memory per metre / coil station at matched resolution.

Do not reward angular interpolation in an unsupported sector as a measured result. Record interpolation, extrapolation and direct measurement as separate evidence classes.

## Ownership

- Symthaea #11736 owns coil requirements, stored energy, quench protection and magnet-level comparison.
- Symthaea #5677 owns generic magnetics and field calculation.
- Symthaea #6122 owns model credibility / VVUQ.
- Symtropy #6550 owns spatial reel genealogy and coverage.
- Symtropy #6601 / PR #6602 owns exact test-condition profiles.
- Symtropy #6577 owns direct-vs-proxy metrology.
- Symtropy #6522 owns process-to-microstructure-to-conductor linkage.

## Claim ceiling

This slice validates synthetic coordinate and vector-transform semantics only. It does not establish the accuracy of a real as-built coil frame, a measured (Ic) surface, improved conductor margin, quench safety, or a physical magnet qualification. Real claims require rights-cleared source data, a calibrated as-built field map, verified tape-frame registration, method-transfer qualification and independent validation.
