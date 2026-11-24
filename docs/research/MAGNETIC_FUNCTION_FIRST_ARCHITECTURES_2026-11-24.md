# MAG-FUNC-001 — Function-first spatially programmed magnetic architectures

**Revision:** 0.1.0  
**Prepared:** 2026-11-24  
**Status:** research constitution and analytic reference; no finite-magnet simulation or physical component is claimed  
**Symthaea issue:** [MAG-FUNC-001 #13521](https://github.com/Luminous-Dynamics/symthaea/issues/13521)  
**Symtropy process owner:** [MAG-FUNC-PROC-001 #7310](https://github.com/Luminous-Dynamics/symtropy/issues/7310)

## Executive thesis

Move magnetic design from “maximize bulk material properties, then shape a magnet” toward **design the required magnetic function, then co-design geometry, material zoning, local anisotropy, manufacturing and magnetization to realize it**.

For many applications, the useful object is a spatial field function:
- vector field over a defined three-dimensional domain;
- local field gradients and polarity boundaries;
- force/torque as a function of position and operating state;
- leakage field and coupling to neighbouring components;
- field stability over temperature, mechanical load and life.

The inverse-design chain is:

required field / force / torque / gradient
→ target field map and uncertainty envelope
→ magnetic topology and material zoning
→ local easy-axis / magnetization direction
→ geometry and interfaces
→ manufacturing and magnetization sequence
→ predicted field and component function
→ measured field map and discrepancy-led redesign

This is not a claim that the inverse problem is solved or that additive manufacturing is automatically better than conventional assembly.

## Source-pinned frontier

### Spatially patterned hard/soft magnetic poles

Behera, Lv & Singamneni, “Patterned magnetic pole configurations in bonded magnets,” *Scientific Reports* 16, 13102 (published 11 March 2026), DOI [10.1038/s41598-026-43131-5](https://doi.org/10.1038/s41598-026-43131-5).

The work demonstrates multi-material selective laser sintering of nylon composites with local hard NdFeB and soft FeSi/FeCo magnetic regions and local field orientation. The initial demonstration used two magnetic islands and a five-layer, 50 μm build. The article reports a 230 mT field in the print region and post-print magnetization fields of 1.5–1.9 T. It reports mT-scale probe fields and spatially distinct polar regions.

**Measurement boundary:** reported surface probe-field contours and differential field maps under stated external-field conditions are not intrinsic material remanence Br, intrinsic coercivity Hcj, or maximum energy product. A proper reproduction needs the raw point grid, probe position/standoff/orientation/calibration, background-field control, exact magnetization history and subtraction method.

The paper also notes powder-placement variability and field-driven material redistribution as sources of output variability. These are central manufacturing variables. The article says some equipment/process solutions are patent-pending. Treat exact apparatus embodiments as IP-sensitive and perform prior-art/freedom-to-operate review rather than copying them.

### Field-aligned lithography plus electron-beam post-cure

Kim et al., “Magnetic alignment and electron beam post-curing of NdFeB materials via lithography-based composite manufacturing,” *Virtual and Physical Prototyping* 20 (online 5 December 2025), DOI [10.1080/17452759.2025.2593543](https://doi.org/10.1080/17452759.2025.2593543).

This combines in-situ magnetic alignment in 60–80 wt% NdFeB photocurable slurry with electron-beam post-curing (EBPC). The study reports field-alignment-associated remanence improvements of 42.05–137.40% and coercivity improvements of 14.77–122.82%, depending on configuration. An optimal 100 kGy EBPC condition reportedly increased flexural strength and modulus by more than fivefold; the study used three replicates per condition.

At one reported endpoint, the paper writes Hci approximately 1.90 T, Ms approximately 101.02 A·m²/kg, and Mr approximately 87.57 A·m²/kg. Preserve its units as written pending full table-level verification. Mass-specific magnetization cannot be compared directly with volumetric polarization/remanence in T without source-qualified density and consistent definitions.

**Dataset identity issue:** the publisher states Zenodo DOI 10.5281/zenodo.17404587; the NLM Dataset Catalog surfaces a record titled for the same work at 10.5281/zenodo.17510759 with CC BY 4.0. The two identifiers are in conflict. Do not ingest either as an authoritative dataset until the DOI is resolved against publisher, author or Zenodo metadata.

### Recycled NdFeB process-route comparators

- Schäfer et al. (2026), recycled EOL hot-deformed NdFeB powder in PA12 by LPBF: [MAG-AM-001 #13506](https://github.com/Luminous-Dynamics/symthaea/issues/13506), DOI [10.1002/aesr.202500339](https://doi.org/10.1002/aesr.202500339). Source-reported printed remanence is 0.34 T and coercivity 1,238 kA/m for its HD-derived route, with 32.6% porosity and source-defined alignment degree 0.75.
- Rathfelder et al. (2025), recycled NdFeB by powder extrusion moulding (PEM), debinding and sintering: DOI [10.1038/s41598-025-19111-6](https://doi.org/10.1038/s41598-025-19111-6). It reports Br/Js up to 0.96; an M1 200 °C nozzle case gives Br = 984 mT, Hcj = 1,255 kA/m, (BH)max = 158 kJ/m³, density 7.21 g/cm³. This is a sintered output, not a polymer-bonded printed magnet. Br/Js is not the same alignment metric as the LPBF <cosθ> metric.

These process routes demonstrate why “best printed magnet” cannot be a single pooled ranking. Geometry, density, material form, post-processing, magnetization and application all change the comparison.

## Design paradigm

### 1. Specify the field function first

Bind an application-specific target:
- vector B(x,y,z) or H(x,y,z) over a defined volume/surface;
- reference frame, sample grid, standoff and polarity map;
- allowable field-map error and uncertainty;
- required force/torque or gradient over position/current;
- operating temperature, external field and duty cycle;
- leakage-field constraints and downstream interfaces.

A sparse set of scalar values cannot substitute for a field map.

### 2. Co-design material, topology and magnetization

Candidate variables may include:
- hard/soft/nonmagnetic region geometry and volume fractions;
- magnet material grade and process-conditioned B-H/recoil curves;
- local easy-axis and magnetization direction;
- pole/yoke/return-path topology, gap and shielding;
- interfaces, gradients and local material dispersion;
- post-print magnetization sequence.

Do not presume a Halbach-like arrangement or hard/soft pairing wins. Mutual demagnetization, interface defects, material nonlinearity and local saturation may defeat the nominal concept.

### 3. Compare integrated parts with conventional assemblies

The engineering hypothesis is that a functionally programmed monolithic/graded part may achieve comparable or better application function while reducing separate magnet segments, assembly steps, adhesive/fastener interfaces, tolerance accumulation or mass.

This must be tested at equal application target, material budget, measurement geometry and duty condition. Conventional assembled multipole/Halbach components remain the baseline—not a straw-man uniform magnet.

### 4. Use Symthaea and Symtropy for different jobs

Symthaea owns requirement-to-field-function translation, material/source/evidence identity, candidate and topology selection, uncertainty, Pareto comparison and experiment selection.

Symtropy owns exact component/region geometry, scenario identity, solver orchestration, run receipts/replay, manufacturing/process twin and predicted-to-measured field-map discrepancy analysis.

Neither system replaces qualified magnetostatic, nonlinear B-H, hysteresis, eddy-current or micromagnetic solvers. Predicted fields remain predictions.

## Qualification and benchmark ladder

### F0 — target and source admission
Freeze target function, geometry, operating profile, measurement grid, exact material grades, evidence class, source rights and patent/IP status.

### F1 — analytic known-answer reference
The companion machine-readable fixture at data/research/magnetic_function_field_reference_cases_v0_1.json defines ideal point-dipole cases and metamorphic properties:
- axial/equatorial field sign and magnitude;
- two-dipole superposition and symmetry;
- linearity in dipole moment;
- inverse-cube scaling under geometric scale;
- global sign reversal and rotational covariance;
- singularity rejection.

This reference is exact for ideal point dipoles in free space, not finite magnets or nonlinear materials. It tests coordinate/sign/unit/superposition correctness and provides a far-field sanity reference.

### F2 — external solver verification/validation
Use qualified external solvers for finite geometries and material laws. Bind exact geometry, constitutive inputs, B-H/hysteresis model, boundary conditions, mesh, solver version, convergence and uncertainty. Check the solver against analytic references before physical comparison.

### F3 — published-case reproduction
Reconstruct the Behera et al. two-region configurations if raw points and apparatus metadata become available. The article's raw data are available from authors on request; a digitized contour is not a substitute for exact 3D data.

Reproduce LCM/EBPC only after the dataset DOI conflict and exact magnetic-property units are resolved.

### F4 — matched architecture tournament
Compare:
1. conventional segmented/assembled array;
2. uniform one-piece bonded magnet;
3. printed multi-material hard/soft topology;
4. printed local-easy-axis topology;
5. functionally graded/topology-optimized candidate.

Use the same application target, measurement grid, material budget, operating condition and compute/physical-experiment budgets. Stratify routes where material form differs.

### F5 — component and lifetime qualification
Measure force/torque vs position, ripple, efficiency/losses, field leakage, thermal behaviour, mechanical integrity, ageing, dimensional tolerance and service-life evidence for a named component.

## Scorecard

Keep metrics independent:
- vector field-map MAE/RMSE and uncertainty coverage;
- target-field correlation and polarity-boundary localization;
- peak field and field-gradient error;
- leakage field under defined external/background conditions;
- force/torque constant and ripple where applicable;
- material-region and easy-axis placement error;
- local saturation/demagnetization margin;
- mass/volume, interface count, assembly steps and tolerance stack;
- yield, repeatability, accepted-part cost, process energy and lifecycle burden;
- out-of-distribution detection and abstention.

No hidden scalar should compensate for violating a hard functional or safety constraint.

## Prospective targets

After source maps and measurement uncertainty are available, freeze exact numeric limits. Initial research goals:
- reproduce the published two-region field topology;
- beat a matched conventional assembled architecture on a declared Pareto dimension without violating other hard constraints;
- reduce assembly interfaces or accepted-part cost while preserving application field/force/torque;
- validate field-map predictions on held-out geometries and material lots;
- show active design selection beats random, space-filling, factorial/fractional-factorial and conventional Bayesian optimization at equal budgets.

These are proposed goals, not current results or standards.

## Claim and safety ceiling

No physical coil/current/magnetization command, energized field-generation apparatus, laser/EBPC operation, powder handling, component release or deployment authority is introduced. Physical experiments require qualified facilities, process and radiation-safety review, equipment interlocks and independent authorization. Patent-pending apparatus disclosed by third parties requires legal/IP review.

A synthetic analytic reference is not a finite-magnet model. Solver convergence is not physical validation. A surface probe map under external field is not intrinsic remanence. A coupon is not a qualified motor/actuator. No SOTA superiority is claimed.

## References

1. Behera et al. (2026), [Patterned magnetic pole configurations in bonded magnets](https://doi.org/10.1038/s41598-026-43131-5).
2. Kim et al. (2025), [Magnetic alignment and EBPC of NdFeB composites](https://doi.org/10.1080/17452759.2025.2593543).
3. Rathfelder et al. (2025), [Recycled NdFeB powder extrusion moulding](https://doi.org/10.1038/s41598-025-19111-6).
4. Schäfer et al. (2026), [Recycled NdFeB LPBF bonded magnets](https://doi.org/10.1002/aesr.202500339).
5. He et al. (2026), [Active data selection and regression for sintered NdFeB](https://doi.org/10.1038/s41524-025-01914-w).
6. IEC 60404-18:2025, [measurement of permanent-magnet properties in an open circuit using a superconducting magnet](https://webstore.iec.ch/en/publication/76495).
