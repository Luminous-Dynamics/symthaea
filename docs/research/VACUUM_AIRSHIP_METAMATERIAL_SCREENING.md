# Vacuum-airship metamaterial screening: research and implementation plan

**Status:** research design and arithmetic screening only  
**Parent:** [PERSISTENT-PLATFORMS-001 (#7343)](https://github.com/Luminous-Dynamics/symthaea/issues/7343)  
**First code tranche:** `symthaea_engineering::vacuum_airship_screening`

## Decision

Treat vacuum buoyancy as a high-risk structural-feasibility branch, not as an established vehicle technology. The first Symthaea contribution should be a deterministic SI mass-balance and pressure-evidence screen. It must not invent a material strength, infer buckling capacity from lattice geometry, or equate a converged simulation with a qualified structure.

An architected lattice is worth studying because topology changes load paths and stiffness-to-mass efficiency. It does not repeal the mass budget or the instability of thin shells under external pressure. The vehicle only works if a complete pressure-resistant envelope, frame, joints, seals, propulsion, energy storage, controls, landing hardware, and contingencies together remain below the available buoyant mass.

## Verified research anchors

1. **Toli (October 2026), “The kenemostat: Structural feasibility of a vacuum-buoyant geodesic vehicle including global buckling,” Aerospace Science and Technology 177(C), 112799.** [DOI](https://doi.org/10.1016/j.ast.2026.112799). The paper analyzes a geodesic polyhedral frame and distinguishes individual-member Euler buckling from global shell-like buckling. Its abstract reports analytical and finite-element agreement for its modeled geometry and estimated payload scaling beyond a break-even size. It also identifies limitations including joint design, manufacture, and dynamic loading. This is a computational study, not a fabricated, pressure-tested, or flight-demonstrated airship. Its idealized members, joint assumptions, and imperfection model must be reconstructed before its estimates become inputs to this repository.
2. **Akhmeteli & Gavrilin (2021), “Vacuum Balloon—A 350-Year-Old Dream,” Engineering 2(4), 480–491.** [Open article](https://doi.org/10.3390/eng2040030). The paper examines a rigid spherical sandwich shell and finite-element buckling analysis using commercially available material data. The authors explicitly note simplifying assumptions and that manufacturing imperfections were outside the FEA scope. It is a useful alternate architecture, not validation of an airship.
3. **Recent progresses in lightweight carbon fibre reinforced lattice cylindrical shells (2022), Progress in Aerospace Sciences 135, 100860.** [DOI](https://doi.org/10.1016/j.paerosci.2022.100860). This review covers lattice, lattice-sandwich, and stiffened composite shells, including manufacturing and failure modes. Cylindrical-shell findings are relevant background but must not be transferred directly to a geodesic sphere without a justified model.
4. **Multi-material and multi-manufacturing lattice truss sandwich structures: Conception and practice (2026).** [DOI](https://doi.org/10.1016/j.compositesb.2025.113125). This work reports fabrication of composite sandwich structures with a lattice core. It supports exploring manufacturable lattice-core architectures; it does not establish vacuum-airship pressure capability.
5. **NASA/SP-8007-2020/REV 2, Buckling of Thin-Walled Circular Cylinders.** [NASA Technical Reports Server](https://ntrs.nasa.gov/citations/20205011530). NASA emphasizes imperfection sensitivity and the limits of idealized buckling calculations for cylindrical shells. This guideline is not a universal knockdown-factor source for geodesic frames or spherical composite shells. Geometry-specific buckling guidance and testing remain necessary.

## Why the first model is intentionally small

For an explicitly provided enclosed volume (V), ambient air density (ho_a), interior density (ho_i), and a complete declared non-payload mass (m_v):

[
m_{\mathrm{net\ lift}}=(\rho_a-\rho_i)V
]

[
m_{\mathrm{payload\ allowance}}=(\rho_a-\rho_i)V-m_v
]

Use (ho_i=0) only for the idealized vacuum limit. Use operating-condition-specific densities for all fair comparisons. A negative payload allowance is retained rather than clamped to zero, so a design that is too heavy remains visibly infeasible.

For an initial spherical reference geometry:

[
V=\frac{4}{3}\pi R^3
\qquad
A=4\pi R^2
]

At a declared standard sea-level air density of 1.225 kg/m³:

| Sphere radius | Enclosed volume | Ideal gross vacuum lift as mass |
|---:|---:|---:|
| 20 m | 33,510.322 m³ | 41,050.144 kg |
| 40 m | 268,082.573 m³ | 328,401.152 kg |

These values are arithmetic reference cases, not structural designs or payload claims. They exclude every kilogram of vehicle mass and make no assertion that the shell can withstand the external pressure.

Vacuum has higher gross lift than helium by the helium mass that would otherwise occupy the same volume, not by the full mass of displaced air. The density comparison must use the same temperature and pressure. The additional lift is small compared with the total displaced-air mass, so the structural mass penalty may dominate.

## Cargo-transport assessment: vacuum must beat the helium mass penalty

The cargo use case is credible only as a conditional research target. It is not yet supported as a feasible vehicle by the available evidence. The strongest first comparison is vacuum against helium at the *same external displacement volume*, with the full non-payload mass for each configuration entered separately.

For a common external volume \(V\):

\[
m_{\mathrm{lift,vac}}=\rho_{\mathrm{air}}V
\qquad
m_{\mathrm{lift,He}}=(\rho_{\mathrm{air}}-\rho_{\mathrm{He}})V
\]

Therefore the vacuum vehicle's extra gross lift is only:

\[
\Delta m_{\mathrm{gross}}=\rho_{\mathrm{He}}V
\]

Vacuum carries more payload than helium only when its *additional* non-payload vehicle mass is less than \(\rho_{\mathrm{He}}V\). This threshold does not estimate how heavy the vacuum pressure hull will be; it tells the researcher exactly what that hull and its additional systems must beat.

Using standard sea-level ambient density 1.225 kg/m³ and helium density 0.16928 kg/m³ as illustrative matched-condition inputs:

| Spherical radius | Volume | Ideal vacuum gross lift | Ideal helium gross lift | Maximum incremental vehicle-mass penalty for vacuum to retain more payload |
|---:|---:|---:|---:|---:|
| 20 m | 33,510 m³ | 41.05 t | 35.38 t | 5.67 t |
| 40 m | 268,083 m³ | 328.40 t | 283.02 t | 45.38 t |

These are gross lift calculations, not vehicle payload results. Actual atmospheric conditions alter all densities. A helium craft still has its own envelope, ballonets or buoyancy-control equipment, propulsion, batteries/fuel, cargo handling, and structure. A fair study must compare complete designs and common safety requirements, not compare a vacuum shell with an otherwise fully-equipped helium airship.

A practical scale check is a **100-tonne cargo target**. A 20 m radius sphere has only 41.05 tonnes of ideal gross vacuum lift, so it cannot carry 100 tonnes even if the vehicle weighed nothing. A 40 m radius sphere has 328.40 tonnes of gross vacuum lift; after a 100-tonne cargo load, no more than 228.40 tonnes remain for *the entire vehicle* before any operating reserves or margins. That gross bound is misleadingly generous: Toli's 2026 geodesic-frame analysis estimates roughly 4 tonnes of payload at R=20 m and 60 tonnes at R=40 m for its specific assumed structural model, below a 100-tonne target at the 40 m scale. Akhmeteli & Gavrilin's 2021 sandwich-shell analysis assumes a payload fraction q=0.1; applying that declared fraction to the gross displacement mass gives a rough 4.1-tonne payload at 20 m and 32.8 tonnes at 40 m before additional vehicle systems. These papers use different structures and assumptions, so their estimates are not directly interchangeable; neither represents a built and demonstrated cargo airship.

The Rust module now exposes `compare_vacuum_and_helium` to keep this mass penalty explicit, plus `screen_cargo_fit` to check cargo mass and cargo volume independently against declared capacities. Usable cargo volume must be supplied from the chosen configuration; it is never assumed equal to the entire hull volume. Passing these arithmetic gates does not establish route or operational viability.

## What cargo market could suit an eventual vehicle?

Existing studies suggest that conventional cargo airships, if made operationally available, might serve as a middle tier between slow/low-cost surface freight and costly/faster air freight, especially for remote delivery and project cargo. A mode-choice study of 96 logistics professionals found hypothetical market shares from 5% to 34% depending on cargo type; these stated-preference results are not proof of actual uptake. A 2024 refrigerated freight comparison modeled a 100-tonne electrically powered cargo airship against long-haul trucking. A separate solar-airship route study projected very low energy use and emissions but several-fold longer flight times than conventional aircraft. These results support researching a niche, not assuming a universal freight advantage.

For a vacuum airship, the practical first niches would be high-value or otherwise difficult-to-deliver cargo where slow travel is acceptable and remote landing / hovering access is valuable. It must first beat a helium cargo airship structurally and economically. The principal questions are whether the extra pressure-resistant mass fits under the helium-mass ceiling, whether cargo can be loaded without compromising the hull, how loading/unloading changes mass and trim, and what happens after damage or loss of vacuum. A standard pressurized cargo bay within a vacuum envelope is possible only by paying for its pressure boundary and supports, and by counting every associated mass and displacement effect. External sling cargo might simplify the interior-volume issue but makes load attachment, sway, weather, clearance and ground handling harder.

Research context:
- Toli's 2026 geodesic study discusses idealized structural feasibility and estimates critical size / payload scaling, but remains an analysis rather than a flight demonstration: [Aerospace Science and Technology](https://doi.org/10.1016/j.ast.2026.112799).
- Akhmeteli & Gavrilin's 2021 sandwich-shell study estimates a small theoretical payload fraction using commercial-material properties but says manufacturing imperfections were not explicitly modelled in their analysis: [Engineering](https://doi.org/10.3390/eng2040030).
- Prentice's cargo-airship mode-choice study surveyed 96 logistics professionals and reports hypothetical 5–34% market share depending on freight type: [Journal of Air Transport Management](https://doi.org/10.1016/j.jairtraman.2019.101741).
- Prentice's 2024 study compares trucking with a modeled 100-tonne electric cargo airship for refrigerated supply chains: [Journal of Transportation Technologies](https://doi.org/10.4236/jtts.2024.142012).
- A 2023 route-optimization study for solar airships estimates reduced energy/emissions in its model with longer travel times than conventional aircraft: [International Journal of Automotive and Mechanical Engineering](https://doi.org/10.1080/14786451.2023.2189488).

## Material concept: a strong, recoverable “structural bubble”

The phrase “strong bubble that flexes back into shape” points toward a useful requirement, but it should not be interpreted as a soft rubber envelope holding its shape by itself. An ordinary gas balloon relies on positive internal pressure to tension its skin. A vacuum envelope sees atmospheric pressure pushing inward. A soft membrane can remain intact in tension while still deflecting so far that its supported geometry becomes unusable; a compressed frame can snap through or buckle before its fibers reach their tensile strength.

**Working concept:** a fiber-reinforced membrane + an elastic-but-stiff geodesic skeleton + redundant tension bracing + compliant, damage-tolerant joints. Make the pressure barrier replaceable and non-primary where possible; make the structural architecture resist buckling without requiring large deformations; use elastic recovery as a secondary property for small, bounded loads rather than as the primary collapse-resistance mechanism.

### Candidate material stack

| Layer / subsystem | First material families to test | Role and principal caveat |
|---|---|---|
| Primary compression skeleton | Continuous-fiber CFRP beams or pultruded/formed composite profiles; also compare a carbon-fiber thermoplastic composite (CFRTP) profile | High specific stiffness is crucial because buckling scales with geometry and modulus. CFRP laminates are anisotropic and sensitive to fiber alignment, joints, and impact-induced delamination; coupon strength alone does not qualify a strut or frame. |
| Load-sharing shell / node regions | 3D-woven or through-thickness-reinforced composite; graded lattice/sandwich region around nodes | Woven through-thickness architecture is worth investigating for delamination and compression-damage resistance. Published multi-scale analysis of 3D-woven composite spherical shells under external hydrostatic pressure shows that weave architecture, radius/thickness ratio, and initial defects strongly affect strength and buckling. This is promising adjacent evidence, not proof for a full-scale atmospheric vacuum vessel. |
| Pressure seal / membrane | Thin, replaceable, low-permeability film or elastomeric barrier backed by continuous-fiber reinforcement where load-carrying membrane tension is needed | The soft layer provides sealing, abrasion tolerance, and accommodation of very small geometric change. Do not credit it with primary compressive stiffness unless measured and structurally modelled. Its seams, penetrations, permeability, aging and tear propagation are first-class design variables. |
| Internal bracing | Redundant tension cords/cables, with high-specific-strength fiber such as UHMWPE as a candidate | Cables work in tension only; they need an explicit load path and pretension policy. Creep, temperature, abrasion, termination efficiency, inspection, and long-term loss of pretension must be measured. The kenemostat paper itself explores a self-equalizing internal cable system, but its idealized mass and load distribution must be independently checked. |
| Recoverable joints / local compliance | Replaceable flexures, toughened composite nodes, mechanically captured interfaces, or qualified thermoplastic-composite joints | Permit tiny elastic motion and distribute impact while retaining alignment. Joint rotation, backlash, stress concentration, creep, fatigue and loss of pretension can lower global stiffness. Compliant joints must not introduce a snap-through mechanism. |
| Damage monitoring / repair | Strain or fiber-optic sensing, local vacuum/leak sensing, replaceable seal tiles, inspectable mechanical connections | These detect damage and support a controlled repair process. They do not make the structure self-healing or safe after damage unless a validated residual-strength case says so. |

A thermoplastic matrix is attractive where welding and repairability could reduce joint count and ease repair, but those gains should be treated as manufacturing hypotheses. Reviews of aerospace thermoplastic-composite joining describe resistance, ultrasonic, induction and other fusion-welding routes, while also stressing process control, joint strength, nondestructive inspection and qualification challenges.

### The main structural variant to pursue first

Use **Variant A as the conservative baseline**: a triangulated geodesic frame with a bonded/stiffened load-sharing skin and replaceable low-permeability outer barrier. Compare it against **Variant B**: a detached, fiber-reinforced membrane supported in tension by geodesic frame edges. In the October 2026 kenemostat analysis, both are examined; the bonded/stiffened variant is modelled as more globally stiff, while the detached-membrane variant is lighter in the idealization but needs careful frame and global-buckling treatment. Their quoted buckling values are numerical predictions for the paper's assumed idealized geometry and material properties, not validated allowable pressure for our candidate.

My initial priority is the **hybrid**: a stiff main geodesic compression network, membrane panels supported over controlled short spans, 3D-woven/toughened material at high-gradient zones, and redundant tension bracing. Add flexibility only in locations where it provides measured benefit; avoid a globally soft shell.

### Pressure load scale the frame must return from

For an ideal sphere, the compressive resultant transmitted across a great-circle cut is

```text
C = max(p_external - p_internal, 0) · πR²
```

At a full sea-level differential of 101.325 kPa, this is approximately **127.33 MN at R=20 m** and **509.32 MN at R=40 m**. The latter is four times the former because this resultant scales with radius squared. These values are global section-load scales; they are not the load on any single member, which depends on the geodesic topology, member distribution, joint stiffness, cable preload, skin coupling and imperfections.

The Rust screen now exposes `screen_spherical_pressure_load` to calculate that ideal resultant. Use it to establish the order of magnitude and generate solver load subjects—not to infer that an individual member can carry a particular fraction of the load or that a vessel is stable.

### Define “flexes back into shape” quantitatively

Do not accept a visual impression that the sphere “looks round again.” For a frozen test article and load profile, measure all of the following:

1. **Loaded shape:** the full-field displacement/deformation map while external differential pressure is applied, not just one diameter or a few points.
2. **Residual shape:** maximum and RMS surface deviation from the baseline geometry after unloading, with metrology uncertainty stated.
3. **Stiffness retention:** pressure–deformation curve before and after the prescribed cycle; record stiffness degradation and hysteresis.
4. **Damage / leakage:** delamination or matrix damage inspection, joint movement, fiber breakage, seal leakage and ingress rate.
5. **Capacity retention:** repeat the qualified structural capacity analysis/test after cycling. Shape recovery does not imply retained buckling capacity.
6. **Repeatability:** repeat across independently manufactured specimens, not just multiple cycles on one unusually good sample.
7. **Environment:** repeat under declared temperature, moisture, aging, UV/thermal-cycle and impact/damage conditions relevant to service.

The code now includes `assess_elastic_recovery`, which evaluates recorded peak and residual full-field deformation, post/pre stiffness retention, residual-shape metrology uncertainty, and a pre-registered cycle count. The residual geometry gate is **Pass** only if its measurement upper bound meets the declared tolerance; **Fail** only if the lower bound exceeds tolerance; otherwise it is **Indeterminate**. A recovery metric can be negative if residual deformation exceeds peak displacement, and stiffness loss fails separately even when the skin appears to return to shape. The API checks recovery claims only—it cannot establish pressure capacity or structural safety.

The acceptance thresholds are **not yet chosen**. They must be based on geometry tolerances, seal requirements and an independently justified structural safety case before test results are exposed. A sudden snap-through or unstable post-buckling event is a failure mode, not an acceptable way to flex and rebound. An evacuated-shell test can involve catastrophic implosion; subscale pressure tests require a suitably rated chamber, remote operation, exclusion controls and a separately approved test plan.

### Design-of-experiments matrix for Symthaea

Do not optimize on “material strength” as a single number. Freeze a testable candidate matrix with factors such as:

- **Architecture:** bonded skin; detached tension membrane; lattice-core sandwich; graded hybrid.
- **Fiber/textile architecture:** unidirectional laminate baseline; quasi-isotropic layup; woven/3D-woven/through-thickness reinforcement.
- **Matrix/joining:** qualified thermoset baseline versus a candidate thermoplastic/joining process.
- **Support span:** membrane panel span and node spacing.
- **Member geometry:** section area, second moment of area, effective length, curvature and member frequency.
- **Joint mass/stiffness:** rotational stiffness, manufacturing tolerance, eccentricity and defect classes.
- **Imperfections:** frozen measured or statistically bounded geometric deviations, missing/under-performing members, joint compliance and local seal defects.
- **Load histories:** proof cycles, subcritical repeated cycles, thermal cycling, and separately qualified impact/damage cases.

For each sample, the objective vector should include total structural-system mass, target payload allowance, buckling margin under imperfection-sensitive nonlinear analysis, peak and residual deformation, damage/leakage, fatigue/creep, inspectability, repairability and manufacturing burden. Keep hard safety gates separate from cost/performance objectives.

### Why these choices are evidence-based, but not yet proven for this vessel

- The 2026 kenemostat work models a geodesic frame, a bonded/stiffened sealing skin and a detached tension-membrane variant. It discusses both member and global buckling and identifies joint/manufacturing/dynamic-load limitations: [Aerospace Science and Technology, DOI 10.1016/j.ast.2026.112799](https://doi.org/10.1016/j.ast.2026.112799).
- Wang et al. analyze 3D-woven composite spherical shells under external hydrostatic pressure with multi-scale modelling; their results identify weave architecture and initial defects as consequential: [Journal of Industrial Textiles, DOI 10.1177/1528083720981999](https://doi.org/10.1177/1528083720981999).
- A review of thermoplastic-composite resistance welding explains routes for joining composite substructures and continuing issues with process control and NDT: [Materials 2024, DOI 10.3390/ma17194797](https://doi.org/10.3390/ma17194797).
- Aerospace self-healing-composite reviews show promising mechanisms for cracks and interlaminar damage but describe specific trigger, efficiency and implementation limitations; healing should not be equated with restored collapse resistance: [Materials 2022, DOI 10.3390/ma15238521](https://doi.org/10.3390/ma15238521).

## New first-pass improvement: bound the envelope's allowable areal mass

For a spherical pressure boundary with reference radius `R), the idealized outer area and enclosed volume are:

```text
A = 4πR²
V = (4/3)πR³
```

If all non-envelope hardware and the target payload are explicitly accounted for, the maximum mass left for the envelope is:

```text
m_envelope,max = (ρ_air − ρ_inside) V
                 − m_fixed_non_envelope − m_target_payload
```

The corresponding maximum mean envelope areal mass is:

```text
σ_envelope,max = m_envelope,max / A
```

For an ideal vacuum sphere with no frame, joints, propulsion, systems, or payload, this simplifies to `σ_max = ρ_air R / 3`. At `ρ_air = 1.225 kg/m³`, a 20 m radius gives only **8.17 kg/m²**, and a 40 m radius gives **16.33 kg/m²**, available for the *entire* envelope layer under those deliberately unattainable zero-other-mass assumptions. Every non-envelope kilogram and every target-payload kilogram tightens that ceiling. This is a strong early rejection screen for overweight concepts; it is not a material recommendation or pressure-strength estimate.

The API returns a **signed** ceiling. A negative envelope mass/areal-mass allowance means the fixed systems plus target payload already exceed ideal net lift even before the envelope is added. The calculation assumes that one radius represents both enclosed volume and pressure-boundary area; thickness, appendages, deformations, and topology-specific surface area are not resolved.

## Pressure evidence is a separate gate

A pressure case uses explicit exterior and interior absolute pressures. The screen computes only the differential load:

[
\Delta p=p_{\mathrm{outside}}-p_{\mathrm{inside}}
]

The new Rust API compares that load with an externally supplied, conservatively qualified allowable external differential pressure only when a capacity value and non-empty provenance references are provided. The references are carried for audit; the module cannot authenticate them or prove their independence.

Do not feed an ideal-geometry linear eigenvalue buckling pressure directly into this field as if it were a qualified allowable. The capacity input must be scoped to the exact geometry generation, material and laminate, joints, boundary conditions, imperfection envelope, and load case, with an independent check or test record. The API intentionally reports a *declared capacity ratio*, never an automatic “safe” boolean.

## Candidate metamaterial families

Compare at least three families against a conventional composite sandwich baseline:

- **Geodesic member frame + sealing skin:** triangulated frame with members sized for compression, joint mass, and global stability. Test both a skin bonded to the frame and a membrane restrained by frame edges.
- **Lattice-core sandwich shell:** inner/outer skins with an architected core. Track skin wrinkling, core shear/crushing, face/core debonding, and local/global buckling; do not assume a high unit-cell stiffness proves shell-level stability.
- **Graded hybrid lattice:** locally change cell topology, density, section, and orientation by load demand, with continuous manufacturable transitions and explicit defect tolerances. Compare against uniform lattices to establish whether grading earns its added joints and manufacturing complexity.

Initially parameterize topology/frequency, member section, skin thickness, core depth, joint geometry and mass, bracing layout, material/laminate, and cell-size gradient. Do not introduce speculative CNT material allowables. If high-performance fiber or nano-enabled material data becomes relevant, distinguish coupon measurements, laminate properties, and full-scale joint properties.

## Optimization objective and hard constraints

Use a Pareto result rather than one opaque score. Useful objectives include:

- maximize payload allowance and payload fraction;
- minimize complete dry mass and structural-material cost;
- minimize sensitivity to geometric defects, joint flexibility, and manufacturing variation;
- minimize feature count and manufacturing difficulty without hiding inspection/repair burden.

Every candidate must be rejected or left unresolved when any hard constraint is missing:

- member compression, local member buckling, and joint failure;
- global frame/shell buckling with geometric imperfections and boundary-condition sensitivity;
- skin failure, core shear/crushing, debonding, leak-path integrity and seal aging;
- fatigue, thermal cycling, gust/dynamic loads and damage tolerance;
- complete mass budget including joints, cables, adhesives, pumps, redundancy, landing gear and realistic minimum gauges;
- manufacturable geometry, inspectability, repairability and tolerance stack.

The optimizer may propose candidates and select the next analysis. It must not stand in for the structural solver or convert its own ranking into evidence of safety.

## Symthaea integration path

1. **Arithmetic baseline (this tranche).** Pure deterministic SI functions for sphere volume, explicit mass aggregation, ideal net buoyant mass, signed payload allowance, and evidence-bound pressure-capacity comparison.
2. **Independent known-answer checker.** Recompute reference cases separately and freeze expected values. Keep malformed, non-finite, negative-mass, missing-pressure-evidence, and missing-provenance negative controls.
3. **Candidate generator.** Parameterize only dimensions/material profiles with provenance. Use Symthaea to organize trade-offs, search candidates, and choose information-rich tests; do not treat HDC similarity as a constitutive material law.
4. **External geometry and FE bridge.** Reuse ENG-MESH #5332 and qualified solver bridges. Preserve geometry-to-mesh receipts, load/support/material-region identity, solver build/profile, convergence sequences, and exact quantity of interest. Compare independent analytical and numerical checks where possible.
5. **Uncertainty / adversarial campaign.** Vary member and joint mass, laminate allowables, imperfection amplitudes/shapes, dimensional tolerances, seal degradation, boundary conditions, and dynamic loads. Preserve correlated and common-mode uncertainties instead of assuming independence.
6. **Bench validation.** Manufacture representative coupons, joints, and shell subassemblies. Compare measured stiffness, failure mode, leakage, and collapse pressure with preregistered predictions before expanding scale.
7. **Scale gate.** A complete subscale shell must predict its own measured deformation and failure within a declared error envelope. Only then consider larger articles and eventually separately approved flight tests.

## Acceptance criteria for this first tranche

- The same explicit SI inputs produce deterministic outputs.
- Ideal vacuum and non-vacuum gas densities remain separate inputs.
- The mass breakdown includes envelope/seals, lattice/frame/joints, propulsion, energy storage, avionics/comms, and other systems.
- Negative payload allowance is preserved.
- NaN, infinity, negative masses, zero/negative dimensions, and derived overflow are rejected.
- External pressure is never treated as a structural capacity.
- Missing capacity or missing provenance cannot produce a capacity ratio.
- A ratio is labelled as an arithmetic comparison and carries the source references; it is not a pass/fail safety decision.
- No material allowables, safety factors, densities, or solver results are silently defaulted.
- Independent checker and physical-test gates remain outstanding until evidence exists.

## Next refinement: deterministic uncertainty bounds and geometry alternatives

### Propagate stated uncertainty instead of hiding it in a single assumed mass

The Rust screen now exposes screen_spherical_payload_bounds. For a fixed radius, the caller supplies lower and upper bounds for:

- ambient-air density;
- internal-gas density;
- complete structural-system areal mass; and
- fixed non-structural vehicle mass.

The lower payload allowance uses the lowest ambient density, highest internal density, highest structural areal mass, and highest fixed mass. The upper allowance uses the opposite endpoints. These are deterministic outer bounds for all independent combinations inside those intervals, not confidence intervals and not probability-weighted forecasts.

The result also reports the payload-span contribution from each input interval separately: ambient density, internal-gas density, structural-system areal mass, and fixed vehicle mass. For a fixed radius, these contributions add to the total interval width because the screening equation is affine in those inputs. This creates a useful experiment-prioritization signal: the widest contribution identifies which input bound currently dominates the arithmetic uncertainty. It does not establish which physical failure mechanism dominates, and it is only as defensible as the supplied bounds.

For a reference sphere with radius 20 m, one deliberately hypothetical example uses air density 1.20–1.25 kg/m³, internal density 0–0.01 kg/m³, structural-system areal mass 4–5 kg/m², and fixed non-structural mass 14–18 tonnes. The resulting payload mass interval is approximately **−3.26 to +7.78 tonnes**. This is an example of the method, not a prediction: those mass bounds have not been sourced or validated, and the assumed structure has no demonstrated external-pressure capacity.

For this illustrative set, the interval width decomposes into approximately **1.68 tonnes** from ambient-air density, **0.34 tonnes** from internal-gas density, **5.03 tonnes** from structural-system areal mass, and **4.00 tonnes** from fixed vehicle mass. Within these assumed ranges, structural-system areal mass is the largest contributor, followed by fixed-system mass. That points to a concrete first data task: construct and bound a component-complete structural/system mass ledger. It does not indicate which failure mode controls pressure capacity.

The resulting MassBudgetStatus is one of:

- RobustlyNonNegative: all input combinations in the declared bounds give non-negative payload allowance.
- BoundSensitive: the interval crosses zero, so the mass conclusion changes with the assumptions.
- NegativeThroughout: even the optimistic bound is negative.

This status only describes the mass equation. RobustlyNonNegative does **not** mean structurally feasible. The inputs must cite the actual origin and applicability of their bounds before results should be used for engineering decisions. The current interval holds radius fixed; radius tolerance, geometry variation, covariance between parameters, manufacturing variability, and operating-condition changes need separate treatment rather than silently assuming they are covered.

### Bound pressure demand and declared capacity separately

The Rust screen now exposes `assess_external_pressure_bounds`. It accepts explicit lower/upper bounds for exterior and interior absolute pressures, plus an optional declared allowable-capacity interval and evidence identifiers. It first propagates the pressure bounds into a positive external-differential demand interval:

```text
signed Δp ∈ [p_external,min − p_internal,max,
             p_external,max − p_internal,min]
demand = max(signed Δp, 0)
```

If every possible case has no positive external differential, it reports `NoExternalPressureDemand`. Any supplied capacity interval is still validated for finite, ordered, positive endpoints even in this branch, so malformed declarations cannot be silently ignored. If demand can be positive but no capacity is declared, capacity remains unknown. If capacity metadata is incomplete, comparison is blocked. Otherwise, the result distinguishes three arithmetic relationships: the minimum declared capacity covers the maximum declared demand; the intervals overlap; or even the maximum declared capacity is below the minimum positive demand.

For example, a hypothetical demand bound of 98–102 kPa and a hypothetical declared allowable-capacity bound of 105–120 kPa yields `DeclaredCapacityCoversDemandBounds`. That means only the *declared endpoint intervals* are ordered. It does not mean the evidence is genuine, the capacity is qualified for the exact design, or the shell is safe. The code cannot authenticate the referenced solver/test records, inspect their scopes, verify independence, establish imperfection coverage, or account for damage and dynamic effects.

Use demand bounds tied to an explicit operating envelope (altitude, weather, internal pressure and transients) and capacity bounds from geometry-specific, independently reviewed nonlinear analysis and relevant physical tests. A linear eigenvalue or perfect-geometry result must not be presented as a qualified allowable without an applicable and justified methodology.

### Translate the global resultant into explicit load-path demand

The Rust API now includes `screen_load_path_demand`, a generic arithmetic bridge from the spherical great-circle compressive resultant to a declared effective load-path count. It reports an equal-share reference load and a second, factor-scaled value using an explicitly supplied peak-to-mean concentration factor. Neither value is a member capacity or a validated maximum.

This separation is intentional: load-path count and load concentration vary with topology, joint stiffness, boundary conditions and spatial imperfections. The function does not infer either one from radius or from an architecture label; the caller must provide a justified count and concentration factor, and the output cannot be used as a substitute for a structural model.

For context, Toli's 2026 kenemostat paper gives the approximate effective great-circle crossing count (n_{eff}=30f/\pi), yielding the mean load estimate (F=p\pi^2R^2/(30f)). For its modeled 20 m circumradius, frequency-3 case at 101.3 kPa, the paper reports a 4.44 MN analytical estimate and 4.82 MN mean from its beam-element FEA (about 8.6% higher), plus a peak-to-mean ratio of 1.32. Applying 1.32 to the analytical estimate gives about 5.87 MN, while applying it to the reported FEA mean gives about 6.36 MN. Those are values for that paper's idealized model—not transferable factors, qualified allowables, or predictions for this repository's candidates. Source: [Toli, *Aerospace Science and Technology* 177(C), 112799 (2026)](https://doi.org/10.1016/j.ast.2026.112799).

Use this function as a transparent handoff to a geometry-specific member/joint analysis. For actual design ranking, record the load-path derivation, full spatial imperfection set, as-built configuration, and force distribution from the qualified model; compare the estimate against a physical subassembly before raising confidence in the model.

### Require exact scope alignment before comparing capacity evidence

The bounded pressure screen now has a scope-aware wrapper, `assess_scoped_external_pressure_bounds`. A pressure-capacity record is compared only when five immutable scope identifiers match the requested case:

- geometry revision (prefer an exact mesh/CAD artifact ID or content hash);
- material / laminate model revision;
- load-case definition;
- boundary-condition and support profile; and
- imperfection basis (measured geometry, a documented envelope, or a justified imperfection set).

This addresses a gap in a bare provenance-ID check: a pressure-capacity record might exist and have non-empty metadata while describing a different geometry, laminate, load case, support condition or perfect-geometry model. A matching string is not authentication, but an explicit mismatch is enough reason to block the comparison. Missing or blank scope values return `ScopeIncomplete`; unequal fields return `ScopeMismatch`. No capacity stays unknown, and a no-demand pressure envelope does not require an irrelevant capacity-scope record.

The identifiers should point to immutable manifests, not hand-written labels. A future evidence manifest should bind exact hashes for geometry, mesh, material/batch data, solver version and settings, convergence sequence, imperfection fields, boundary conditions, load profile, output quantity, reviewer/test receipt, and the decision record. If any scoped artifact changes, the capacity result should be treated as a different evidence item until its applicability is re-established.

This is deliberately an applicability gate, not an evidence authenticator. The Rust module cannot establish that referenced objects exist, that hashes correspond to the claimed content, that the reviewer was independent, or that a numerical allowable reflects real failure. Those requirements remain external provenance-verification and physical-validation gates.

### Treat imperfection geometry as a structured input, not one scalar penalty

A September 2026 preprint by Yang and Schneider studies axially compressed cylindrical shells using nonlinear continuation and bifurcation analysis. Even one localized Gaussian defect can change which instability triggers first; the predicted buckling threshold varies non-monotonically and can change discontinuously as defect depth or size changes. The authors' key implication is that imperfection severity cannot be reliably ranked by a single defect-amplitude number alone: [arXiv:2609.19436](https://arxiv.org/abs/2609.19436).

This is evidence about cylindrical shells, not direct validation for a spherical geodesic structure. The research lesson to carry forward is narrower and still valuable: for every candidate imperfection basis, preserve the *spatial field and mode*, not only RMS deviation or maximum dent depth. Include location relative to nodes/joints, defect width/shape, combinations of defects, measured as-built deviations, and the mapping from scan to analysis mesh. Use nonlinear path-following capable of detecting relevant asymmetric modes rather than assuming one symmetric eigenmode necessarily governs.

The next test matrix should therefore include geometrically distinct imperfections with similar maximum amplitude and RMS magnitude. That tests whether shape and location materially change predicted failure, and helps guard against an optimizer learning that an arbitrary scalar defect score is a sufficient safety proxy. Any localized probing or subcritical stability-landscape experiments require an independently reviewed procedure: probing can itself change equilibrium paths and trigger snap-through in sensitive structures.



### Expand geometry search beyond a smooth sphere—but keep the loads honest

Compare the spherical reference against at least two distinct families:

1. **Triangulated geodesic/polyhedral shells.** Member compression and joint eccentricity may govern; use measured node stiffness and explicit imperfections.
2. **Stellated or deliberately faceted shells.** Some shapes may reduce sensitivity to imperfections compared with a smooth sphere, but facet bending, edge compression, local panel buckling, joinery, and volume penalties become central.
3. **Lattice-core sandwich shells.** Track facesheet wrinkling, core shear/crush, debonding, and global shell instability as separate failure modes.
4. **Conventional pressure-capable buoyant baseline.** Compare a helium envelope or another validated buoyant architecture using the same cargo, range, reserve, and operating constraints. This baseline prevents the vacuum concept from benefiting from asymmetric assumptions.

An important research lead is Ning and Pellegrino's study of externally pressurized geodesic and stellated shells, which reports reduced imperfection sensitivity for some examined configurations: [DOI 10.1016/j.jmps.2018.06.008](https://doi.org/10.1016/j.jmps.2018.06.008). The paper's reported gains are specific to its idealized configurations and analysis assumptions; treat them as a geometry-screening hypothesis, not a transferable knockdown factor or certified design value. That supports studying geometry as a design variable, not transferring its result to an untested airship. Likewise, NASA's shell-buckling work emphasizes that geometric/loading imperfections can materially reduce capacity and that knockdown recommendations are geometry- and scope-specific: [NASA NTRS 2016 study of composite sandwich shell buckling](https://ntrs.nasa.gov/citations/20160012029).

### The next evidence gate

For each candidate, retain a complete and versioned evidence bundle: exact CAD/mesh generation, as-built geometry, material/layup and batch data, joint/edge stiffness, defect bounds, load/support definition, solver and convergence record, nonlinear imperfection analysis, and test receipt. Then compare blinded preregistered predictions against instrumented subassembly tests. Do not let a candidate optimizer use a predicted buckling load as a qualified capacity; only an independently reviewed, case-specific analysis and relevant tests can support that conclusion.

The screening code should remain the quick-rejection layer. Candidate search may organize these alternatives and select high-information experiments, but it must not generate material allowables, infer shell strength from HDC similarity, or turn a recovery score into a safety claim.

## Known exclusions

This first tranche does not perform topology optimization, constitutive modelling, composite failure analysis, finite-element analysis, member-load distribution from geometry, uncertainty quantification beyond the declared scalar intervals, leak-rate prediction, atmospheric flight dynamics, active control, regulatory certification, or airworthiness determination. Positive arithmetic payload allowance alone is not evidence of a viable vehicle.
