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

### Propagate uncertainty through load-path demand, not just one assumed estimate

The Rust API now includes `screen_bounded_load_path_demand`. It accepts independent bounds for the total compressive resultant, effective number of paths, and peak-to-mean concentration factor. It calculates bounded equal-share and factor-scaled demands using the monotonic endpoint rules:

```text
mean_min = resultant_min / path_count_max
mean_max = resultant_max / path_count_min

scaled_min = mean_min × concentration_min
scaled_max = mean_max × concentration_max
```

A hypothetical 127.33 MN global resultant with an effective path-count range of 25–35 and a factor range of 1.2–1.5 yields a factor-scaled arithmetic interval of roughly **4.37–7.64 MN per effective path**. Those path-count and factor bounds are placeholders for demonstrating interval propagation, not model-derived or experimentally justified inputs. The resulting interval is only as defensible as the assumptions. It is not a predicted force on any particular member, not a structural allowable, and not a shell safety result.

The bound propagation assumes independently declared intervals and monotonic use of the endpoints. If path count and concentration factor are correlated because the geometry or defect field changes both together, independent endpoint combinations may be overly broad or misleading as a design-specific envelope. In that case, use a joint parametric study over geometry and imperfections and retain each scenario's provenance rather than treating marginals as independent. The current API rejects non-finite or reversed endpoints, non-positive minimum path count, concentration factors below one, and overflowed derived bounds.

This fills an arithmetic gap but not the physics gap: actual path count, load concentration, local member forces and global buckling modes still need geometry-specific nonlinear analysis with imperfections and comparison to physical tests. For the geodesic 2026 reference, the paper's approximate analytical load-sharing relationship and its FEA comparison are useful regression targets for an independently implemented checker, but they must remain tied to that paper's geometry and assumptions. [Toli, DOI 10.1016/j.ast.2026.112799](https://doi.org/10.1016/j.ast.2026.112799).

### Preserve dependence with coherent load-path scenarios

The Rust screen now also exposes `screen_load_path_scenario_set`. Use this when the effective path count and peak-to-mean factor change together with geometry, joint stiffness, boundary conditions, or the spatial imperfection field. Each `LoadPathDemandScenario` keeps one resultant, one effective path count, one concentration factor, a unique scenario ID, and an evidence-manifest ID together. The result retains each row and reports the minimum and maximum scaled demand **among the enumerated cases**, including which scenario produced each extreme.

This complements rather than replaces `screen_bounded_load_path_demand`:

- The independent interval API combines declared marginal endpoints. It is useful as a deliberately broad screening envelope when those endpoint combinations are intended to be allowed.
- The coherent-scenario API evaluates each supplied tuple as a unit. It avoids inventing a combination such as the largest resultant, smallest path count, and largest concentration factor when those values came from mutually exclusive geometries or imperfection cases.

A small synthetic regression example illustrates the distinction: scenario A uses 1,000 N / 10 effective paths / 1.1 concentration, giving 110 N; scenario B uses 2,000 N / 40 paths / 1.3 concentration, giving 65 N. The enumerated scenario envelope is 65–110 N. Combining the separate marginal extrema would produce 2,000 / 10 × 1.3 = 260 N, which is not the demand of either supplied scenario. These values are only an arithmetic test fixture, not an aircraft estimate.

The scenario envelope is **not** a statistical confidence interval, probability estimate, or guarantee over unsupplied cases. It is no more complete than the submitted scenario matrix. Scenario and evidence IDs must be non-empty and scenario IDs unique, but string matching does not verify the underlying model, hash, reviewer, or test receipt. A practical next step is to generate these rows from versioned geometry/imperfection/load manifests and retain full spatial defect fields, solver settings, and independent test evidence for each row. The test matrix should include shapes/locations with similar scalar defect amplitudes as well as measured as-built scans.

### Paradigm shift: optimize the evidence-backed design, not one idealized payload number

A useful shift is to stop treating the task as “find the lightest shell that survives a nominal pressure case.” Instead, search for a **robust, inspectable, damage-tolerant system whose performance claims are traceable to evidence**. Payload is only one objective; local and global buckling, imperfection sensitivity, joint and seal complexity, manufacturing tolerance, leak detection, repairability, inspectability, failure containment, and the test cost needed to reduce uncertainty can change which architecture is actually preferable.

A March 2026 study on variable-stiffness composite cylinders demonstrates a contemporary workflow that combines measured imperfections, sampling, surrogate models, and multi-objective optimisation across mass, buckling capacity and robustness: [Computers & Structures, DOI 10.1016/j.compstruc.2026.108141](https://doi.org/10.1016/j.compstruc.2026.108141). It is a cylinder study, not evidence for a spherical geodesic vehicle; transfer the workflow concept, not its values or model. The relevant research direction is a **Pareto set** rather than one scalar winner: explicitly show trade-offs and reject designs that only win under a brittle ideal-geometry assumption.

To enforce the evidence side of that shift, the Rust screen now includes `assess_candidate_evidence_gates`. It accepts a declared profile and lists blockers across eight classes: payload mass-budget status, pressure-capacity scope, nonlinear imperfection analysis, independent structural review, representative-shell validation, damage-tolerance testing, operating-envelope review, and a content-addressed evidence manifest. The gates are conjunctive: strong payload arithmetic cannot cancel an absent pressure-capacity record, a scope mismatch, or missing physical validation. A bound-sensitive or negative-throughout mass budget cannot be promoted merely because other declared evidence states look mature.

The top bookkeeping state is `EvidenceBundleReadyForIndependentAssessment`, deliberately **not** “feasible,” “safe,” “flight-ready,” or “certified.” Each maturity value is caller-declared. The function does not authenticate the reviewer, inspect manifest bytes, verify hashes, qualify solver settings, or validate test results. Even a content-addressed manifest needs an external verifier to confirm that the bytes and claims match.

This is an assurance-case pattern: preserve each critical claim and its evidence separately, expose explicit blockers, and avoid compensatory scoring across non-substitutable requirements. It also creates a better Symthaea optimisation boundary. Symthaea can rank candidate experiments by expected information gain and explore Pareto trade-offs, but its search score must not override a hard evidence gate or infer physical capacity from HDC similarity. A future robust optimiser should return candidate designs together with their uncertainty envelope, worst-case scenario identifiers, open evidence obligations and next best discriminating test—not just a high score.

### Scale-up must model defect populations, not only geometric similarity

The 2026 kenemostat study is an important new structural result: it treats member and global buckling in an idealized geodesic vacuum-buoyant frame and reports analytical/beam-FEA agreement for its specific model. It also explicitly leaves joint design, manufacturing constraints and dynamic loading as limitations. Its claimed scale law and critical radius should therefore be treated as a hypothesis to stress-test against measured joint mass/stiffness, minimum-gauge skin, assembly tolerances, common-cause manufacturing drift, and a growing number of manufactured features—not as a flight-readiness verdict: [Toli, DOI 10.1016/j.ast.2026.112799](https://doi.org/10.1016/j.ast.2026.112799).

Derveni et al. found, in validated finite-element studies of pressurized hemispherical shells with distributed imperfections, that the most severe defect in the modeled population governed buckling onset. This is a critical scale-up warning: if the number of joints, members or inspected surface regions grows, the tail of the defect population may matter more than the average defect. Their results concern the studied thin shell and do not directly establish the governing defect for a geodesic lattice, so that transfer itself must be tested: [arXiv:2410.08973](https://arxiv.org/abs/2410.08973).

The code now exposes `screen_feature_defect_population(feature_count, per_feature_exceedance_probability)`, which calculates the conditional identical-independent-feature expression

\[
P(\text{at least one exceedance})=1-(1-p)^N.
\]

This is a sensitivity tool for a *declared defect criterion*, not a structural failure probability. For example, supplying (p=10^{-4}) gives about 2.664% probability of at least one exceedance across 270 features and about 63.214% across 10,000. These values are illustrative arithmetic only; neither (p) nor the independence assumption is measured for this vehicle. The API deliberately does not convert the result into a safety factor. Real manufacturing populations can be heterogeneous or correlated, inspection misses defects, and one defect does not necessarily cause system failure. Candidate analysis should use representative measurements, explicit correlation/common-cause models, and the topology's actual redistribution/failure mechanics.

### From inspection metrics to residual escaped-defect risk

The Rust API now exposes `screen_inspection_escape_population(feature_count, per_feature_critical_defect_probability, conditional_probability_of_detection)`. It computes the per-feature escape probability as defect probability multiplied by one minus the conditional probability of detection, then propagates that declared probability across the feature population using an identical-independent-feature model. The result reports expected undetected critical-defect count as well as the probability that at least one exceeds inspection. The companion API `screen_bounded_inspection_escape_population` accepts lower/upper ranges for both defect prevalence and probability of detection, choosing the optimistic endpoints for its lower result and the adverse endpoints for its upper result.

For a hypothetical critical-defect prevalence of 0.01% per feature and a 90% probability of detection, the model gives an at-least-one-escape probability of about **0.270% at 270 features** and **9.516% at 10,000 features**. Raising detection probability reduces the conditional escape estimate. These values are arithmetic examples, not measured defect prevalence, inspection performance, or structural failure probabilities. They assume independent and identical features and independent inspection outcomes; spatial/common-cause defects, correlated misses, missed-to-found inspection transfer, criticality variation, and damage redistribution are not represented. The model must not be used for a certification or airworthiness claim.

For example, uncertainty bounds of 0.1–1% for critical-defect prevalence and 80–95% for conditional detection, across 1,000 features, propagate to an illustrative 0.05–2 expected escapes and about 4.877–86.494% probability of at least one escape. These are endpoints from the supplied intervals and the assumed model, not confidence bounds unless a separate statistical analysis justifies the intervals. The width itself is useful: it shows that weak evidence on defect prevalence or inspection performance can make a point-estimate risk figure misleading.

A May 2026 in-situ L-PBF gyroid-lattice study demonstrates calibrated segmentation and directional deviation metrics, reporting successful detection of the controlled deviations studied with a false-alarm rate below 2% in its case study: [Grasso et al., DOI 10.1016/j.addma.2026.105228](https://doi.org/10.1016/j.addma.2026.105228). The reported false-alarm rate is not a probability-of-detection curve: the latter requires representative flawed and unflawed samples, severity/location coverage and a declared threshold. The Rust API now includes `screen_inspection_qualification_counts`, which takes counts of detected defects, defects present but missed, false positives, and clean trials, then returns observed probability of detection and false-alarm rate with Wilson score intervals using an explicit caller-supplied z-score. Keeping both rates matters: false-alarm rate measures how often a clean trial is flagged, whereas probability of detection measures how often a target defect is found. A low false-alarm rate alone cannot demonstrate adequate detection. The intervals quantify finite-sample uncertainty under a binomial sampling model; they do not correct biased sampling, dependent trials, or a qualification dataset that does not represent the target geometry/process.

A small synthetic example with 18 detections in 20 trials containing a target defect and 2 false positives in 50 clean trials has observed POD 90% and false-alarm rate 4%. At z≈1.96, the respective Wilson intervals are roughly 69.9–97.2% and 1.10–13.46%. The wide POD interval makes the sample-size problem visible instead of letting the point estimate appear authoritative. Those sample counts are illustrative—not a recommended sample plan or an acceptance criterion.

The PR therefore keeps defect prevalence and inspection POD as separate caller-supplied quantities, and separately gates inspection validation. A future qualification dataset should be stratified by feature size/orientation, critical defect class, defect severity, position, manufacturing batch, imaging conditions, segmentation settings and independent confirmatory metrology.

This suggests a paradigm shift toward an as-built structural digital thread: use manufacturing inspection to create a versioned geometry/defect record, map that record to the analysis model, update the candidate's evidence package, and identify which measured defect or uncertainty dominates the next test. NASA's current challenge problem for certification by analysis likewise emphasizes simulation credibility, verification/validation, uncertainty quantification and evidence sufficient to support a particular regulatory requirement—not a generic claim that a model is accurate: [NASA NTRS 20250005704](https://ntrs.nasa.gov/citations/20250005704). For this vehicle, model credibility must be claimed only for the specific geometry, loads, material/production process and output quantity actually validated.

The candidate evidence gate now also requires (a) a complete mass ledger reconciled to a measured or build estimate, (b) scale transfer validated over a relevant size range, (c) defect-tail characterization for the target manufacturing population, and (d) demonstrated inspection detection performance on representative critical features. Analytical similarity alone and one small specimen are insufficient. This last gate is motivated by the difficulty of detecting defects in lattice features such as thin struts, nodal overlaps and open porosity; a 2026 review of non-destructive evaluation for additively manufactured lattices discusses geometry-specific detectability limits and the value of combining inspection methods with machine learning: [DOI 10.1016/j.jallcom.2026.186865](https://doi.org/10.1016/j.jallcom.2026.186865). That review is relevant to additive manufacturing, not proof that one inspection method works for every lattice/process. These are caller-declared maturity states, not authenticated evidence; the gate reports missing obligations and never promotes a design to “safe.”

### Require exact scope alignment before comparing capacity evidence

The bounded pressure screen now has a scope-aware wrapper, `assess_scoped_external_pressure_bounds`. A pressure-capacity record is compared only when eight immutable scope identifiers match the requested case:

- geometry revision (prefer an exact CAD/as-built scan ID or content hash);
- mesh revision or physical specimen/build revision;
- material / laminate model or tested material/batch revision;
- exact analysis solver/settings or physical test method/instrumentation procedure;
- load-case definition;
- boundary-condition and support profile;
- imperfection basis (spatial field, mode family, and bounded defect set); and
- capacity basis (the output quantity and method used to derive the allowable, including the distinction between an eigenvalue, a nonlinear limit load, and a test-derived allowable).

This addresses a gap in a bare provenance-ID check: a pressure-capacity record might exist and have non-empty metadata while describing a different geometry, mesh/specimen, laminate, analysis/test method, load case, support condition, imperfection field, or capacity derivation. A matching string is not authentication, but an explicit mismatch is enough reason to block the comparison. Missing or blank scope values return `ScopeIncomplete`; unequal fields return `ScopeMismatch`. No capacity stays unknown, and a no-demand pressure envelope does not require an irrelevant capacity-scope record.

The identifiers should point to immutable manifests, not hand-written labels. A future evidence manifest should bind exact hashes for geometry, mesh, material/batch data, solver version and settings, convergence sequence, imperfection fields, boundary conditions, load profile, output quantity, reviewer/test receipt, and the decision record. If any scoped artifact changes, the capacity result should be treated as a different evidence item until its applicability is re-established.

This is deliberately an applicability gate, not an evidence authenticator. The Rust module cannot establish that referenced objects exist, that hashes correspond to the claimed content, that the reviewer was independent, or that a numerical allowable reflects real failure. Those requirements remain external provenance-verification and physical-validation gates.

### Treat imperfection geometry as a structured input, not one scalar penalty

A September 2026 preprint by Yang and Schneider studies axially compressed cylindrical shells using nonlinear continuation and bifurcation analysis. Even one localized Gaussian defect can change which instability triggers first; the predicted buckling threshold varies non-monotonically and can change discontinuously as defect depth or size changes. The authors' key implication is that imperfection severity cannot be reliably ranked by a single defect-amplitude number alone: [arXiv:2609.19436](https://arxiv.org/abs/2609.19436).

This is evidence about cylindrical shells, not direct validation for a spherical geodesic structure. The research lesson to carry forward is narrower and still valuable: for every candidate imperfection basis, preserve the *spatial field and mode*, not only RMS deviation or maximum dent depth. Include location relative to nodes/joints, defect width/shape, combinations of defects, measured as-built deviations, and the mapping from scan to analysis mesh. Use nonlinear path-following capable of detecting relevant asymmetric modes rather than assuming one symmetric eigenmode necessarily governs.

The next test matrix should therefore include geometrically distinct imperfections with similar maximum amplitude and RMS magnitude. That tests whether shape and location materially change predicted failure, and helps guard against an optimizer learning that an arbitrary scalar defect score is a sufficient safety proxy. Any localized probing or subcritical stability-landscape experiments require an independently reviewed procedure: probing can itself change equilibrium paths and trigger snap-through in sensitive structures.



### Recent validation-method lead: multi-mode imperfections and a physical dome test

A paper published online on 17 August 2026 reports nonlinear imperfection studies and a full-scale buckling test for an ellipsoidal launch-vehicle propellant-tank dome assembly. In that specific case, a model combining the first through third buckling modes predicted the experimental buckling load within 0.5%: [doi:10.1016/j.ast.2026.113564](https://doi.org/10.1016/j.ast.2026.113564). This is useful as a workflow lead—measure/represent manufacturing defects, compare single-mode and multi-mode imperfections, predict before testing, and publish the comparison—not as a transferable knockdown factor or validation for a vacuum-airship geodesic lattice. The specimen geometry, material, joints, boundary conditions, and loading must remain case-specific.

A separate study on spherical domes under external pressure describes a numerical/analytical method checked against a literature set of more than 800 experimental buckling results, reinforcing that imperfection-aware design methods need a clearly defined calibration population and applicability domain: [doi:10.1016/j.ijpvp.2019.104015](https://doi.org/10.1016/j.ijpvp.2019.104015). Neither study removes the need for candidate-specific subassembly and shell tests. Their value here is procedural: keep the full imperfection basis, prediction, test article, and error metric linked in each evidence record.

### Expand geometry search beyond a smooth sphere—but keep the loads honest

Compare the spherical reference against at least two distinct families:

1. **Triangulated geodesic/polyhedral shells.** Member compression and joint eccentricity may govern; use measured node stiffness and explicit imperfections.
2. **Stellated or deliberately faceted shells.** Some shapes may reduce sensitivity to imperfections compared with a smooth sphere, but facet bending, edge compression, local panel buckling, joinery, and volume penalties become central.
3. **Lattice-core sandwich shells.** Track facesheet wrinkling, core shear/crush, debonding, and global shell instability as separate failure modes.
4. **Conventional pressure-capable buoyant baseline.** Compare a helium envelope or another validated buoyant architecture using the same cargo, range, reserve, and operating constraints. This baseline prevents the vacuum concept from benefiting from asymmetric assumptions.

An important research lead is Ning and Pellegrino's study of externally pressurized geodesic and stellated shells, which reports reduced imperfection sensitivity for some examined configurations: [DOI 10.1016/j.jmps.2018.06.008](https://doi.org/10.1016/j.jmps.2018.06.008). The paper's reported gains are specific to its idealized configurations and analysis assumptions; treat them as a geometry-screening hypothesis, not a transferable knockdown factor or certified design value. That supports studying geometry as a design variable, not transferring its result to an untested airship. Likewise, NASA's shell-buckling work emphasizes that geometric/loading imperfections can materially reduce capacity and that knockdown recommendations are geometry- and scope-specific: [NASA NTRS 2016 study of composite sandwich shell buckling](https://ntrs.nasa.gov/citations/20160012029).

### The next evidence gate

For each candidate, retain a complete and versioned evidence bundle: exact CAD/mesh generation, as-built geometry, material/layup and batch data, joint/edge stiffness, defect bounds, load/support definition, solver and convergence record, nonlinear imperfection analysis, and test receipt. Then compare blinded preregistered predictions against instrumented subassembly tests. Do not let a candidate optimizer use a predicted buckling load as a qualified capacity; only an independently reviewed, case-specific analysis and relevant tests can support that conclusion.

The screening code should remain the quick-rejection layer. Candidate search may organize these alternatives and select high-information experiments, but it must not generate material allowables, infer shell strength from HDC similarity, infer a real member's load from an unvalidated path count, or turn a recovery score into a safety claim.


### Robust design is not just a conservative scalar factor

Probabilistic buckling work and experimentally grounded knockdown approaches show why calibration domain matters. NASA's probabilistic composite-shell study uses specimen geometry, measured imperfections and material data to estimate buckling-load distributions for a selected cylinder population: [NASA NTRS 2003-0062248](https://ntrs.nasa.gov/citations/20030062248). A separate deterministic method for spherical shells under external pressure reports a validation set of roughly 600 experiments: [Wagner, Hühne and Niemann, DOI 10.1016/j.ijmecsci.2018.03.029](https://doi.org/10.1016/j.ijmecsci.2018.03.029). These are valuable examples of calibration and validation discipline, not universal factors for the proposed lattice shell. The candidate's geometry, constituent system, joints, pressure boundary and defect population must fall within a demonstrated applicability domain—or require new validation.


### Cross-check the positive concept against its own mass assumptions

The NASA/MIT discrete-lattice vacuum-airship paper is an important architectural precedent, but its own theoretical development makes simplifying assumptions that a production-scale case must revisit: idealized isotropic lattice behavior, selected scaling exponents, standard sea-level operating conditions, and a detailed design phase still to follow for imperfection/safety-factor treatment. The Metlen–Palazotto thesis independently reports a predicted structure-to-buoyancy ratio (W/B=0.94) for its geodesic-frame/membrane concept—positive only before the complete vehicle-level payload, propulsion, energy, joints and operational reserve are fully reconciled under a given mission definition. These studies should neither be dismissed nor promoted to proof of a practical aircraft: they provide candidates and scaling hypotheses to re-run with buildable components and a reconciled system mass ledger. Sources: [NASA NTRS 20190001133](https://ntrs.nasa.gov/citations/20190001133), [Metlen, 2013 thesis](https://scholar.afit.edu/etd/838/).

### What a credible paradigm shift would look like in practice

Treat each design as a falsifiable, versioned claim bundle: (1) one mass ledger reconciled against component estimates and measured build mass, (2) exact CAD/as-built and mesh/specimen revisions, (3) representative distributions of defects and inspection sensitivity, (4) nonlinear global and local failure analyses, (5) pre-registered test predictions with explicit error metrics, and (6) demonstrated damage tolerance inside a declared operating envelope. Let Symthaea propose candidate geometries and rank the next tests by how sharply they discriminate between competing failure hypotheses; do not let it replace the solver or decide an allowable. Compare vacuum designs to helium and other realistic baselines using identical service requirements. The first outcome can legitimately be a no-go result if reconciled mass, defect tails or scale transfer prevent a credible positive margin. Falsification and early stopping are part of an efficient discovery system, not a failure of the research.

## Known exclusions

This first tranche does not perform topology optimization, constitutive modelling, composite failure analysis, finite-element analysis, member-load distribution from geometry, uncertainty quantification beyond the declared scalar intervals, leak-rate prediction, atmospheric flight dynamics, active control, regulatory certification, or airworthiness determination. Positive arithmetic payload allowance alone is not evidence of a viable vehicle.
