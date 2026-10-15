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

## Known exclusions

This first tranche does not perform topology optimization, constitutive modelling, composite failure analysis, finite-element analysis, uncertainty quantification, leak-rate prediction, atmospheric flight dynamics, active control, regulatory certification, or airworthiness determination. Positive arithmetic payload allowance alone is not evidence of a viable vehicle.
