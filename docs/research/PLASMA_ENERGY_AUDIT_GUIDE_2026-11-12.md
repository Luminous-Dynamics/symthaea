# Plasma Energy Audit — Offline CLI and fixture guide

**Implementation:** [PLASMA-ENERGY-AUDIT-001 #11751](https://github.com/Luminous-Dynamics/symthaea/issues/11751)  
**Parent:** [PLASMA-ENERGY-001 #11718](https://github.com/Luminous-Dynamics/symthaea/issues/11718)  
**Date:** 2026-11-12  
**Claim ceiling:** source metadata and declared synthetic accounting only; no solver, thermal model, physical result or technology superiority.

## 1. Command

From the Symthaea workspace root:

```sh
cargo run -p xtask -- plasma-energy-audit
```

Emit structured JSON:

```sh
cargo run -p xtask -- plasma-energy-audit --json
```

Paths may be overridden:

```sh
cargo run -p xtask -- plasma-energy-audit \
  --plasma-sources data/research/plasma_energy_reference_cases_v0_1.json \
  --hts-sources data/research/hts_fusion_magnet_reference_cases_v0_1.json \
  --thermal-fixtures data/research/thermal_store_accounting_fixtures_v0_1.json
```

The command is offline and uses existing `xtask` dependencies. It does not invoke WarpX, Zapdos, PROCESS, NekRS, SAM, THEA, a plasma power supply or a thermal solver.

## 2. Source-register audit

The source audit requires:
- an HTTP(S) source URL with a non-empty authority (scheme-only values such as `https://` are rejected);
- a non-empty programme and claim ceiling;
- declared, unique evidence classes;
- at least one uniquely identified source case and negative control;
- every case's evidence class to appear in the declared set;
- a source URL using HTTP(S);
- an explicit `luminous_reproduction=false` field on every case;
- an expected disposition for each negative control;
- every `FUSION_DIRECT_ENERGY_CONVERSION` and `FUSION_FUEL_ENERGY_PARTITION` case has `particle_energy_origin`, `fusion_born_output_demonstrated=false`, and `comparison_status`.

This is metadata integrity only. The DEC-specific gate records that fusion-born electrical output was not demonstrated/attributed in the source case. This does not assert that a physical fusion-born contribution was exactly zero; unknown contributions remain unquantified. It prevents externally heated plasma recovery or a laboratory fundamental-unit experiment being silently promoted to demonstrated fusion-born energy recovery. It does not independently check paper contents, reproduce reported values, or establish that a source supports every interpretation in its case description.

Current frozen input:
- plasma-energy register: 29 cases / 29 controls, including five direct-energy-conversion cases, one D–³He/DT fuel-mixture model, five miniature/modular low-temperature plasma device references, and thermal-processing/ETES comparison references;
- HTS fusion magnet register: 6 cases / 12 controls.

## 3. Synthetic thermal-store fixture

The fixture is `data/research/thermal_store_accounting_fixtures_v0_1.json`. It uses integer joules and an explicitly synthetic evidence class.

Each scenario has:
- a unique scenario ID and technology label;
- a functional heat-service boundary (service identity/type, outlet temperature, optional pressure/flow, delivery duration, trace digest);
- an exact wall-plug energy-boundary contract identity, including which conversion and auxiliary systems are inside the meter perimeter;
- a metering mode;
- wall-plug meter inputs with unique input and meter-boundary identities;
- explicit primary-process and auxiliary meter roles, or one exclusive total meter;
- stored-energy increase, charge conversion losses, auxiliary dissipation;
- energy after the storage hold and hold losses;
- stored energy withdrawn, useful delivered heat and discharge losses;
- expected disposition.

The wall-plug boundary contract ID is mandatory and participates in comparability. Two scenarios can each close their own accounting yet remain NOT_COMPARABLE when their input metering perimeters differ. The branch-meter rule is intentionally strict: an auxiliary meter must be present. A true zero auxiliary draw must be represented by an explicit meter observation of zero. An absent meter or unknown energy is not equivalent to zero.

### Accounting identities

`Primary-process electrical input = stored-energy increase + charge conversion loss`

`Auxiliary electrical input = declared auxiliary dissipation`

`Stored-energy increase = energy after hold + storage hold loss`

`Energy withdrawn = useful heat delivered + discharge loss`

`Total wall-plug input = useful heat delivered + charge loss + auxiliary dissipation + hold loss + discharge loss`

An exclusive total meter can be used instead of branch meters, but cannot be summed with branch meters. Duplicate meter boundaries are rejected.

Only after all gates pass is the derived fraction computed:

`wall-plug-to-useful-heat = useful heat delivered / total wall-plug input`

It is not electricity-to-electricity round-trip efficiency, and it is not a thermodynamic exergy efficiency.

## 4. Dispositions

| Disposition | Meaning |
|---|---|
| `PASS` | Declared identities, boundary, meters and energy equations are complete and close |
| `BLOCKED` | Required evidence or service boundary is unknown/incomplete |
| `REJECTED` | Identity, meter scope, unit or accounting invariant is violated |
| `COMPARABLE` | Both scenarios pass and their entire service contracts match exactly |
| `NOT_COMPARABLE` | Valid individual scenarios do not deliver the same declared service |

A `PASS` on a synthetic case does not reproduce an experiment. A `COMPARABLE` disposition means only that the frozen service and wall-plug energy-boundary contracts match; it does not establish technical merit or commercial competitiveness.

## 5. Frozen synthetic reference results

The current fixture deliberately contains:
- a resistance-heated synthetic baseline;
- a plasma-heated synthetic candidate;
- a separate 600°C service candidate;
- unknown auxiliary input;
- an absent auxiliary meter;
- total/branch meter overlap;
- duplicate meter boundary;
- charge, hold and discharge balance defects;
- an incomplete heat-service boundary;
- unknown energy unit;
- a valid exclusive-total-meter scenario.

The same-service reference rows currently compute:
- resistance fixture: 732,804 ppm (73.2804%);
- plasma fixture: 675,609 ppm (67.5609%).

These values are invented test inputs used to exercise arithmetic and comparability. They are **not** measured or predicted technology performance, are not an estimate of plasma efficiency, and must never enter an industry leaderboard.

## 6. Test commands and claim limits

When a Rust-capable environment is available, run:

```sh
cargo fmt --check
cargo test -p xtask plasma_energy_audit
cargo clippy -p xtask --all-targets -- -D warnings
```

The draft PR authors 19 Rust tests, including tests against the frozen source and thermal fixtures, duplicate-identity controls and DEC energy-origin refusal tests. At the time this guide was written, the available execution environment did not contain `cargo` or `rustc`; therefore no Rust test, format, compile or Clippy pass is claimed.

The evaluator does not:
- solve plasma, MHD, fusion, heat-transfer or electrochemical equations;
- estimate thermal material properties or degradation;
- validate a laboratory paper's numerical results;
- authorize a heater, high-voltage/RF system, quench test or physical experiment;
- determine that plasma, resistance or another process is superior;
- replace Symtropy's deterministic run receipts or external solver capability profiles.

The next validation step is to execute the exact-head Rust tests on a Rust-capable host, then proceed to source-specific numerical reproduction only after raw source data, units, boundaries, rights and specialist-solver capability are admitted.


## D–³He fuel-mixture and direct-conversion boundary

The plasma source register includes a 2026 numerical ARC-class tokamak fuel-mixture study comparing D–T with 50% D / 50% ³He. It reports a 525 MW model comparison with source-specific temperatures/densities, average energy per reaction and neutron fractions. A derived arithmetic check records the modeled charged-energy fractions and their ratio.

This is not a DEC experiment. The D–³He primary reaction produces charged alpha/proton products, but the D–³He fuel mixture still produces neutrons through D–D and secondary D–T reactions. The fuel-partition record is explicitly origin-tagged and keeps `fusion_born_output_demonstrated=false` because it contains no direct-converter experiment. Its much lower modelled neutron share is bought with a much higher temperature/density point and unresolved ignition/stability requirements in the source model. The charged-energy fraction must not be promoted to a converter efficiency or plant net electricity.


## Miniaturized and modular plasma device source cases

The source register now adds five external laboratory references under `LOW_TEMPERATURE_PLASMA_MINIATURIZATION` and `LOW_TEMPERATURE_PLASMA_NUMBERING_UP`:

- The 2026 low-cost helium APPJ reference reports a 100 g / 15 × 10 × 2 cm power supply, 54 mm glass tube, 2 L/min helium flow, less than USD 45 component cost and source-specific in-vitro results. The source explicitly leaves long-term electrode wear/drift and practical treatment-area limits unresolved.
- The 2025 membrane-stabilized microfluidic DBD chip reports 7.31–11.32 nm gold nanoparticle sizes with 1.8 nm standard deviation. This does not establish industrial throughput or whole-system energy performance.
- The 43-element LTCC CAP array demonstrates a 10 × 10 cm treatment region, with a source-specific 91% CFU decrease on a named biofilm coupon under its conditions.
- The 2023 RF microplasma array demonstrates five individually ignitable/quenchable cells; expansion to tens or hundreds of cells is discussed by the authors but not a demonstrated scale-out result.

The new negative controls prevent low parts cost from becoming lifecycle qualification, particle-size precision from becoming industrial throughput, one biofilm endpoint from becoming universal efficacy, and a five-cell demonstration from becoming a hundred-cell array claim.

The programme is [MICROPLASMA-001 #11780](https://github.com/Luminous-Dynamics/symthaea/issues/11780).


A fifth miniaturized-device source case records the Leibniz Institute's portable argon MobiPlas: integrated refillable gas tank, mobile powerbank, Raspberry Pi Pico controller, and approximately 10 minutes of operation per gas fill. Its paper also records occasional current instability associated with heating in the supply current-limitation circuit. The INPTDAT catalogue assigns TRL 4; that label remains institutional catalogue metadata and is not treated as independent qualification. The source case and negative control prevent short gas autonomy / single prototype leakage-current observations from becoming lifetime or regulatory claims.


The 2026 miniature helium APPJ article annotates its 2 L/min helium use at approximately £0.2/min under its stated supply-cost basis. At that stated rate, sustained operation would be roughly £12 per operating hour; this is the paper's source-specific estimate, not a universal helium price. This makes gas autonomy and gas consumed per useful output first-order miniature-device metrics, not footnotes.


## 9.14 Derived source arithmetic audit — 2026-11-15

The offline source auditor now validates the machine-readable `derived_reference_checks` array rather than merely ignoring it. For each derived check it requires a unique identity, a `DERIVED_ARITHMETIC_*` evidence class, non-empty inputs and derivation, an interpretation/claim ceiling, and `luminous_measurement=false`. When a source_case_id is provided, it must resolve to a registered source case.

The ammonia microwave check `NH3-MW-ENERGY-YIELD-TO-SPECIFIC-ENERGY-2026` receives an independent arithmetic oracle:
- specific energy = 1000 / reported H2 energy yield (g/kWh);
- expected results: 22.222... kWh/kg H2 at 45 g/kWh and 19.6078... kWh/kg H2 at 51 g/kWh;
- relative reduction = 1 - (45 / 51), while preserving the fact that the two values belong to distinct NH3 conversion operating points.

Negative tests mutate the reported derivation, source-case link and Luminous-measurement flag to ensure the source audit refuses them.

The combined source register on this branch now contains 31 plasma source cases, 33 negative controls and 5 derived checks. The register retains the six microplasma source cases unique to this branch in addition to the two 2026 ammonia cases added from the source-register lineage.


## 9.15 Industrial plasma Power-to-X TEA comparator — 2026-11-15

Kaufmann et al., *Electrification of the chemical industry via plasma power to X – A techno economic view*, Energy 358 (2026), DOI https://doi.org/10.1016/j.energy.2026.141338.

The source models plasma-based biogas-to-syngas pathways and downstream Fischer–Tropsch fuels or methanol. Its leading biogas-reforming model reports approximately EUR 840/t syngas at about 44% plant efficiency for a 300 MW scale. Electricity accounts for roughly 50–60% of total syngas cost. Under favourable renewable-electricity assumptions and reduced full-load hours, the study models syngas costs below EUR 600/t.

This is a techno-economic model, not a physical plant demonstration or observed product price. The 44% figure is a study-specific plant-pathway efficiency, not plasma coupling efficiency. Cost competitiveness must be evaluated under the same accepted syngas composition, feedstock and electricity boundary, capacity factor, power price, location, CAPEX/OPEX, maintenance and downstream product pathway.

The merged source corpus has 32 cases / 36 negative controls / 5 derived checks. It preserves this branch's six microplasma cases in addition to the three 2026 cases added from the main plasma-energy source-register lineage.


## 9.16 Liquid-metal blanket MHD: SAM/NekRS/ALEX benchmark reference — 2026-11-15

Hua et al., *MHD capabilities in SAM and NekRS for fusion liquid metal blanket applications*, Fusion Engineering and Design (September 2026), DOI https://doi.org/10.1016/j.fusengdes.2026.115888.

This paper provides a source-pinned comparison of reduced-order SAM MHD modelling, high-fidelity NekRS MHD CFD, analytical duct references, and ALEX measurements for conducting square and round ducts under applied magnetic fields. Insulating-duct validation is explicitly future work.

This strengthens [PLASMA-MHD-BENCH-001 #11733](https://github.com/Luminous-Dynamics/symthaea/issues/11733). It does not establish whole blanket performance. A valid reproduction must keep model/solver/experiment outputs distinct and pin wall conductivity, geometry, field profile, Hartmann/Reynolds/interaction parameters, boundary conditions and measurement uncertainty.


## 9.17 Warm-plasma CO2 quench source frontier — 2026-11-15

The merged source register now includes four quench-related cases:
- June 2026 atmospheric gliding-arc plasma-cold gas counterflow experiment (DOI 10.1016/j.jece.2026.122322), preserving the 792 W / 23.6% conversion / 25.1% efficiency point separately from the 302 W / 46.2% efficiency highlight.
- October 2026 Al2O3 particle-assisted / SO2 catalytic O-O recombination kinetic model (DOI 10.1016/j.jcou.2026.103579), preserving 79% / 47% and 72% / 43% as numerical predictions only.
- 2026 multidimensional CO2 arc quench CFD model compared to nozzle / heat-exchanger experiments (DOI 10.1016/j.cej.2026.176010); >50% conversion remains conditional on perfect wall cooling.
- July 2026 temperature-management experiment/model (DOI 10.1016/j.rechem.2026.103513), which reports source-specific thermal measurements and radiation losses.

Eight new negative controls refuse point mixing, relative-vs-absolute errors, predicted-to-measured promotion, ideal-wall extrapolation, chemistry/regime transfer and omitted cooling/auxiliary loads.

The current merged register contains **37 source cases / 48 negative controls / 5 derived checks**. It preserves the six miniaturized-plasma sources unique to this audit branch.


## 9.17 Warm-plasma CO2 afterglow-quench evidence — 2026-11-15

The merged audit corpus adds four quench source cases from the source-register branch and eight adversarial controls:
- counterflow MAGD experiment: preserve 792 W / 23.6% conversion / 25.1% efficiency separately from 302 W / 46.2% efficiency where conversion remains unknown;
- particle-assisted catalytic kinetic model: keep 79% / 47% and SO2-assisted 72% / 43% as model outputs;
- multidimensional arc quench CFD: distinguish experimental-comparison operating range from the >50% conversion ideal-wall-cooling prediction;
- thermal-management measurements/model: retain radiation loss and gradient values as source-specific, not universal constants.

The source register must not combine operating points, turn relative improvements into absolute conversion, promote model predictions into experiments, or omit quench energy / pressure-drop / maintenance boundary terms. The merged audit corpus now contains **37 source cases / 46 negative controls / 5 derived checks**.


## 9.18 Counterflow operating-point refusal oracle — 2026-11-15

The Rust source auditor now validates the CO2 counterflow configurations as separate operating points:
- 792 W / 4 slm MAGD: conversion 23.6%, energy efficiency 25.1%;
- 302 W low-power point: efficiency 46.2%, conversion explicitly unknown;
- 2.5 flow-ratio / 30 mm optimization point: 207.3% and 203.6% are relative increases, not absolute conversion/efficiency values.

Four new source-audit tests accept the distinct source record and reject borrowing the 792 W conversion into the 302 W point, converting a relative improvement into an absolute fraction, and duplicate configuration IDs. The focused audit remains solver-free; it qualifies source metadata / operating-point identity, not plasma physics or the source experiment.

The xtask suite now has 31 authored Rust tests. The merged source register remains **37 cases / 46 negative controls / 5 derived checks**.


## 9.19 Corpus snapshot — 2026-11-15

The audit branch retains six microplasma source cases in addition to the source-register content, for 37 unique cases / 46 unique negative controls / 5 derived checks. Its xtask source contains 31 Rust tests, including source-arithmetic and CO2 counterflow operating-point refusal tests. The focused exact-head workflow is queued; no latest test pass is claimed until it completes.


## 9.20 Separate DBD CO2-to-syngas comparator — 2026-11-15

The merged source registry adds a coaxial DBD experiment for humidified CO2-to-syngas (RSC Advances 16 (2026), DOI 10.1039/D5YA00293A). It reports 4.5–23.3% energy efficiency and H2/CO molar ratios of 0.1–0.2 across humidified feeds; a pure-CO2 point is separately reported at 13.3% efficiency and predominantly CO.

This is a different feed/product lane from dry warm-arc CO2-to-CO quenching. H2/CO ratio is not conversion or CO selectivity. Three new controls refuse feed/product comparability errors, pure-feed/humidified-point mixing and metric-semantic substitution.

The audit-branch source corpus now contains **38 source cases / 49 negative controls / 5 derived checks**.


## 9.21 Adjacent DBD syngas source lane — 2026-11-15

The merged corpus adds an experimental coaxial DBD source for humidified CO2-to-syngas, RSC Advances 16 (2026), DOI 10.1039/D5YA00293A. The reported 4.5–23.3% energy-efficiency range and H2/CO ratio 0.1–0.2 belong to humidified CO2/H2O cases; a 13.3% pure-CO2 point is separate. Do not compare these directly with dry warm-arc CO2-to-CO quench results. H2/CO ratio is a composition metric, not conversion or selectivity.


## 9.16 Harmonized methane-pyrolysis comparison and carbon-coproduct qualification

The current machine-readable corpus contains three distinct methane-pyrolysis evidence classes:
- 156-dataset direct-heating review with multi-objective Pareto comparison;
- arc-plasma Aspen Plus / discounted-cash-flow TEA with separate on-site and no-on-site power boundaries;
- harmonized cradle-to-gate LCA/total-cost study with plant-gate H2 at 200 bar, PSA/compression and avoided-carbon-black product credit.

Every methane-pyrolysis source case now carries a `coproduct_accounting` contract with:
- explicit source credit policy;
- `credit_claim_status=SOURCE_MODEL_ONLY`;
- carbon-product qualification, carbon-market capacity and offtake statuses frozen as `NOT_VALIDATED`;
- `realized_revenue_evidence=false`;
- `no_credit_baseline_required=true`.

This is a source/claim floor. It does not preclude future evidence from qualifying a product, market or offtake; it prevents source-reported modeling assumptions from being promoted into Luminous physical or commercial evidence. The Rust auditor now enforces these conservative conditions.


### Corrected ammonia operating-point identity — 2026-11-15

The September 2026 microwave NH3 source now stores three explicit operating-point records. The 150 g/h throughput point is 92.4% conversion at 50 g H2/kWh (217 kJ/mol); it is not the >99% conversion point (45 g/kWh; 242 kJ/mol). The highest energy yield is 51 g/kWh (210 kJ/mol) at 76% conversion.

The derived check now independently calculates 22.22, 20.00 and 19.61 kWh/kg H2 respectively, and separate relative improvements versus the high-conversion point. A new negative unit test mutates the throughput point's derived specific energy to ensure the points cannot be cross-paired.

The audit module now has 36 authored Rust tests. These are test definitions; the focused workflow's result is separately tracked.


## 9.18 Ammonia synthesis full-chain target and evidence floor

The merged register now has a separate `AMMONIA_SYNTHESIS` lane for the 2026 JACS plasma-electrocatalytic nitrogen-reduction experiment and the 2026 industrial-roadmap energy-yield comparison. Do not combine these with ammonia cracking.

The third linked benchmark target `NH3-PLASMA-SYNTHESIS-FULL-CHAIN-001` requires:
- >=3 g accepted NH3/kWh at plant gate;
- H2-generation energy included;
- plasma and electrolysis electricity included;
- gas/feed and product separation energy included;
- declared product purity and pressure contract;
- Haber–Bosch comparison with H2-production energy included.

The Rust validator checks all five guards, the numeric threshold, comparator IDs and `RESEARCH_TARGET_NOT_RESULT` ceiling.


## 9.19 Power-to-X syngas target validation

The source manifest now includes `BIOGAS-PLASMA-SYNGAS-COST-001`, marked `RESEARCH_TARGET_NOT_RESULT`. The auditor requires same-point cost/efficiency thresholds and the five plant-boundary guards for syngas quality, grid-side input, electricity-price/capacity-factor assumptions, downstream service, and no unqualified coproduct credit.


## 9.20 Three-point ammonia operating-point integrity

The NH3 microwave source record now stores three explicit `operating_points`:
- `NH3-MW-ATM-HIGH-CONVERSION`: >99% / 45 g H2/kWh / 242 kJ/mol;
- `NH3-MW-ATM-HIGH-THROUGHPUT`: 92.4% / 150 g H2/h / 50 g H2/kWh / 217 kJ/mol;
- `NH3-MW-ATM-HIGH-ENERGY-YIELD`: 76% / 51 g H2/kWh / 210 kJ/mol.

The Rust auditor requires exactly these three unique operating-point identities and checks the metric pairings. It rejects moving the 150 g/h throughput value onto the >99% point or copying the 45 g/kWh yield into the 92.4% / 150 g/h point.


## 9.23 Ammonia cracking thermal integration target

The fifth machine-readable target, `NH3-THERMAL-INTEGRATION-FULL-CHAIN-001`, binds the 2026 microwave NH3 source case to a matched no-recovery baseline and a same-operating-point engineering challenge:
- >99% conversion;
- at least 150 g H2/h;
- at least 15% reduction in net grid-to-accepted-H2 specific energy;
- heat-balance absolute residual no greater than 5% of declared thermal input.

The Rust audit requires all thresholds at the same point and validates the paired baseline, full-grid boundary, measured useful heat, exchanger/insulation pressure drop and auxiliary loads, thermal storage losses, no double counting of heat/reaction enthalpy, thermal closure with uncertainty, unchanged product contract, matched duty cycle, and grouped uncertainty-supported repeats.

Four new tests check the valid target and reject a missing no-double-count guard, a reduction below the 15% floor, and a target that allows metrics to come from different operating points. This benchmark is a research objective and does not assert that the paper's 150 g/h point also achieved >99% conversion.


## 9.24 N2 vibrational-state transfer and catalyst-interface evidence audit

The merged manifest now contains 41 source cases / 64 negative controls / 5 derived checks / 6 research targets. The new case-level audit for `NH3-PLASMA-ELECTROCATALYSIS-JACS-2026` requires source-separated physical controls, Raman and product observations, and preserves the reported 4,300 K / 11.8% values as model-derived only.

The six-target corpus includes `NH3-PLASMA-N2V-TRANSFER-CAUSAL-001`; its target validator requires a state-resolved catalyst-interface measurement or independently qualified measurement operator, discharge-to-interface transfer uncertainty, isotope attribution, matched thermal/electrical controls and full plant boundary.

The Rust audit module now has 55 authored tests. The latest run on the earlier 55-test revision exposed one valid-target test-fixture omission; the production validator was retained and the fixture guard routing corrected. The corrected exact-head workflow is queued; no pass is claimed until it completes.


## 9.26 Integrated ammonia competitiveness baseline and challenge — 2026-11-16

The audit corpus now includes the integrated SOEC–Haber–Bosch model (Díaz-Motta et al., IJHE 236, published 22 May 2026, DOI 10.1016/j.ijhydene.2026.155184) and the IEA 2026 Haber–Bosch-plus-ASU component assumptions. The former reports 8.14 kWh/kg NH3 at 3,037 t/day and 63% LHV plant efficiency, but is a steady-state model whose stated boundary excludes offshore generation, storage and downstream transport. The latter reports 2.22 GJ/t NH3 electricity for HB+ASU and 0.18 t H2/t NH3 feed; hydrogen production is not included.

The new derived source check computes reciprocal yields, the source-metric ratio and the JACS area-rate geometric extrapolation. It deliberately carries `comparison_status=NOT_COMPARABLE_BOUNDARIES_UNRESOLVED`; it is not a head-to-head result.

A seventh target, `NH3-PLASMA-SYNTHESIS-SOEC-HB-COMPETITIVENESS-001`, now requires <=7.32 kWh/kg accepted NH3 (at least 10% below the 8.14 kWh/kg modeled baseline) with matching product and plant boundaries, including H2 generation, nitrogen supply, feedwater conditioning/desalination, auxiliaries, duty cycle and explicit baseline exclusions. The prior >=3 g/kWh target remains an intermediate research milestone, not industrial parity.

The corpus now has 44 unique cases / 76 unique negative controls / 6 derived checks / 7 research targets.


## 9.29 Pulsed-plasma methane regeneration and full-chain ammonia flexibility — 2026-11-16

The audit corpus now includes four new source cases:
- CH4-PLASMA-PULSED-HEATING-REGEN-2026 (1 November 2026): the source's 95% CH4 conversion / 100% H2 selectivity result is a 10/90 CH4/He point at 46 V / 5 kV. The separate 27 kWh/kg H2 figure is under pure methane at 46 V / 3 kV. CO2-plasma regeneration removes deposits as CO and is reported over four cycles. The source audit now requires these two operating points to remain separate and regeneration energy to remain unknown where unreported.
- NH3-HB-ROBUST-LOAD-FLEX-2026: dynamic Haber–Bosch model at 10–120% load and 3%/min.
- NH3-HB-FATIGUE-YEAR-SCHEDULING-2026: a one-year renewable profile with modeled 10–100% load, 3%/min and fatigue-aware feedforward control.
- NH3-P2A-THREE-LEVEL-DISPATCH-2026: modeled three-level scheduling reducing annual reactor cycling by >80% relative to the study's fully flexible comparison.

Two additional research targets are audited:
- CH4-PPH-FULL-CHAIN-PARETO-001: >=90% conversion, >=99% H2 selectivity, <=25 kWh/kg accepted purified H2 at the complete wall-plug boundary, 100 reaction/regeneration cycles and >=90% productivity retention. All thresholds must hold at one feed/operating point; the source's diluted conversion and pure-methane energy points are not combined.
- AMMONIA-SUPPLY-CHAIN-DYNAMIC-FLEX-001: accepted NH3 at 10–120% load and up to 3%/min ramps with a one-year renewable trace, H2/N2 buffers, pressure/thermal fatigue and accepted product contract. A plasma source ramp cannot stand in for plant-level product ramp; flexible Haber–Bosch is the baseline.

New tests enforce source-point pairings, four-cycle regeneration claim ceiling, and dynamic target guard/comparator integrity. These are benchmark contracts, not Luminous results or physical process qualification.


## 9.30 Hybrid plasma–thermal APPJ methane conversion — 2026-11-16

Sabir et al., *Methane conversion by a plasma-assisted process: Role of hybrid plasma–thermal configurations in controlling conversion pathways*, International Journal of Hydrogen Energy 285 (2026), article 158179, DOI https://doi.org/10.1016/j.ijhydene.2026.158179.

The source reports a thermally assisted atmospheric-pressure plasma jet that couples methane cracking and non-oxidative methane coupling to co-produce H2, C2-C3 hydrocarbons and solid carbon. Placing plasma directly within the heated zone reportedly raises methane conversion by more than fivefold versus a source-compared configuration; the lowest carbon deposition is around 36% over the studied temperature range. Recovered solid carbon is characterized as having turbostratic nanostructural features.

The source register and Rust auditor keep the following distinctions executable:
- >5-fold is a **relative** conversion gain; absolute CH4 conversion remains unknown until the exact article comparator and data are reconstructed.
- ~36% is a source-specific **carbon-deposition** metric with its denominator still to be reconstructed; it is not carbon capture, recovered feed-carbon fraction or saleable product yield.
- H2 rate/purity/pressure and C2-C3 species-specific yield remain unquantified in the register.
- Turbostratic structure is characterization, not commercial-grade validation, market capacity or offtake.
- Complete grid-to-accepted-product energy is not established by the source abstract.

The dedicated research programme is [CH4-PLASMA-THERMAL-APPJ-001 #12451](https://github.com/Luminous-Dynamics/symthaea/issues/12451), separate from pulsed DBD/resistive-heating and CO2-plasma regeneration in #12430. Symtropy integration is tracked by [CH4-PLASMA-THERMAL-APPJ-TWIN-001 #5525](https://github.com/Luminous-Dynamics/symtropy/issues/5525).

### Audit implementation

The machine-readable case is `CH4-PLASMA-THERMAL-APPJ-2026`. The Rust source auditor requires the source DOI, APPJ configuration identities, relative conversion lower bound, unresolved comparator, unresolved deposition denominator, null captured-carbon/recovery values, qualitative turbostratic characterization, unquantified H2 and C2-C3 product fields, NOT_COMPARABLE status and no-credit coproduct policy. Five new tests exercise acceptance and refusals for absolute-conversion promotion, deposition-to-recovery promotion, carbon-grade promotion and inferred H2 rate.

The source-register branch contains 43 source cases / 82 negative controls / 5 derived checks / 9 benchmark targets. The merged offline-audit branch contains 49 cases / 88 controls / 6 derived checks / 9 targets, preserving six modular-plasma cases unique to that lineage. These counts are exact to the frozen branches identified in the PR descriptions; they are not counts of physical experiments performed by Luminous.
