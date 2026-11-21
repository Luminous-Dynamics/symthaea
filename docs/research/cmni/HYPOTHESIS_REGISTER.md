# CMNI falsifiable hypothesis register

**Snapshot:** 2026-11-20  
**Programme:** https://github.com/Luminous-Dynamics/symthaea/issues/13023  
**Evidence anchor:** Karahadian et al. (2026), https://doi.org/10.1038/s41467-026-74421-1

## Purpose

The 2026 Pd/Ti hydride study reports a low-energy D–D yield plateau below approximately 2 keV and explicitly states that its bi-exponential/energy-dependent effective screening description is phenomenological, not a unique microscopic model. The authors note that the fitted effective screening parameter can lump together physically distinct mechanisms.

This register turns those mechanisms into competing, measurable hypotheses. A fit parameter is not a mechanism. Multiple mechanisms may coexist, and an observation that is compatible with a hypothesis does not identify it uniquely.

## Evidence boundary

The 2026 result is accelerator-driven D–D fusion in metallic foils/hydrides. The low-energy proton and higher-energy neutron channels have different measurement coverage; do not infer simultaneous cross-channel validation at every energy. Electrochemical loading without an ion beam produced only background-level signals in the reported experiment. This register does not treat that result as evidence for accelerator-free room-temperature LENR.

## Hypothesis register

| ID | Hypothesis | Distinguishing predictions | Measurements that discriminate | Falsifier / weakening evidence |
|---|---|---|---|---|
| H0 | Conventional baseline + measured target density/stopping/beam spread/instrument response explains the observations | Residuals become statistically consistent with the baseline after source-faithful nuisance and calibration treatment | Independently measured beam-energy distribution, stopping, target profile, current/charge, detector response and background | Structured residuals persist across independent calibrations and campaigns, outside quantified uncertainty |
| H1 | Homogeneous electronic screening is sufficient within each material/phase | One effective screening parameter per material/phase predicts held-out energies and loading states; no systematic depth dependence remains | High-energy calibration region, material electronic properties, held-out low-energy measurements | Constant-parameter model systematically fails held-out low-energy data after nuisance uncertainties are propagated |
| H2 | Heterogeneous near-surface defect/trap environments contribute to the plateau | Yield tracks a depth-dependent distribution of defects and trapped D; changing defect population at matched D occupancy shifts yield in a predictable direction | Depth-resolved D profiling (SIMS, NRA, ERDA); open-volume defect measurements (positron annihilation spectroscopy); controlled anneal/irradiation history and pre/post characterization | Plateau persists unchanged when independently measured defect/trap populations are strongly changed at matched D profile, or predicted depth correlations are absent |
| H3 | Non-equilibrium D transport/occupancy is a major driver | Yield has time-, flux-, temperature- and loading-history dependence; Pd/Ti differences track independently measured diffusion/replenishment/loss and near-surface D density | Time-resolved D profiles, loading/unloading history, beam flux and duty cycle, temperature, RGA/vacuum, pre/post depth profiles | Results are invariant to transport/history interventions once D profile is matched, or transport model predicts the wrong direction/time constant |
| H4 | Electronic-structure differences control a substantial component | At matched D occupancy, defects and geometry, material-to-material changes correlate with independently calculated/measured electronic structure and screening response | DFT/electronic-structure calculations, conductivity/DOS measurements, controlled composition/phase series | Effect follows D density/defects alone and does not transfer with electronic-structure predictors on held-out materials |
| H5 | A distinct low-energy nuclear mechanism (e.g. resonance) contributes | Energy-localized structure, branching/particle spectra, angular distributions or other nuclear signatures not explained by smooth screening/transport models | Fine energy scans with calibrated spread; independent particle spectroscopy; channel ratios in actual overlap regions; predicted product searches | Predicted resonance features/products are absent at preregistered sensitivity across the relevant energy range |
| H6 | Instrument, calibration, analysis, or environmental artifacts explain some/all of the apparent effect | Apparent signal follows detector state, beam composition/current, energy calibration, background, dead time, threshold or sample-order effects rather than material state | Independent detectors; blinded calibration/injection; randomized condition order; beam-only/no-beam and H2O/D2O controls; clock and DAQ audit | Same effect survives blinded controls, independent instrumentation, calibration changes and independent-lab replication |
| H7 | A claimed heat anomaly is chemical, electrical, thermal or accounting-related rather than nuclear | Heat follows reaction chemistry, recombination, thermal lag, input accounting or apparatus state without quantitatively matching nuclear products | Traceable calorimetry plus full energy balance, blank/reference-heater runs, gas/liquid accounting, independent nuclear-product diagnostics and isotope assays | Heat is time-correlated and quantitatively reconciled with independently measured nuclear products and the reaction energy balance across replication |\n| H8 | A non-negligible D₂⁺ molecular-ion component creates a second per-deuteron energy population after dissociation | Under prompt symmetric dissociation, D fragments from singly charged D₂⁺ carry about half the molecular kinetic energy; at the same acceleration potential their D–D center-of-mass energy is about half that of D⁺. Yield predictions therefore depend on mass-resolved species fractions and dissociation depth, not just nominal voltage and a single energy-spread scalar | Mass-resolved beam-species fractions vs operating point; energy distribution per species; dissociation/fragmentation evidence; independent beam current and charge calibration; time-resolved or energy-resolved product data | A sufficiently low D₂⁺ fraction or negligible/late dissociation makes its forward-modelled contribution too small to change held-out yields within uncertainty, or mass-resolved diagnostics show no distinct lower-energy component |

## Causal structure

The model must represent the following pathways separately:

`sample preparation/history → phase, strain, defects, surface state`

`loading + beam flux + temperature + time → local D occupancy and transport`

`material electronic structure + local D + defects → candidate reaction environment`

`beam energy/species + stopping + local state → reaction probability`

`reaction products + geometry + detector response + background → measured observation`

A direct correlation between effective (U_e) and measured yield does not identify which upstream pathway is causal. The same (U_e) value may absorb stopping-power errors, occupancy differences, depth weighting, and detector normalization.

## High-information intervention families

These are study-design families, not operating instructions for laboratory equipment.

1. **Beam × electrochemical loading:** compare beam-only, beam-plus-deuterium-loading, loading-only, and matched blank/control states. The loading-only condition tests for signals without accelerator-driven reactions.
2. **Isotope control:** compare deuterium-bearing and hydrogen-bearing conditions while matching apparatus, current, thermal history, and material preparation as closely as possible.
3. **Material transfer:** compare Pd and Ti (and later other justified materials) with independently matched/quantified occupancy, thickness, defect state and surface condition.
4. **Defect intervention:** alter/relax defect populations while measuring whether the D profile remains comparable; require pre/post defect and occupancy diagnostics.
5. **Transport intervention:** perturb loading history, beam flux/duty and thermal state to test predicted lags, depletion and replenishment; record time-resolved state rather than only final averages.
6. **Depth intervention:** use characterized surface/overlayer and depth profiles to test whether response follows the near-surface reaction region.
7. **Excitation transfer:** only after the baseline is stable, evaluate alternative incident ion species or non-ion-beam stimuli suggested in the literature (photons, THz phonons, plasmons). These are hypotheses for tests, not established triggers or mechanisms.

## Pre-registration template

Every candidate experiment must publish before the data are unblinded:

- H0/H1/... hypotheses being separated.
- One quantitative prediction per hypothesis, in observable units.
- The exact dependent and independent variables, ranges, and measurement uncertainty.
- Required calibration, backgrounds, negative controls and positive references.
- Detector channels, expected spectra, energy windows, efficiency, and which channels are actually available.
- Sample history and pre/post materials characterization.
- Expected effect size and power/detection sensitivity.
- A result that would falsify each hypothesis.
- Analysis code revision, exclusion rules, model priors, thresholds and statistical framework.
- How correlated systematic errors and multiple comparisons are handled.
- Data release, provenance, and independent review plan.

The analysis must report null outcomes and upper bounds with sensitivity. A non-detection without a declared detection limit is not a useful constraint.

## Priority ordering

1. Explain/reproduce beam-driven observations with source-faithful nuisance models.
2. Separate beam-energy/stopping/density effects from material-state effects.
3. Distinguish defect/trapping from transport and electronic-structure pathways.
4. Search for independent nuclear signatures only where the selected mechanism predicts them.
5. Keep any accelerator-free LENR/heat claim in a separate protocol requiring direct nuclear-product evidence, complete energy accounting and independent replication.

## References

- Karahadian et al. (2026), https://doi.org/10.1038/s41467-026-74421-1
- Chen et al. (2025), https://doi.org/10.1038/s41586-025-09042-7
- ARPA-E LENR Exploratory Topic criteria, https://arpa-e-foa.energy.gov/FileContent.aspx?FileID=68c066ad-2aa8-4bbe-bfbf-fcb73762fdda


## H8 follow-up: beam-species mixture is a required nuisance model

The 2026 Methods report a typical ion source composition of approximately 85% D⁺ and 15% D₂⁺ and calculate deuteron flux by weighting each D₂⁺ ion as two deuterons. Under the explicit prompt-dissociation limiting assumption, 15% D₂⁺ ion-current fraction corresponds to:

[
f_{D	ext{ from }D^+}=rac{0.85}{0.85+2(0.15)}approx0.7391,qquad
f_{D	ext{ from }D_2^+}=rac{2(0.15)}{0.85+2(0.15)}approx0.2609.
]

For a singly charged ion accelerated through (V) kV, D⁺ has (E_{cm}approx V/2) keV for D on stationary D. If D₂⁺ dissociates while approximately retaining the molecular velocity, each D fragment has (E_{cm}approx V/4) keV. This is a conditional forward-model scenario; the actual dissociation probability and depth are not established by the published scalar beam-energy spread.

This becomes an explicit competing explanation/nuisance model. The model must be compared at matched total deuteron flux and cannot be described as explaining the low-energy plateau without mass-resolved beam diagnostics and prospective predictions.

Source: Karahadian et al. (2026), Methods, https://doi.org/10.1038/s41467-026-74421-1


## H9 — isotope composition, electrolyte contamination and fuel-state uncertainty

| ID | Hypothesis | Distinguishing predictions | Measurements that discriminate | Falsifier / weakening evidence |
|---|---|---|---|---|
| H9 | Variations in D2O purity / H2O contamination change D/H occupancy in Pd and explain some between-campaign yield variation, rather than requiring a change in nuclear mechanism | Response varies with measured D2O/H2O composition, water-exposure history and independently characterized D/Pd state; matched H2O controls reduce the beam-driven DD signature | Electrolyte isotope assay before/after use, direct D/H loading measurements, RGA, controlled D2O/H2O mixtures, XRD or depth-sensitive isotope profiling, matched beam and detector conditions | Yield variation remains after isotope composition/fuel state are matched, or controlled D2O/H2O changes fail to predict the observed response within uncertainty |

The 2025 paper reports an 88.3(3)% decrease when 2 M K2CO3 in H2O replaced D2O and suggests ambient H2O contamination of D2O may explain between-campaign variability. It also explicitly states that D/Pd during the fusion runs was not directly measured; ~0.7 was inferred from separate XRD characterization. These are valuable controls and a material-state uncertainty, not proof that every confounder is excluded.

Keep these control types separate:
- **Sheath voltage OFF:** stops incident D+ implantation; signal returns to background.
- **Electrochemical cell OFF while beam remains active:** tests persistence of loaded D in Pd; reported variation at most 2%.
- **H2O instead of D2O:** isotope/material-loading control; reported 88.3(3)% reduction.
- **Seven-target beam-loading set:** repeatability at 0.2 mA / 120 min; not part of the 0.5 mA three-campaign Phase I/II comparison.

Source: Chen et al. (2025) and Supplementary Information, https://doi.org/10.1038/s41586-025-09042-7


## H10 — spatially nonuniform beam fluence and irradiation history

| ID | Hypothesis | Distinguishing predictions | Measurements that discriminate | Falsifier / weakening evidence |
|---|---|---|---|---|
| H10 | Voltage-dependent beam focusing and nonuniform local fluence alter the near-surface defect/D state and can contribute to energy-dependent yields independently of total charge | At matched total charge and material average, yield/state changes with spot size, radial profile, target alignment, beam drift and peak current density; a spatial state model predicts held-out profiles better than a uniform beam/depth-only model | 2-D beam profile at target plane for each bias; Faraday-cup aperture/collection calibration; spot translation/drift; local dose/implantation mapping; pre/post spatial material and defect characterization | Yield is invariant within uncertainty across controlled profile changes at matched charge and local state, or spatial model does not improve held-out prediction |

The 2026 Methods report an approximately 1 cm² spot and voltage-dependent electrostatic focusing above 5 kV. This makes total incident charge insufficient to fully characterize the local irradiation field unless the spot profile is stable and verified. The 2026 paper also reports very shallow implantation depths compared with 250 µm foil thickness. H10 is an alternative contributor to test, not an explanation established by the paper.
