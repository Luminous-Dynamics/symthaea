# Planetary Optical Impact and Spectral Compatibility — v0.1

**Parent:** [PLANETARY-ENERGY-SERVICE-001 #11756](https://github.com/Luminous-Dynamics/symthaea/issues/11756)  
**Focused issue:** [PLANETARY-OPTICAL-IMPACT-001 #11796](https://github.com/Luminous-Dynamics/symthaea/issues/11796)  
**Status:** requirements and synthetic contract  
**Claim ceiling:** metadata, evidence completeness and synthetic declared-threshold audit only; no optical-safety or physical qualification claim.

## 1. Design intent

Energy infrastructure should minimize unnecessary optical disturbance to nearby people, habitats, scientific instruments, observatories and potentially sensitive environments. This is **low-optical-impact engineering**, not covert concealment.

The design must remain inspectable, safe to navigate, auditable and compatible with legitimate scientific observations. It must not optimize for evading legitimate observers, defeating instrumentation or disabling safety indicators. Required hazard, navigation and emergency signals must remain available.

The relevant design question is:

> Can the installation deliver its required electrical and thermal service while keeping reflected, scattered and emitted radiation within a declared impact budget for every protected observer and instrument?

The output is a vector of wavelength-, direction-, temperature- and observer-specific impact quantities, not a universal "signature" scalar.

## 2. Established engineering basis

### 2.1 End-to-end stray light is a system problem

NASA GSFC-STD-1000H §5.15 requires applicable optical systems to undergo end-to-end stray-light modelling and system-level test. Its rationale explicitly includes optical/hardware scattering, thermal self-emission, unintended optical paths, glints, vignetting and as-built surface properties. The NASA rule is a strong engineering pattern for our observer-impact budget; project applicability and binding requirements must be determined by the responsible mission authority.

Source: https://standards.nasa.gov/sites/default/files/standards/GSFC/H/0/GSFC-STD-1000RevH_Approved.pdf

### 2.2 Astronomy needs angular and wavelength-dependent reflectance data

The IAU CPS consolidated recommendations specify reflected-sunlight brightness guidance for relevant satellite orbital conditions, recommend reducing high-amplitude glints, and call for public radiometric visibility models informed by wavelength-dependent BRDF and validated observations. Its magnitude-7 criterion is an orbital astronomy reference, not a universal lunar surface limit or a biological/eye-safety threshold.

Source: https://cps.iau.org/documents/44/Consolidated-CPS-Recommendations.pdf

### 2.3 Visible reflectance and thermal emission are separate

NASA's spacecraft thermal-control guidance distinguishes solar absorptivity from IR emissivity. Radiators generally require low solar absorptivity and high IR emissivity. NASA also warns that coatings and tapes can change optical properties during a mission, so end-of-life conditions must be modeled rather than relying on beginning-of-life datasheets.

Source: https://www.nasa.gov/smallsat-institute/sst-soa/thermal-control/

**Consequence:** a surface can be visually dark and still emit substantial thermal IR; a low-emissivity surface can reduce one radiometric signature while causing overheating. Do not call a surface "invisible" merely because its visible reflectance is low.

## 3. Surface function must determine its spectrum

Do not apply one coating or finish to every component.

| Surface role | Desired function | Principal trade |
|---|---|---|
| PV collection face | Transmit/absorb useful solar photons | Minimize unwanted glint without sacrificing photon capture, durability or cleaning |
| Structural frame / backside | Avoid sharp specular paths and unwanted scatter | Surface finish, thermal balance, charging and abrasion |
| Thermal radiator | Reject heat efficiently into a permitted sink direction | Low solar absorptance, adequate IR emissivity, line-of-sight and observer background |
| Optical concentrator / mirror | Deliver controlled flux to a receiver | Preserve designed optical path and prevent stray/hazardous beams |
| Beacon / status indicator | Safe navigation and emergency awareness | Use minimum necessary emission while retaining required visibility |
| Laser power/data terminal | Deliver authorized optical link with safety controls | Independent interlocks, beam termination, exclusion geometry and fail-safe state |

### 3.1 Solar collectors

Investigate anti-reflective multilayers, graded-index and textured surfaces, durable nanostructures, and edge/frame glint reduction. Measure BRDF over the relevant incidence and viewing angles. An anti-reflective coating's normal-incidence reflectance is not a sufficient predictor of low glint at grazing angles.

### 3.2 Thermal radiators

Preserve heat rejection. Consider spectral-selective materials that reduce unnecessary solar absorption while retaining adequate thermal-IR emission toward a suitable sink. Optimize radiator placement and field of regard so that hot surfaces do not contaminate sensitive instrument observations where geometry permits.

### 3.3 Indicators and optical power

Never suppress emergency/navigation indicators to improve an optical-impact metric. Optical power and communication emitters require a separate safety case; unknown interlock or exclusion state must force a safe, non-emitting state. This benchmark does not authorize lasers or hardware actuation.

## 4. Current materials frontier

The 2026 W:VO2 smart optical solar reflector paper reports a flexible 10 × 10 cm² demonstrator with solar absorptance around 0.22, hot-state IR emissivity around 0.80 and emissivity contrast around 0.33. Its reported tests include adhesion, bending, thermal cycling, proton/electron exposure and humidity on test coupons. The paper calls for further in-orbit verification; it is not lunar-qualified.

Source: https://doi.org/10.1002/nap2.70011

A separate 2026 2D-network VO2 metasurface radiator paper reports solar absorptance around 0.26 and temperature-adaptive thermal emittance switching from approximately 0.28 to 0.92, combining material experiments with orbital thermal modelling and terrestrial tests. It is not a demonstration of lunar radiator service or visible-band glare compliance.

Source: https://doi.org/10.1002/lpor.71614

These are encouraging materials directions, not interchangeable metrics. Their spectral windows, angular behavior, temperature, substrate, sample scale and test protocols differ. Compare them only after primary methods have been retrieved and measurements reproduced at matched conditions.

## 5. Optical-impact budget schema

Every promoted surface measurement must bind:

- body/site/asset and coordinate frames;
- surface, as-built configuration, material lot and coating recipe;
- observer/instrument identity, detector spectral response, field of regard, integration time, saturation/background requirements;
- wavelength band and measurement spectral resolution;
- illumination spectrum, incidence angle and polarization if relevant;
- observation angle and angular sampling coverage;
- BRDF / BTDF evidence, specular lobe, diffuse scatter and surface roughness;
- hemispherical and spectral solar absorptance / reflectance / transmittance;
- IR emissivity and temperature;
- observer-directed spectral radiance/irradiance and its uncertainty;
- clean, dusty, aged, radiation-exposed and post-cleaning states;
- exact instrument/calibration/model/test identities and raw-data hashes;
- missing angular/spectral/time coverage and explicit unknowns.

A BRDF receipt ID or spectral-property receipt ID is only a reference string until the upstream owner resolves and verifies the evidence. The v0.1 evaluator does not authenticate those receipts.

## 6. Observer classes and mission contexts

Create distinct impact budgets for:
1. crew habitat windows and external work zones;
2. rover navigation, science cameras and lidar;
3. star trackers, sun sensors and optical navigation;
4. visible/near-IR astronomical telescopes;
5. UV/visible/IR spectrometers and radiometers;
6. cryogenic and low-background thermal instruments;
7. nearby assets receiving reflected or thermal flux;
8. dark/undisturbed and potential biological-protection zones.

Observer thresholds must be mission-specific and established by the instrument owner, safety authority or planetary-protection review. Do not copy IAU orbital satellite limits into a lunar surface design without justification.

## 7. Metrics — preserve separate spectral and service dimensions

**Optical:** peak specular radiance, BRDF lobe, hemispherical reflectance, band-integrated observer radiance, stray-light focal-plane background, signal-to-noise degradation, saturation risk, contrast loss and lost observing time.

**Thermal:** solar absorptance, IR emissivity, spectral radiance, radiator heat rejection, observer-directed thermal flux, radiator mass/area, view factor and component temperature margin.

**Energy:** net delivered electrical energy, PV optical transmission/collection penalty, coating-induced temperature changes, parasitic loads and useful thermal service.

**Durability:** property drift after UV, radiation, vacuum, thermal cycling, dust/abrasion, micrometeoroid-representative impacts and cleaning.

**Safety/transparency:** interlock coverage, fail-safe state, shutdown latency, safety indicator availability, as-built optical-property disclosure and configuration provenance.

Do not sum optical radiance across unrelated detector bands without a defined instrument response. Do not add electrical and thermal energy as if they are equivalent.

## 8. Synthetic benchmark dispositions

The first offline fixture has eight cases:
- a solar-collector record that is complete enough for a contract-only pass;
- a low-glare claim blocked by missing BRDF data;
- a radiator that exceeds a synthetic observer-radiance budget;
- a low-emissivity surface that cannot meet a declared radiator minimum;
- an optical transmitter blocked by unknown independent interlock;
- a beacon configuration blocked because its emergency indicator is unavailable;
- an incomplete structural surface record;
- a solar surface with no declared observer limit.

The command is:

~~~bash
cargo run -p xtask -- planetary-optical-impact-audit
~~~

A report of PASS_CONTRACT_ONLY means only that the synthetic record is complete and its declared values meet the synthetic threshold. It does not mean low-glare, biologically safe, instrument-safe, thermally qualified or physically acceptable.

## 9. Negative controls

- BRDF missing but a claim of zero glint is made.
- Normal-incidence anti-reflective data is extrapolated to grazing angles.
- Black/dark finish increases solar absorption and is described as low thermal impact.
- Low-IR-emissivity film is used as a radiator while rejecting too little heat.
- Radiator has high emissivity but its view factor points toward a protected detector or warm surface.
- Surface dust or aging changes BRDF/absorptance/emissivity but the original qualification is retained.
- Emergency indicator or warning signal is disabled to reduce brightness.
- Laser optical output is called safe because it is not visible to the unaided eye.
- Unknown interlock or exclusion-zone state is treated as operational.
- A material-coupon test is promoted to a full-array or lunar qualification.

## 10. Proposed research gates

These are internal research targets, not universal standards:
1. 100% of promoted surface data has spectral, angular, temperature, observer and as-built configuration identities.
2. Zero low-impact claims with missing BRDF or relevant detector-response evidence.
3. End-to-end stray-light model and measurement agree within a pre-registered uncertainty budget for all protected observer cases.
4. Zero optical-emitter operation when independent interlock/fail-safe status is unknown or failed.
5. Required navigation and emergency indicators remain available in every safety-relevant operating state.
6. For matched conditions, target ≥10 dB reduction in peak observer-directed glint relative to a documented baseline with ≤2% loss in net delivered electrical service and no thermal/safety regression. This is a stretch target for selected geometries/bands, not an assumption that all surfaces can meet it.
7. Re-test after representative aging/contamination/abrasion; publish end-of-life property bounds.

## 11. Ownership

- Symthaea: multi-objective surface/architecture candidates, material hypotheses and decision-aware experiment selection.
- Symtropy: compose existing radiometric, optical, structural and thermal model owners; compare predicted and as-built evidence.
- Sol-Atlas: site, instrument, observer and protected-zone spatial geometry.
- Mycelix: coating recipe, material lot, instrument calibration, as-built configuration, test results and qualification lineage.
- Independent optical metrology, observatory and laser-safety authorities: measurement and approval authority.

Do not add duplicate BRDF, ray-tracing, radiative-transfer, observatory pipeline, laser-safety or thermal solver implementations.

## 12. Claim ceiling

This work begins as a requirements/evidence contract and synthetic evaluator. It does not establish invisibility, covert concealment, biological safety, optical non-interference, low-glint performance, thermal compatibility, laser safety or flight/lunar qualification. Those require defined observers, calibrated spectral/angular measurements, validated end-to-end models, independent safety review and representative environmental testing.
