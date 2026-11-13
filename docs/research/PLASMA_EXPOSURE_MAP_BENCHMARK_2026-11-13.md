# PLASMA-EXPOSURE-MAP-001 — Spatially resolved delivered plasma exposure

**Date:** 13 November 2026  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11925  
**Parents:** MICROPLASMA-001 #11780, PLASMA-SURFACE-001 #11795, PLASMA-VM-001 #11847, PLASMA-OED-001 #11907  
**Physical/digital twin:** Symtropy PLASMA-TWIN-001 #1685 and PLASMA-ARRAY-SCALEOUT-001 #4926  
**Status:** synthetic-only mapping contract  
**Claim ceiling:** no physical exposure reconstruction, dose field, chemistry prediction, surface qualification or hardware execution claim.

## 1. Research question

Can Luminous reconstruct the *local interaction field delivered to a real surface* from synchronized geometry, motion, electrical/optical observations and calibrated spatial registration—and show that the reconstruction transfers to unseen thickness, curvature, edge and motion conditions?

The target is not one universal plasma-dose scalar. It is a local, time-dependent vector of observed and inferred process/exposure channels, each with its own units, uncertainty, provenance and applicability.

## 2. Literature motivation

A 2024 RF cold-atmospheric-plasma slit-jet study reports that 5 mm and 30 mm polypropylene specimens had different plasma treatment efficiency even when the slit-to-top-surface gap was fixed at 10 mm. Substrate thickness changed the distance to the grounded boundary and influenced filament number/speed; edge and center regions also differed under selected operating conditions. DOI: https://doi.org/10.1007/s11090-024-10517-0.

A 2014 APPJ polymer pretreatment study reports effects from nozzle gap, treatment velocity, voltage and duty cycle on surface-energy outcomes and overlap-bond strength. DOI: https://doi.org/10.1063/1.4873744.

A 2020 APPJ modified-area study reports coupled and nonlinear effects of jet distance, gas flow, target grounding, incidence angle and ambient conditions on modified area. A shorter gap or higher flow is not a universal monotonic improvement. DOI: https://doi.org/10.3390/polym12051028.

These are literature anchors, not Luminous reproductions. They establish that a nominal recipe, commanded standoff or visible plume cannot by itself prove equivalent local surface exposure across different parts.

## 3. Paradigm shift: recipe -> exposure field -> product response

The full engineering chain should be:

```
machine command / recipe
    != realized tool trajectory
    != local geometry and boundary conditions
    != delivered plasma interaction field
    != local surface/interface state
    != product performance
```

For each spatial region (r) on surface (Omega), preserve an exposure descriptor:

```
X(r,t) = {
  local standoff,
  local grounded-boundary distance,
  surface normal / curvature / edge distance,
  incidence angle,
  relative velocity,
  dwell and pass overlap,
  electrical waveform / power observations,
  optical / filament observations,
  gas / pressure / ambient state,
  thermal history,
  uncertainty and provenance
}
```

This is a vector, not a universal scalar. A scalar integrated exposure may be derived only for a frozen material/process/application profile when a validated mapping establishes which channels and mechanisms it represents.

OES intensity, plume length, commanded voltage, wattage, contact angle and nominal dwell are not interchangeable with radical flux, ion flux, absorbed energy or useful chemical modification.

## 4. Spatial identity and registration

Every map binds:
- canonical MAT-013 #4344 sample/article/lot/conditioning lineage;
- canonical MAT-INTERFACE-001 #6227 surface/interface subject;
- geometry identity, revision and digest;
- machine/tool/nozzle identity and process configuration digest;
- trajectory identity and timebase;
- machine, article and surface coordinate frames;
- explicit machine-to-article and article-to-surface transforms;
- sensor identity, calibration/validity, clock offset, raw trace digest and acquisition time;
- region IDs, spatial coordinates, coverage, interpolation source and uncertainty;
- unmapped/occluded/saturated regions and explicit reasons.

Transform and interpolation uncertainty must travel with every mapped region. Missing regions cannot silently become zero exposure.

## 5. Field channels and their measurement class

Separate directly observed, geometry-registered, and model-inferred quantities:

| Channel | Typical source | Required interpretation |
|---|---|---|
| Local standoff | Geometry + calibrated tool pose | Distance along declared local surface normal |
| Ground-boundary distance | Article/fixture geometry | Not identical to top-surface standoff |
| Surface normal / curvature | CAD/scan and frame registration | Versioned geometry-derived value |
| Incidence angle | Tool pose + surface normal | Signed angle and uncertainty |
| Relative speed / dwell / overlap | Encoder/robot trajectory | Timebase-aligned kinematic quantity |
| Electrical power/waveform | Calibrated electrical instrumentation | State exactly where power was measured |
| Optical/filament signal | Calibrated camera/OES | Instrument- and view-specific proxy, not a universal chemistry meter |
| Gas/pressure/ambient | Calibrated process instrumentation | Exact gas, flow basis, pressure, humidity and location |
| Thermal history | Calibrated thermal sensor or validated model | Measurement and model inference remain distinguishable |

Each inferred value needs a model/profile identity and uncertainty. A missing calibration, stale clock, frame mismatch or unbounded interpolation must cause HOLD/BLOCK rather than produce a confident map.

## 6. Benchmark outputs

Primary mapping metrics:
- required-region coverage fraction;
- unmapped/occluded area and maximum contiguous unknown region;
- machine-to-article and article-to-surface registration residual;
- clock synchronization residual and calibration age;
- local standoff / ground-distance / incidence / dwell distributions;
- edge/center and leading/trailing field variation;
- interpolation uncertainty by region;
- field reconstruction transfer error on held-out geometry and motion;
- worst-region false-negative risk for under-treatment/over-treatment.

Product outcome metrics (adhesion, failure mode, bondability lifetime, barrier preservation, conductivity) remain in PLASMA-SURFACE and application children. A field-map PASS does not imply a surface or product PASS.

## 7. Matched baselines

At the same geometry, process and measurement budget, compare:
1. recipe-only: nominal standoff, power, flow and speed;
2. kinematic scalar: dwell/pass/overlap map without plasma-state observations;
3. optical/plume-only map;
4. geometry + motion + synchronized electrical/optical/thermal observations;
5. full mapped field with independent witness-region response measurements.

More pixels do not constitute a better map. The method must improve held-out spatial reconstruction or downstream region prediction at a matched measurement and calibration budget.

## 8. S0 synthetic scenarios

The fixture contains controls for:
- complete planar map;
- same nominal top-surface gap but changed substrate thickness/ground-boundary distance;
- missing trailing-edge region;
- unknown region falsely filled with zero;
- stale electrical/optical/timebase synchronization;
- coordinate-frame / transform mismatch;
- duplicate region identity;
- high interpolation uncertainty;
- expired calibration;
- geometry/configuration digest mismatch.

The Rust audit checks identity, frame/transform completeness, region coverage, duplicate/unknown region semantics, numeric domains, calibration and clock freshness, raw-trace identity and interpolation uncertainty. It performs no plasma physics or exposure-to-chemistry inference.

## 9. Proposed research objectives

Once a real mapping instrument and spatial pitch are frozen:
- 100% of scored regions carry article, geometry, coordinate-frame and configuration identity;
- no unknown region is encoded as zero exposure;
- spatial registration uncertainty remains below one declared sampling pitch;
- clock skew remains within the calibrated acquisition window;
- region coverage and worst-region uncertainty are reported, never hidden by global averages;
- held-out thickness, curvature, edge and motion profiles are evaluated separately;
- model transfer abstains when geometry or grounding boundary lies outside validation support.

These are proposed objectives, not standards or achieved performance.

## 10. Ownership and reuse

- MAT-013 #4344: sample, batch, article and conditioning lineage.
- MAT-INTERFACE-001 #6227: surface/interface identity and evolution.
- FIELD / SE-OBS / ROB-INSTR / ROB-EXP: physical observation, instrument calibration, timebase and experiment identity.
- ENG-SEM / SE-SEM: physical quantities, units, frames and transforms.
- PLASMA-VM-001 #11847: calibrated run-to-specimen and telemetry-to-product linkage.
- PLASMA-SURFACE-001 #11795: bondability/product acceptance.
- PLASMA-OED-001 #11907: value/cost selection of additional spatial and destructive observations.
- Symtropy PLASMA-TWIN-001 #1685: geometry and fault replay, not physical evidence substitution.

This programme does not create a second spatial coordinate ontology, calibration registry, sample registry, evidence receipt or generic digital twin.

## 11. Claim ceiling

A software PASS establishes only synthetic spatial identity, coverage and refusal semantics. It does not establish a physical plasma exposure field, radical/ion flux, absorbed energy, chemical dose, surface activation, bondability, process recipe, device qualification, safety case or authorization for HV/RF/gas operation.
