# MARINE-ACOUSTIC-ECOLOGY-001
## Underwater radiated-noise qualification for cargo propulsion and WAPS co-design

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Implementation:** #11603

> This is an evidence and benchmark programme, not a quiet-ship certification, biological impact assessment, legal compliance determination or validated propeller design.

## 1. Paradigm shift: propulsion performance is multi-dimensional

A vessel should not be optimized for fuel and cargo service while treating underwater radiated noise as an externality to be measured later. Propeller cavitation, hull/propeller/rudder interaction, machinery state, WAPS force balance and operating policy can jointly determine both propulsion energy and underwater acoustic emissions.

The vessel-level design objective should therefore preserve a Pareto frontier across:
- delivered cargo work, arrival service and availability;
- fuel, primary energy and well-to-wake GHG;
- shaft power and propulsive efficiency;
- underwater radiated-noise spectra and uncertainty;
- lifecycle cost, maintenance and equipment life;
- route/location-specific environmental exposure, where independently supported.

No single scalar "quietness" score should replace frequency-resolved spectra. An overall dB change can conceal increases in biologically relevant bands.

## 2. Current standards and industry evidence

### 2.1 IMO underwater-noise work

The IMO Revised Guidelines for the Reduction of Underwater Radiated Noise from Shipping to Address Adverse Impacts on Marine Life are MEPC.1/Circ.906/Rev.1. At MEPC 84, the Committee agreed in principle to extend the experience-building phase to the end of 2028, commission a further IMO URN study and invite proposals for a policy roadmap ahead of MEPC 85.

The guidelines are management guidance, not a universal numeric quiet-ship limit or a ship-specific certification method. A noise measurement does not by itself establish biological impact.

### 2.2 ISO 17208 measurement family

- **ISO 17208-1:2016**, including Amendment 1:2024: prescribed-condition deep-water ship-noise measurements. It reports far-field RMS sound-pressure measurements normalized to 1 m as radiated noise level in one-third-octave bands. The standard is a measurement method, not a biological-impact criterion.
- **ISO 17208-2:2019**: derives an equivalent monopole source level from eligible deep-water measurements, with a specified nominal source depth and broadside aspect. Source level is not interchangeable with the directly measured radiated-noise level.
- **ISO 17208-3:2025**: shallow-water measurement and source-level procedures that account for shallow-water propagation; its radiated-noise reporting uses decidecade bands.
- **ISO 7605:2025**: ambient underwater sound measurement and reporting, including hydrophone calibration/deployment, acquisition and processing. Ambient sound is distinct from vessel self-noise.

The selected standard part, water depth, band convention, test geometry, calibration, operating condition and reported quantity must travel with every result. Deep water is defined as depth greater than max(150 m, 1.5 × overall ship length); shallow water is less than that threshold. A test exactly on the boundary is not accepted by either strict inequality and must be treated as method-selection ambiguity. Do not combine deep-water and shallow-water results without a declared method to transform them.

### 2.3 Recent propeller-noise research

Gaggero et al., *Propeller design by optimisation for underwater radiated noise reduction: a full scale application*, Ocean Engineering (2025), DOI 10.1016/j.oceaneng.2025.121412, report a propeller designed for URN reduction, model-scale validation and a full-scale refit/test on a small passenger vessel. This is a useful full-scale process precedent, not a cargo/WAPS performance baseline.

Khraisat et al., *Cavitation underwater radiated noise methodology applied to a propeller operating in-behind condition*, Ocean Engineering 349 (15 March 2026), DOI 10.1016/j.oceaneng.2025.124084, compare predicted pressure pulses/URN methods with model-scale experiments and full-scale sea trials. Their reported work highlights sensitivity of permeable FW-H integration to integration surface and rotating/stationary interface errors, and investigates cavitation-volume dynamics as a monopole source. This is a solver-methodology comparator, not universal solver validation.

## 3. Testable research hypotheses

**H1 — Installed propulsor noise differs from isolated propeller noise.** Hull wake, rudder, propeller loading, machinery transients and WAPS side force/leeway alter inflow, cavitation and acoustic output. Validate isolated components and installed configurations separately.

**H2 — Energy/noise trade-offs are condition-specific.** A propeller geometry or RPM schedule that reduces noise at one speed/load may increase fuel, vibration, or noise in another band or condition. Score the whole operating envelope.

**H3 — WAPS changes propeller loading rather than providing a free acoustic benefit.** Wind-assist can reduce shaft load in some conditions, but it may also change yaw/leeway/rudder demand and propeller inflow. Measure commanded and achieved WAPS states, shaft power, RPM, rudder and environmental conditions together.

**H4 — Ambient sound and propagation affect interpretation.** A ship source-level change and a receiving-habitat exposure change are not the same claim. Keep ambient sound, propagation, source level and ecological sensitivity as separate evidence layers.

**H5 — Measurement uncertainty determines whether a claimed reduction is resolvable.** Report frequency-resolved expanded uncertainty, hydrophone calibration validity, background measurements, repeated passes and test-site conditions. A change smaller than combined uncertainty is not a demonstrated improvement.

## 4. Matched experimental ladder

For one vessel/service fingerprint, evaluate:

| ID | Configuration | Isolated question |
|---|---|---|
| A0 | Existing propeller/afterbody baseline | Reference acoustic and energy signature |
| A1 | Propeller-only geometry/operating variant | Propeller contribution |
| A2 | Propeller + rudder + afterbody co-design | Installed propulsor interaction |
| A3 | A2 + WAPS fixed achieved state | Wind-assist influence at matched conditions |
| A4 | A3 across WAPS operating states | Active, feathered, stowed, retracted, faulted |
| A5 | A4 across loading/speed/sea-state slices | Transfer across operating envelope |
| A6 | Route/season exposure analysis | Only after measured source and propagation evidence |

Hold constant or explicitly model vessel configuration, speed through water, shaft power/RPM, draft/trim, loading, propeller/rudder state, WAPS state, wind/wave/current, water depth, test-track geometry and ambient sound. Do not call two measurements "matched" merely because both occurred at nominal service speed.

## 5. Evidence contract and first Rust audit

The first acoustic evidence schema and Rust audit qualify:
- selected ISO standard part and measurement quantity;
- water-depth and ship-LOA applicability checks for deep-water versus shallow-water method selection;
- hydrophone identity and calibration record, calibration validity interval and frequency coverage;
- calibration certificate digest and rights/status;
- run/trace identity, acquisition window, sample rate and expected/observed/missing counts;
- frequency-band convention, ordered/non-overlapping bands and band-edge consistency;
- RMS SPL values, reference pressure and expanded uncertainty;
- operating conditions, shaft state, loading, WAPS state and service fingerprint;
- ambient-noise reference identity where available.

The audit checks the evidence contract. It does not recalculate proprietary standard procedures, certify the hydrophone, validate acoustic propagation, or decide biological impact. Synthetic fixtures must always remain INCOMPLETE.

## 6. Benchmarks and falsification criteria

All targets are research goals, not achieved results.

| Dimension | Initial target |
|---|---:|
| Hydrophone/calibration identity and validity | 100% resolution |
| Required trace and sample provenance | 100% coverage |
| Frequency band order/uniqueness | 100% structural validity |
| False standard-conformance dispositions | 0 |
| Undeclared operating-condition mismatches | 0 |
| Known-answer acoustic arithmetic | 100% |
| Noise-reduction claim | Exceeds combined expanded uncertainty in each claimed band/condition |
| Repeatability | Independent repeated runs reproduce the effect within the declared uncertainty model |
| Pareto reporting | 100% energy/noise/service outputs reported separately |

Do not set a universal dB target. Use the contract, relevant habitat context, frequency weighting and legal/owner criteria where those are explicitly supplied. An approximately 3 dB change corresponds to a factor of about two in mean-square pressure/intensity only when the same reference and measurement definition are used; it is not automatically a 3 dB ecological benefit.

### Negative controls

- Wrong ISO part for deep/shallow water condition.
- One-third-octave bands supplied to the shallow-water decidecade profile or vice versa.
- Hydrophone calibration expired, out of frequency range or for a different instrument.
- Duplicate run IDs, duplicate band centres or overlapping band edges.
- Invalid trace digest, invalid time window, zero sampling rate or mismatched sample counts.
- Spectrum without calibration coverage or with unknown uncertainty.
- Candidate/baseline with mismatched service, speed, shaft power, draft/trim or WAPS achieved state.
- Ambient measurement absent or not aligned; result must disclose the limitation rather than pretend ambient is zero.

## 7. Industry comparison and defensible advantage

ISO 17208 provides standardized source-measurement procedures. IMO provides a non-numeric management guideline and ongoing experience-building programme. Recent academic work demonstrates propeller-noise optimization and full-scale testing, while cavitation prediction research is still sensitive to model and numerical choices.

Luminous should not claim to replace these standards, test facilities or acoustic solvers. The differentiator should be a reproducible evidence thread linking the acoustic trace, hydrophone calibration, installed geometry, model/mesh, propeller/rudder/WAPS state, cargo service, shaft energy and uncertainty-qualified outcome.

A design only beats SOTA when independently measured improvements recur in held-out operating conditions, exceed combined uncertainty and do not conceal adverse spectral bands, fuel or service impacts.

## 8. Environmental scope boundary

A source-level measurement is not a marine-mammal impact estimate. A biological exposure study must separately model source depth/directivity, propagation, bathymetry, sound-speed profile, ambient sound, route/season, receiving location and species-specific auditory/ecological context. These must not be inferred from a ship's ISO 17208 report alone.

Biofouling and invasive-species risk are also distinct environmental dimensions; they deserve their own evidence protocol and must not be folded into the acoustic record.

## 9. Sources

- IMO Revised URN Guidelines MEPC.1/Circ.906/Rev.1: https://glonoise.imo.org/documents/2524
- IMO MEPC 84 underwater-noise outcome and experience-building extension: https://www.imo.org/en/mediacentre/meetingsummaries/pages/temporary-mepc-84th-session.aspx
- ISO 17208-1:2016 + Amd 1:2024: https://www.iso.org/standard/62408.html
- ISO 17208-2:2019: https://www.iso.org/standard/62409.html
- ISO 17208-3:2025: https://www.iso.org/standard/81321.html
- ISO 7605:2025: https://www.iso.org/standard/82844.html
- Gaggero et al., Ocean Engineering (2025), DOI 10.1016/j.oceaneng.2025.121412: https://doi.org/10.1016/j.oceaneng.2025.121412
- Khraisat et al., Ocean Engineering 349 (2026), DOI 10.1016/j.oceaneng.2025.124084: https://doi.org/10.1016/j.oceaneng.2025.124084

## 10. Claim ceiling

An audit pass means only that the supplied acoustic record satisfies its declared schema/semantic contract. It is not ISO certification, measurement validity, solver validation, biological impact assessment, quiet-ship status or regulatory approval. Every reported result must carry the standard part/revision, vessel/configuration, operating condition, water depth, band convention, calibration, ambient context, uncertainty, trace digest and excluded conditions.
