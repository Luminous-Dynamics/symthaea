# MARINE-WIND-ENERGY-MRV-001
## Traceable wind-assisted energy measurement and proposed MRV readiness

**Status:** research specification v0.1.0  
**Created:** 10 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** #11469 regulatory route accounting; #11430 trial evidence; #11493 causal identifiability; COM(2026)620 / procedure 2026/0210(COD).

> This work prepares a traceable measurement/evidence architecture for a pending EU legislative proposal. It is not an adopted reporting obligation, does not create the future implementing standard, and is not a verifier or legal compliance decision.

## 1. Why this is a strategic opportunity

The European Commission's COM(2026)620 proposal, published 17 July 2026, would amend Regulations (EU) 2015/757 and (EU) 2023/1805. In its current parliamentary procedure (2026/0210(COD)), the proposal remains **awaiting committee decision** as of 10 November 2026.

The proposal would align ship GHG and energy monitoring under one MRV system, one monitoring plan and one ship report. Its proposed "energy used on board" metric is expressed in GJ and covers propulsion plus onboard equipment at sea and at berth. It identifies wind-assisted propulsion systems (WAPS) as substitute energy sources and defines wind assistance to include rotor sails, kites, rigid/soft sails, suction wings and turbines.

Most importantly, proposed Article 5(4) and the Annex I amendment would empower the Commission to specify, through future implementing acts, international standards and certification references to determine the amount of energy used from wind power. The proposed Annex I treatment keeps energy used from substitute sources distinct from fuel energy; it is not yet a final formula. **Those standards and the final legal text are not yet in force or specified by this proposal.**

That creates an opportunity: build the auditable measurement and provenance infrastructure now, but keep any future regulatory calculation method pluggable and versioned. Do not invent the future method.

## 2. The central distinction: three different quantities

The system must keep three results separate:

1. **Wind-energy amount reported under a specified method** — the energy quantity determined using the applicable international standard and certification references once they exist and apply. Until then, its regulatory status is NOT_DETERMINED.
2. **Research estimate of useful wind contribution** — an engineering estimate based on declared physical observations, model, assumptions and uncertainty. This can be useful for design/control research, but is not regulatory wind-energy evidence.
3. **Counterfactual avoided propulsion energy/fuel** — the estimated difference between actual propulsion energy/fuel and a defined counterfactual without WAPS. This is a causal/model-based effect, not the same as energy used from wind.

Do not substitute an avoided-fuel estimate or an aerodynamic force-times-speed calculation for a future regulatory wind-energy method unless an applicable approved standard explicitly defines that calculation and its boundaries.

## 3. Evidence architecture

### 3.1 Configuration and device identity

Bind every result to:
- vessel/configuration revision and digest;
- route/service fingerprint and reporting window;
- device type (rotor sail, kite, rigid/soft sail, suction wing, turbine or other declared type);
- device serial/identity, geometry revision, foundation/interface revision and control-software revision;
- operating state (deployed, active, trimmed, feathered, furled, stowed, retracted, unavailable, faulted);
- commanded state and achieved/observed state separately.

### 3.2 Environmental and vessel observations

Record time-aligned:
- apparent and true wind vector, angle, speed, gust and height/reference point;
- vessel heading, speed through water and speed over ground;
- wave height, direction/period, current and water depth where available;
- draft, trim, loading condition, rudder angle and propulsion state;
- WAPS geometry/trim/control state and available device telemetry.

These observations establish context and support physics/research analysis. They are not, by themselves, the prescribed regulatory wind-energy calculation.

### 3.3 Energy consumers and substitute sources

Keep fuel/energy consumers distinct:
- main propulsion engines/generators;
- auxiliary and hotel loads;
- boilers and other fuel consumers;
- shore power;
- renewable energy generated onboard;
- wind-assistance energy quantity, when determinable under the applicable method.

Capture units, measurement boundary, fuel batch/pathway, lower-heating-value basis, equipment maps, converter losses, calibration, timebase, sampling completeness and rights-aware provenance. Do not silently combine fuel, electrical input, shaft output and source-energy terms.

### 3.4 Future method/certification plug-in

A standard-derived or certified wind-energy record must identify:
- exact standard/international reference, edition and applicable clauses;
- implementing act reference and effective date, once issued;
- calculation method, revision, software/model digest and input boundary;
- device type/configuration and validated applicability envelope;
- certification body/scheme, certificate ID, revision, validity window and digest;
- verifier/attestation identity and reporting period.

Until those references exist, keep the WAPS energy quantity NOT_DETERMINED for regulatory reporting. A separate RESEARCH_ESTIMATE may be stored with its own method and uncertainty, but must not be represented as certified.

## 4. No-double-counting and energy closure

Maintain at least three distinct ledgers:
- fuel energy consumed by each onboard fuel consumer;
- electricity/other substitute-source energy used onboard, with origin and losses;
- WAPS-derived wind-energy quantity under the specified future method.

Do not count the same energy twice at different boundaries. If a mechanical or electrical conversion path is represented, identify whether the quantity is source energy, delivered energy, shaft energy or useful propulsion work. The total GJ reporting boundary must follow the final regulation/methodology; this specification does not pre-judge it.

For hybrid vessels, separately track initial/final battery SOC, charge/discharge energy, SOH, temperature and conversion losses. A battery discharge cannot be credited as a wind saving. WAPS avoided-fuel estimates must use matched/causally qualified counterfactuals and cannot replace the energy-source ledger.

## 5. Versioned evidence data contract

A separate v0.1.0 schema and synthetic fixture are created alongside this specification:
- marine-wind-energy-evidence-v0.1.0.schema.json
- fixtures/synthetic-wind-energy-evidence-v0.1.0.json

It binds:
- proposal identity and proposal-only status;
- vessel/service/configuration and reporting period;
- WAPS device identities and achieved states;
- observation channel/calibration/timebase/raw-trace evidence references;
- consumer and substitute-source energy ledgers;
- wind-energy determination status and method provenance;
- separate research estimates and counterfactual avoided-energy estimates;
- limitations, source rights, exclusions and audit trail.

The schema is a research contract, not a new statutory format. It must evolve when the legal text, implementing acts, international standard or certification scheme is finalized.

## 6. Research/engineering experiments

1. Establish a device-state telemetry and measurement map for each WAPS family.
2. Define test-bench and sea-trial instrumentation: wind vector, device state/force/moment, shaft torque/RPM, fuel flow, electrical bus, battery SOC and vessel motion.
3. Quantify timing, calibration, sampling and uncertainty contributions end-to-end.
4. Compare research estimates against independently measured aerodynamic force/moment and shaft/fuel outcomes without calling the estimate regulatory WAPS energy.
5. Design an adapter interface for future implementing-act methods; freeze all method inputs, standards and revisions.
6. Run causal qualification separately for avoided-fuel claims through #11493.
7. Maintain regulatory change monitoring for COM(2026)620 / 2026/0210(COD), including committee amendments and future implementing acts.

## 7. Proposed benchmarks

Research targets, not regulatory requirements or achieved results:
- 100% of energy quantities identify source, unit, physical boundary, method and evidence class.
- 100% WAPS device states bind commanded vs achieved state and configuration revision.
- 0 research counterfactuals labelled as standards-derived or certified.
- 0 double-counted source/conversion energy in known-answer ledger fixtures.
- 100% certificate/standard/method revision resolution for any record labelled certified.
- 100% timebase/calibration/sampling/evidence-reference resolution for instrumented trial inputs.
- 0 silent unknown-to-zero conversion.
- Reproducible replay from raw trace digests, calibration/method inputs and exact revisions.

## 8. Competitive advantage

The strongest opportunity is not to invent a wind-energy regulatory formula before the Commission does. It is to make Luminous the engineering data/evidence layer that can:
- ingest device and ship measurements across WAPS families;
- maintain calibrated, time-aligned evidence and configuration lineage;
- separate measured source energy, engineering estimates and counterfactual savings;
- replay the future standards method exactly once published;
- produce a reproducible, verifier-ready evidence package without claiming verifier approval.

This complements DLR Odyssa, specialist CFD and WAPS suppliers rather than replacing their physics models.

## 9. Sources and current status

- European Commission, COM(2026)620 final, 17 July 2026: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52026PC0620
- European Parliament Legislative Observatory, procedure 2026/0210(COD): https://oeil.europarl.europa.eu/oeil/en/procedure-file?reference=2026%2F0210%28COD%29
- European Commission Annex to COM(2026)620 (proposed Annex I method and Article 5(4) wind standard/certification hook): https://publications.europa.eu/resource/cellar/bef58239-81f4-11f1-bf5e-01aa75ed71a1.0001.03/DOC_2
- EU MRV Regulation (EU) 2015/757: https://eur-lex.europa.eu/eli/reg/2015/757/oj/eng
- FuelEU Maritime Regulation (EU) 2023/1805: https://eur-lex.europa.eu/eli/reg/2023/1805/oj/eng

As of 10 November 2026, procedure 2026/0210(COD) remains awaiting committee decision. No final wind-energy determination standard or certification implementation is represented here.

## 10. Claim ceiling

A research estimate is not a statutory wind-energy quantity. A method-derived quantity is not certified unless the applicable standard, implementing act and certificate are resolved and valid. A schema/audit pass is not a verifier statement, legal compliance determination, fuel-saving result, physical qualification or ship approval.
