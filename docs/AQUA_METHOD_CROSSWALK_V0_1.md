# Luminous Aqua — Shower Test-Method Crosswalk v0.1

**Status:** method crosswalk and schema boundary; not a compliance determination  
**Workstream:** [AQUA-METHOD-CROSSWALK-001 #12817](https://github.com/Luminous-Dynamics/symthaea/issues/12817)  
**Related:** #12784 low-pressure; #12772 spray geometry DOE; #12692 spray-field outcomes; #12816 electric shower / DWHR

## 1. Core rule

A pressure value is not fully identified by its number and unit. It must carry:
- exact product category and specimen / market variant;
- test method and edition;
- pressure datum / measurement location;
- number of pressurized inlets and whether pressures are applied simultaneously;
- dynamic versus static state;
- pressure tolerance and measurement uncertainty;
- flow / temperature condition;
- test fixture, orientation and stabilized / transient window.

Aqua must not compare 150 kPa at a terminal fitting inlet with 150 kPa applied to each hot/cold mixer inlet or 150 kPa at a faceplate centre as if they were the same measurement.

## 2. Current method references

### 2.1 EN 18021:2025 — functional-performance type test

**Status:** current; entered into force 2 July 2025 in the UNI catalogue. The standard summary states that its tests are laboratory type tests, not production quality-control checks.

The scope distinguishes PN10 taps / shower systems from PN5 shower outlets / shower sets. It excludes shower taps alone, bath-filling taps and the bath-filling function of combined taps, among other excluded products.

The accessible Table 1 summary for taps and shower systems lists:
- dynamic-pressure lower operating limit: 0.05 MPa (0.5 bar);
- static-pressure maximum: 1.0 MPa (10 bar);
- recommended dynamic pressure: 0.1–0.5 MPa (1–5 bar), with the footnote stating the recommended dynamic value is measured at the point of discharge;
- water temperature up to 70°C only for short periods not exceeding one hour, recommended up to 65°C.

The accessible Table 2 summary for shower outlets and shower sets lists:
- dynamic-pressure lower operating limit: 0.05 MPa (0.5 bar);
- static-pressure maximum: 0.5 MPa (5 bar);
- recommended dynamic pressure: 0.1–0.3 MPa (1–3 bar);
- water temperature up to 70°C only for short periods not exceeding one hour, recommended up to 42°C.

These category-specific limits must not be transferred between outlets/sets and taps/systems. The standard's full test procedures, conditions and tolerances must be taken from the controlled edition; the public catalogue summary is not a substitute for the standard.

Sources:
- UNI official catalogue: https://store.uni.com/en/en-18021-2025
- SIS current national adoption listing: https://www.sis.se/en/produkter/testing/ss-en-180212025/
- BS EN 18021:2025 listing: https://tienda.aenor.com/p/norma-bsi-bs-en-18021-2025-000000000030479514

### 2.2 Unified Water Label (UWL) Taps & Showers Technical Criteria, January 2023

UWL Table 1 gives the following **dynamic pressures applied to each inlet**:

| System / point | Applied pressure | Meaning |
|---|---:|---|
| Type 1 — P1 | 0.15 MPa +0.01/-0 MPa (150 kPa +10/-0) | Lower pressure point |
| Type 1 — P2 | 0.30 MPa +0.02/-0 MPa (300 kPa +20/-0) | Reference pressure point |
| Type 2 — P3 | 0.01 MPa +0.002/-0 MPa (10 kPa +2/-0) | Low-pressure installation point |

The document defines these as pressures applied to each inlet. They are not one upstream residual-pressure reading and not the same datum as an EN point-of-discharge channel. For a mixer, preserving hot and cold inlet channels separately is necessary; a single scalar pressure field cannot represent simultaneous two-inlet conditions.

The UWL document also requires calibrated measurement equipment, and its product-registration maximum flow represents a declared rating subject to the scheme's performance tolerances.

Source: https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf

### 2.3 UK Energy Technology List (ETL) — non-domestic shower programme

The ETL category is explicitly for non-domestic applications. It uses UWL procedures for flow, pressure independence and spray coverage.

Its flow-rate procedure requires P2 to be reached at the centre of the shower faceplate before stabilized flow is recorded. Its pressure-independence criterion requires at least 60% flow at 1.5 bar relative to maximum available flow at 3 bar, only where the 3-bar reference flow is 8.0 L/min or less.

ETL therefore combines multiple method conditions: UWL's pressure-applied-to-inlet procedure and a faceplate-centre P2 requirement for flow-rate measurement. Do not reduce this to one generic pressure datum.

The ETL's 4.5 / 9 / 10.6 L/min electric-shower pathways are conditional programme criteria and are non-domestic UK requirements, not South African law.

Source: https://etl.energysecurity.gov.uk/products/showers/showers

### 2.4 EPA WaterSense v1.1 / ASME-CSA methods

WaterSense uses specific pressure points for flow and spray-force testing, including nominal 140, 310 and 550 kPa with method-specific tolerances. Its pressure-compensation criteria include minimum flow fractions at specified points; its force and spray-coverage criteria use incorporated standard test procedures.

These points are not interchangeable with SANS installation design pressures or UWL's P1/P2/P3 points. Formal WaterSense claims require the full specification, current incorporated ASME/CSA method and its specimen / apparatus rules.

Source: https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt

### 2.5 South African SANS 10252-1:2018 Edition 3.2 — installation design, not product performance testing

Table 3 is an installation design-demand reference:
- "Likely maximum" is the flow a user would normally allow a fitting to discharge, not necessarily the maximum attainable.
- The design-flow-pressure column is used where the manufacturer's flow-pressure curve is unavailable.
- The water-saving showerhead row lists 10 L/min design flow at 50–100 kPa, but this is not a universal minimum operating-pressure rating for every showerhead.
- The separate amended shower-discharge clause is retained as a 10 L/min maximum pending controlled-standard confirmation.

Use the controlled SABS publication and confirm project applicability with the responsible professional before a compliance decision. An IOPSA notice is a secondary summary, not the controlled standard.

## 3. Aqua engineering pressure points

Aqua's current low-pressure protocol uses 50 / 75 / 100 kPa as research points motivated by the local SANS design-flow reference and the EN 18021:2025 lower operating-pressure range. These are not automatic pass points for every product.

Other points have separate purposes:
- 140 ± 7 kPa: WaterSense spray-force / low-pressure reference where the method requires it;
- 150 kPa: UWL P1 nominal pressure and Aqua paired pressure-retention lower point, but UWL formally applies it to each inlet;
- 300 kPa: UWL P2 nominal reference and Aqua paired upper point, again with different exact datum semantics;
- 310 ± 7 kPa and 550 ± 7 kPa: WaterSense flow pressure points;
- 50 / 75 / 100 kPa: SANS / EN low-head engineering tests; not UWL P3 (10 kPa).

Never describe Aqua's 150/300 kPa pair as a UWL conformance test unless the full UWL apparatus, separate inlet pressures, tolerances, temperatures and procedures have been implemented. It is a generic matched engineering comparison when the exact recorded datum, specimen and protocol are the same.

## 4. Software representation boundary

The dependency-free pressure_curve module now records an explicit PressureMeasurementDatum for each dynamic pressure observation:
- TerminalFittingInlet
- PointOfDischarge

Matched exact-point comparisons and two-pressure flow-retention comparisons reject different datums. Manufacturer-declared pressure ranges also carry an optional source datum; if the source does not state one, or states a different one, the helper reports DeclaredDatumNotSpecified / DeclaredDatumMismatch instead of inventing a match.

**Deliberate limitation:** the generic single-pressure-channel record does not represent UWL's simultaneous hot/cold inlet channels or ETL's faceplate-centre P2 channel. Those require separate method-specific channel records and apparatus evidence. The generic helper must not be used to claim UWL or ETL conformity.

## 5. Method-specific data needed before formal conformity

A method-specific test record must additionally bind:
- method name, edition, section and controlled protocol digest;
- product category (outlet, set, tap, shower system or electric shower);
- inlet channel identities and independently measured pressure trace for each inlet where required;
- faceplate-centre pressure channel where ETL flow procedure requires P2 at that datum;
- point-of-discharge pressure channel where the adopted EN method defines it;
- static pressure and dynamic pressure channels as distinct observations;
- test water temperatures, outlet temperature and ambient conditions;
- specimen count, conditioning, mounting, orientation, stabilization and test sequence;
- calibration, uncertainty, sample result and exclusions.

Aqua's spatial deposition map is supplementary diagnostic evidence. It does not replace UWL, WaterSense or EN spray-coverage / functional performance methods.

## 6. Claim ceiling

This crosswalk is a method-analysis and software-contract artifact. It does not establish product compliance, certification, human comfort, hygiene or physical Aqua performance. Formal method testing requires the controlled standard, the full apparatus and appropriate independent or accredited test-lab review.
