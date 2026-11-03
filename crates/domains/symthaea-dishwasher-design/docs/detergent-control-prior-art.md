# Dishwasher detergent-control prior-art screening

Status: preliminary public-source screening, 3 November 2026. This is not a patentability opinion, freedom-to-operate (FTO) analysis or legal advice. Patent abstracts and publication records do not establish claim scope, validity, enforceability or infringement.

## Current product-level incumbents

- **LG 2026 European platform:** QuadWash Pro micro-bubbles, AI SenseClean turbidity sensor across prewash/main-wash/rinse, adaptive temperature/rinse count/detergent amount (up to 32 combinations), Dynamic Heat Dry+ and a one-hour wash/dry programme.
  - https://www.lg.com/global/newsroom/news/home-appliance-solution/lg-electronics-to-showcase-new-dishwasher-lineup-at-eurocucina-2026/
- **Henkel Somat Smartwash (Germany launch 20 Oct 2026):** multi-component detergent/rinse-aid/salt/machine-care dispensing; temperature/movement sensing to identify cycle phase and adjust component release.
  - https://www.henkel.com/press-and-media/press-releases-and-kits/2026-10-20-henkel-launches-somat-smartwash-2258310
- **Miele G8 Diamond (announced 2 Sep 2026):** AutoVision camera/AI load and item recognition, AutoDos, load-specific cleaning and variable hot-air drying; DACH launch announced for April 2027.
  - https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm

These announcements are feature claims, not independent matched cleaning, resource or patent analyses.

## Illustrative patent-family screening

### Sensor-triggered, non-intrusive automatic dosing

**US20030116177A1 — Automatic dispensing system** (publication 26 June 2003). The University of Twente research record describes a non-intrusive device for fabric/dishwashing machines, with one or more product containers, a pump and sensor input from the operating process/machine that triggers product delivery.
- https://patents.google.com/patent/US20030116177A1/
- Research record: https://research.utwente.nl/en/publications/automatic-dispensing-system/

### Multi-component chemical release during programme sequence

**US8679257B2 — Method for adding a multi-component detergent additive in a water-leading household appliance**. The public patent summary describes repeated liquid supply to a multi-component additive during programme execution, including multiple points in the programme.
- https://patents.justia.com/patent/8679257

**US9687139B2 — Method and device for controlled dosing of treating compositions in washing machines**. The public summary discusses separate enzymatic and oxidizing compositions and phase-separated dosing to address ingredient incompatibility. The exact claims and jurisdictional family require attorney review.
- https://patents.justia.com/patent/9687139

**EP2558560B1 — Automatic dishwashing detergent composition**. The published document discusses multi-compartment pouches, different release profiles and timed component delivery; this is formulation/packaging prior art as well as a reminder that phase separation is not new by itself.
- https://data.epo.org/publication-server/rest/v1.2/patents/EP2558560NWB1/document.pdf

### Conductivity/autocalibration and feedback-based concentration control

**US20180199790A1 / US11399691B2 — Autocalibrating dosing method.** The published application describes an initial detergent dose, measurement and storage of conductivity at a first loading, then dosing subsequent recovery quantities until the stored conductivity threshold is reached.
- https://patents.google.com/patent/US20180199790A1/en

**WO2025190615A1 — Dosing control device and dosing control method for a dishwasher** (published 18 September 2025; Winterhalter applicant/owner record). A public patent-index summary describes iterative real-time concentration measurement and dosing control.
- https://patents.google.com/patent/WO2025190615A1/en

### Bulk multi-additive storage, pump control and turbidity feedback

**US20250072705A1 — Bulk additive dispensing system for a dishwasher appliance.** The published application describes multiple additive chambers and a pump assembly, with a turbidity sensor and controller in communication with the dosing system.
- https://patents.google.com/patent/US20250072705A1/en

These examples demonstrate that automatic dosing, sensors, phase-specific release, component separation and conductivity/turbidity feedback have extensive prior art. They are not an exhaustive list and do not establish the current status of any claim in every territory.

## What InstantClean must not claim as novel without further work

- Automatic detergent dosing.
- Sensor-triggered or phase-aware dosing.
- Conductivity-based detergent concentration feedback.
- Multiple ingredient reservoirs, timed delivery or component separation.
- Turbidity-driven adaptation.
- AI/camera-based load recognition.
- Microbubble spray or multi-arm washing by itself.

The current `chemistry::validate_dose_plan` module is a software **validation contract**: it checks a proposed schedule against calibration ranges, phase timing, temperature, hardness, dose ceilings and declared interactions. It does not create a chemical formula, invent a dose, measure dispense accuracy, control a pump or claim a novel dosing mechanism.

## Possible differentiators to investigate (not novelty conclusions)

1. **Independent outcome qualification:** component-level delivery telemetry tied to physical ground-truth cleaning, rinse carryover and drying outcomes, with holdout evidence and fail-closed scope.
2. **Signed formulation calibration manifest:** a machine-readable, supplier-signed and versioned envelope for exact component batches, supported soils/materials, stage temperatures, hardness, dose ranges, inter-component constraints, expiry/revocation and device compatibility. Keep proprietary formula composition private; disclose only validated use constraints.
3. **Interoperable vendor-neutral recipe envelope:** permit multiple appliance manufacturers to consume authenticated formulation profiles without sharing proprietary full recipes or requiring a cloud service. A limited API review surfaced Home Connect programme/option interfaces, not a public cross-brand detergent calibration-manifest standard; this is not proof that none exists.
4. **Verified resource claims:** measure energy, fresh water and chemical mass per successful clean against LG SenseClean and Somat Smartwash at matched load/soil/finish boundaries.
5. **Privacy and resilience:** local manifest verification, no-camera use path, no raw image retention by default, safe fallback for sensor fouling, expired calibration or missing vendor signature.

## Required FTO work before product commercialization

- Search relevant patent families by claims, not just title/abstract, across South Africa, EU, UK, US, China, Japan and intended launch markets.
- Map independent claims to the proposed hardware, dispensing mechanisms, phase control, sensor inputs, formulation manifests and software behaviour.
- Check priority, continuations/divisionals, grant/publication status, expiry, assignments and prosecution history from authoritative patent-office registers.
- Review licensing opportunities and design-around alternatives with qualified patent counsel.
- Separate FTO, patentability and defensive-publication strategies. A published application can be relevant prior art even if no granted enforceable claim is identified in this quick scan.

Related issue: #10216 (phase-resolved chemistry benchmark) and the FTO/interoperability research issue to be assigned after this screening.
