# Luminous Aqua — Site Fit and Installation Envelope Protocol v0.1

**Status:** evidence-scope contract and installation eligibility method; no site or product has been qualified  
**Workstream:** [AQUA-SITE-FIT-001 #12881](https://github.com/Luminous-Dynamics/symthaea/issues/12881)  
**Architecture comparison:** #12875 / draft PR #12878  
**Service qualification:** #12865 / PR #12867  
**Wet-path design and biological methods:** #12703  
**Implementation:** \`symthaea-aqua-bench::site_fit\`

## 1. Decision problem

A product can perform well on a laboratory bench and still be unsuitable for a particular building. The site may not provide the exact dynamic pressure, hot/cold inlet capacity, water chemistry, electrical supply, continuous power, drain fall, core location, service access or maintenance capability required by the selected configuration.

The sequence must therefore be:

\`measured site evidence
 -> architecture-specific installation eligibility
 -> matched product/service trial
 -> independent completed-service qualification
 -> site-specific Pareto comparison\`

This protocol introduces an evidence gate before architectures enter the Pareto frontier. It does not select a winner, calculate engineering thresholds or replace local professional approval.

## 2. Paradigm shift: from product recommendation to evidence-backed eligibility

The result for one exact architecture, device configuration and site profile is one of three states:

- **Eligible:** every required prerequisite has a current, correctly scoped, positive assessment from the expected authority.
- **Ineligible:** at least one current, correctly scoped prerequisite is explicitly Unsatisfied.
- **Indeterminate:** required evidence is missing, stale, revoked, ambiguous, wrongly scoped, method-mismatched or independently unresolved.

The result carries domain-level reasons. There is no weighted score, ranking, or "best shower" recommendation.

A known Unsatisfied prerequisite makes the architecture Ineligible even when another capability remains unresolved. When no valid Unsatisfied result exists, any unresolved required domain makes the overall result Indeterminate. An unknown or absent value is never treated as zero, average, default or satisfied.

## 3. Architecture-specific requirements

Each architecture has a fixed mandatory domain set. An exact product configuration may add prerequisites, but cannot remove baseline requirements.

| Architecture | Mandatory additional requirements | Product-fit interpretation |
|---|---|---|
| Direct-flow passive | Dynamic supply, water temperature, chemistry/scale, drain/waste, installation space/routing, service/maintenance | No electrical, continuous-power, recirculation hygiene or dual high-capacity inlet gate is imposed by this architecture class |
| Direct-flow + indirect DWHR | Direct-flow baseline plus drain geometry, installation space and thermal receiver/heat-sink compatibility | A drain exchanger is only eligible where useful receiver-side heat can be delivered and the physical drain/plant-space layout supports the exact configuration |
| Power-modulated direct-flow + indirect DWHR | DWHR prerequisites plus electrical supply/protection and continuous power availability | Power modulation is not treated as a passive feature; exact electrical and outage requirements must be assessed |
| Same-session recirculation | Dynamic supply, per-inlet supply capacity, water temperature, chemistry/scale, electrical supply/protection, continuous power, drain/waste, installation space/routing, service/maintenance, wet-path hygiene/cleaning, backflow/cross-connection protection | Reservoir, filter, pump, cleaning, retained water and same-session water-quality method are separate eligibility domains |
| Same-session recirculation + indirect DWHR | Recirculation baseline plus thermal receiver/heat-sink compatibility | Both recirculation-specific and DWHR-specific requirements must pass independently |

A product-specific requirement such as network connectivity may be added when the exact product configuration requires it. Connectivity is not imposed on every architecture by default. The same principle applies to softeners, pumps, dedicated electrical circuits, special ventilation and other installation dependencies.

## 4. Site capability domains and authorities

The module assigns expected owners. This is an evidence-routing contract, not a claim that any owner has already issued a site assessment.

| Capability domain | Expected authority |
|---|---|
| Dynamic supply pressure/flow envelope and per-inlet capacity | Symtropy hydraulic authority |
| Water-temperature envelope and thermal receiver compatibility | Symtropy thermal authority |
| Water chemistry and scale | Mycelix Water Quality |
| Electrical supply/protection and continuous-power availability | Competent independent electrical assessor |
| Drain/waste geometry and installation space/routing | Competent installation surveyor |
| Backflow and cross-connection protection | Competent plumbing assessor |
| Service and maintenance capacity | Independent serviceability assessor |
| Wet-path hygiene and cleaning | Independent hygiene/exposure assessor under #12703 |
| Product-specific network/site infrastructure | Independent site infrastructure assessor |

Each owner remains responsible for the measurand, sampling/measurement method, validity interval, uncertainty, acceptance envelope and actual Satisfied / Unsatisfied / Indeterminate decision. Aqua only checks evidence identity, scope, completeness, authority mapping and decision state.

## 5. Current competitor installation envelope: Orbital Shower

The current Orbital installation requirements page is an instructive example of why a site-fit gate is necessary. The manufacturer currently publishes:

| Site prerequisite | Manufacturer declaration |
|---|---|
| Fixed installation | Residential/commercial fixed environments; not mobile installations such as RVs, trailers or boats |
| Power continuity | Must remain powered at all times, including while not in use |
| Operating ambient | Core environment 15–35°C during operation; not below 4°C while idle |
| Core/service access | Core location must permit routine maintenance and technician access |
| Drain geometry | Core no more than 3 m from drain centre |
| Water temperature | Cold 4–25°C; hot 50–65°C |
| Inlet pressure | Declared 2–10 bar; maximum working pressure PN10 |
| Flow capacity | For 12 L/min system flow: 9 L/min at each hot/cold inlet; lower system-flow bands list proportionally lower inlet capacities |
| Water chemistry | Hardness 53–250 mg CaCO₃/L; iron below 0.3 mg/L |
| Electrical | Single phase 230 VAC, 50–60 Hz, nominal 16 A; lower fuse ratings are said to reduce flow |
| Floor geometry | Manufacturer recommends a 20 mm/m slope toward the drain |
| Tile constraint | Shower-zone tile thickness maximum 10 mm |

The manufacturer page also lists an expected pressure/flow table with a 1.5 bar point (8.7 L/min), despite stating that the designed incoming pressure range begins at 2 bar. This remains an unresolved scope discrepancy. Do not silently treat the 1.5 bar data point as an approved minimum operating pressure. Ask the manufacturer to clarify the exact configuration, pressure datum, test condition and supported status.

These are manufacturer installation declarations/recommendations, not independent measurements, building-code approvals or proof that a South African site is suitable.

Source: https://support.orbital-systems.com/en/19639-installation-requirements.html

### Implication for Aqua

The site-fit model must capture at least: measured dynamic inlet conditions; separate hot/cold capacity where required; continuous electrical availability; idle/operating ambient conditions; drain location and levels; plant-room/core access; floor fall; exact wall/tile constraints; water hardness/iron; and the maintenance capability to support filters, cleaning and service.

Aqua must not assume all Johannesburg or South African installations share a common pressure, hardness, temperature, tariff, electrical service or plumbing layout.

## 6. Recirculation adds a separate hygiene/exposure gate

The recirculating-shower evidence base must not be reduced to "a filter exists" or "a cleaning cycle runs".

A 2025 review of microbial indicators for recirculating shower technology notes that the water-quality evidence base remains limited and identifies fecal indicators and waterborne pathogens as relevant to monitoring and public-health assessment. It discusses supplementary parameters such as Staphylococcus aureus, Mycobacterium spp. and fungi, and trace organisms/model viruses for evaluating treatment efficacy. The review does not provide a universal product acceptance threshold for Aqua.

A 2026 review of opportunistic pathogens in shower systems recommends integral assessment across water, biofilms and aerosols and across components including hoses, showerheads and pipes; it discusses Legionella, Mycobacterium and Pseudomonas. Its implication for this protocol is that a water-only site sample cannot qualify aerosol or surface/biofilm outcomes.

These method and organism decisions belong to Mycelix / competent independent laboratories and exposure specialists under #12703. Site Fit only requires a current scope-matched assessment to exist. It does not select pathogen panels, infer health risk, or determine safe exposure.

Sources:
- Pérez Guillemette and Albrechtsen, *Science of the Total Environment* (2025): https://doi.org/10.1016/j.scitotenv.2025.178610
- Ren et al., *Journal of Environmental Sciences* (available online 30 May 2026): https://doi.org/10.1016/j.jes.2026.02.034
- Wet-path engineering and independent test ownership: #12703

## 7. Evidence identity and invalidation

Every current site capability receipt binds:

- exact site profile ID and revision;
- site-condition SHA-256;
- exact architecture profile ID and architecture enum;
- exact device-configuration SHA-256;
- exact architecture-requirements SHA-256;
- capability domain and expected authority;
- issuer principal ID and canonical authority-verification receipt SHA-256;
- evidence record ID and evidence SHA-256;
- method SHA-256 and method-condition SHA-256;
- Current / Stale / Revoked / Ambiguous currentness state;
- Satisfied / Unsatisfied / Indeterminate outcome.

A new site survey, changed pressure/flow or water chemistry, changed pipework/drain, electrical modification, device or installation revision, changed requirement envelope, or changed method scope requires the affected evidence to be refreshed.

The canonical owner must authenticate the issuer and issue the verification receipt. The Rust module checks receipt presence, digest syntax, expected authority and declared currentness. It does not implement cryptographic signature verification, trusted registry lookup, or live revocation checks.

## 8. State machine and claim ceiling

- Missing assessment or evidence -> Indeterminate.
- Stale, Revoked or Ambiguous evidence -> Indeterminate.
- Wrong site, site revision, condition digest, architecture, device digest, requirement digest, method or method conditions -> Indeterminate.
- Wrong authority -> Indeterminate.
- Valid Unsatisfied result -> Ineligible.
- Valid Indeterminate result -> Indeterminate.
- Every required result valid and Satisfied -> Eligible.

Only Eligible configurations may proceed as eligible entries in #12875's site-specific comparison. Eligibility does not mean physically installed, safe, compliant, performant or cost-effective. A qualified service result must still pass #12867's independent completed-service gate, and measured Pareto outcomes must still be generated under a matched test protocol.

No site fit or service qualification is a substitute for competent local plumbing/electrical/building approval, product certification, microbiological clearance, human-factors assessment or physical commissioning.

## 9. Software implementation status

The Rust module defines five architecture classes, thirteen capability domains, expected-authority mapping, mandatory baseline requirements, product-specific additive requirements, site/device/requirement/method digest binding, currentness handling, and Eligible / Ineligible / Indeterminate dispositions.

It currently contains **43 authored unit tests** covering architecture-specific sets, missing/duplicate/unexpected evidence, stale/revoked/ambiguous evidence, wrong site/revision/architecture/device/requirements/method/authority, explicit failure vs uncertainty, additional product-specific connectivity, and site-to-site variation.

These are authored tests, not execution passes. No cargo test, rustfmt, Clippy or repository CI pass is claimed until observed on the exact PR head.

## 10. Decision gate

Before a site is considered eligible:

1. Freeze the exact site profile and survey revision.
2. Identify the exact market variant, hardware, installation, firmware and service configuration.
3. Freeze the declared requirement envelope and its revision.
4. Obtain current method-scoped assessments from the proper authorities.
5. Reconcile any manufacturer documentation scope ambiguity before claiming the relevant operating point.
6. Require local competent electrical, plumbing, building and installation approval.
7. Re-evaluate eligibility after any relevant site/device change.
8. Only then include the configuration in #12875's architecture-specific matched test plan.

No physical prototype, occupied installation, human exposure or actuator authorization follows merely from this software gate.
