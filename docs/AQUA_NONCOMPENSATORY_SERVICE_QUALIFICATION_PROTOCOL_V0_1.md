# Luminous Aqua — Non-Compensatory Completed-Service Qualification Protocol v0.1

**Status:** decision-contract specification and preregistration template; no physical result or product qualification exists  
**Workstream:** [AQUA-SERVICE-QUALIFICATION-001 #12865](https://github.com/Luminous-Dynamics/symthaea/issues/12865)  
**Parent:** integrated bathing service #12660  
**Implementation:** \`symthaea-aqua-bench::service_qualification\`  
**Related:** matched-service adapter PR #12662; spray field #12692; wet-path hygiene #12703; DHW cascade #12709; accessible experience #12712; circular serviceability #12727 / #12844

## 1. Decision problem

Aqua already has contracts for matched test context, method scope, pressure/flow evidence, spray-field observations, service continuity, task-phase plans and repairability receipts. The matched-service adapter checks that the expected owner receipts exist and that two runs share the declared test conditions.

That is necessary but not sufficient for a product-level decision. An evidence packet can be structurally complete while an independently assessed endpoint is failed or inconclusive. A low water or energy result must never compensate for a degraded bathing task, failed spray endpoint, unstable thermal delivery, invalid hygiene/exposure assessment, inaccessible operation or unqualified safety state.

The new decision boundary is:

\`frozen claim profile
  -> exact required assessment domains
  -> authority + method + conditions + evidence binding
  -> Pass / Fail / Indeterminate per domain
  -> Qualified / Failed / Indeterminate service result\`

The product is the **completed bathing service**, not the showerhead, heater, controller or drain heat exchanger in isolation.

## 2. Scope and non-duplication

The Rust module is intentionally a small evidence-decision contract. It:

- fixes which assessment domains are required for each declared claim;
- maps each domain to its permitted evidence authority;
- binds assessments to exact record identity, evidence digest, protocol digest and test-condition digest;
- rejects missing, duplicate, unexpected, wrong-authority, stale-scope or mismatched evidence;
- preserves Fail and Indeterminate as distinct outcomes;
- prevents a comparative claim when the control itself is not qualified;
- never calculates physical, statistical, resource, hygiene, safety, comfort or lifecycle outcomes.

Existing authorities remain unchanged:

| Domain | Evidence authority |
|---|---|
| Measurement validity / calibration admission | Mycelix Metrology |
| Task completion and comfort | Independent task assessor |
| Standardized spray-force / coverage performance | Independent test laboratory |
| Thermal stability | Independent test laboratory |
| Water-quality assessment | Mycelix Water Quality |
| Hygiene and exposure assessment | Independent hygiene / exposure assessor |
| Local product safety / release authority | Independent device authority |
| Accessibility and usability | Independent accessibility / human-factors assessor |
| Water and energy per completed task | Symtropy resource closure |
| Serviceability and cleaning burden | Independent serviceability assessor |
| Continuity and recovery | Independent device authority |

This mapping is a software ownership contract, not a claim that any named organization has already issued an assessment. The authority must be engaged and its record made available before a physical result can qualify.

## 3. Claim profiles

A claim cannot quietly omit an endpoint after results are known. The profile fixes the required domains before a test or comparison begins.

| Claim profile | Required evidence beyond the functional baseline |
|---|---|
| Functional service | None; requires all seven baseline domains |
| Accessible service | Accessibility and usability |
| Water efficiency | Water savings per completed task |
| Water-and-energy efficiency | Water and energy savings per completed task |
| Lifecycle efficiency | Accessibility, water, energy, serviceability and cleaning burden |
| Resilient service | Continuity and recovery |
| Full platform | Every domain, including all baseline and additional domains |

### Functional baseline — mandatory seven

1. Measurement validity.
2. Task completion and comfort.
3. Standardized spray performance.
4. Thermal stability.
5. Water-quality assessment.
6. Hygiene and exposure assessment.
7. Independent local-safety qualification.

The scope is intentionally claim-specific. A report may say that a configuration passed a functional-service profile. It may not silently upgrade that result to accessible, energy-efficient, lifecycle-efficient or resilient service.

The full-platform profile currently has thirteen required domains. The source-level enum fixes the profile-to-domain mapping; a caller cannot submit an ad hoc list to weaken the profile.

## 4. Evidence identity and decision semantics

Every assessment receipt binds:

- exact assessment record ID;
- evidence SHA-256;
- method / protocol SHA-256;
- test-condition SHA-256;
- domain-specific authority;
- one of three independent outcomes.

### Outcomes

- **Pass:** the competent assessor says the domain met its preregistered acceptance rule.
- **Fail:** the assessor says the domain did not meet that rule.
- **Indeterminate:** the evidence is insufficient, measurement uncertainty straddles the decision boundary, the method is not applicable, or the result cannot be resolved.

A missing receipt is a structural error. An explicit Indeterminate assessment is valid evidence of an unresolved outcome, but it is never a Pass.

### Aggregation rule

The module creates no average, weighted sum, utility function or compensating score.

- Any required Fail produces a Failed result.
- With no Fail, any required Indeterminate produces an Indeterminate result.
- Only an exact, complete set of Pass assessments produces Qualified.
- A matched candidate is not qualified for comparison unless the control and candidate both use the same claim profile, protocol and conditions, and the control itself is Qualified.

This is an evidence completeness and decision-scope rule. It does not calculate non-inferiority, significance, confidence intervals or savings.

## 5. Statistical comparison requirements

The independent assessor must preregister:

- the exact primary task and functional unit;
- exact comparator SKU, market variant, mode, installation and maintenance state;
- primary and secondary endpoints;
- non-inferiority margins for quality endpoints;
- superiority thresholds for the claimed resource endpoint;
- measurement uncertainty and uncertainty propagation;
- the statistical model, multiplicity treatment and missing-data rules;
- specimen, technician, day, water-batch and installation blocking;
- sample-size / power justification;
- pilot versus confirmatory data partition;
- treatment of failed, aborted, repeated and indeterminate attempts.

No universal numeric non-inferiority margin is invented in this protocol. It must be selected by the competent domain assessor before confirmatory results are unblinded. Repeated readings on one test article are not independent product specimens.

For a water-efficiency claim, the independent resource assessor must qualify **fresh water per completed task**, not nominal flow multiplied by a planned duration. Startup, warm-up, priming, pauses, boost, rinse, purge, cleaning, failed attempts and recovery must be inside the frozen accounting boundary.

For an energy-efficiency claim, the assessor must qualify purchased energy per completed task at the complete declared system boundary. Include heater / heat-pump energy, pumps, controls, standby, cleaning and other relevant auxiliaries. Recovered heat is credited only when usefully delivered, net of parasitics, temperature-grade mismatch, fouling, storage and distribution losses.

## 6. Industry and standards positioning

Aqua should not claim novelty from a low-flow mode alone.

### EPA WaterSense

The US EPA WaterSense specification caps labelled showerheads at 2.0 US gallons per minute (approximately 7.6 L/min) and includes performance requirements addressing spray force, spray coverage and pressure compensation. This is a useful independent-performance reference, not South African law and not proof that a particular Aqua mode is equivalent or certified.

Official programme: https://www.epa.gov/watersense/showerheads

### UK Energy Technology List — 2025/26 update

The current UK ETL shower criteria are explicitly for non-domestic applications. The September 2026 update retains a 4.5 L/min standard electric-shower maximum pathway and adds a conditional pathway up to 9 L/min when electrical input is modulated to reduce electricity while maintaining stable output temperature with compliant wastewater heat recovery; the 10.6 L/min pathway is additionally conditional on user-behaviour features.

This is strong evidence that the competitive frontier is moving toward coordinated system performance. These UK criteria are not South African law, not a domestic certification route and not an Aqua result.

Official criteria: https://etl.energysecurity.gov.uk/products/showers/showers  
2025/26 update: https://etl.energysecurity.gov.uk/news-events/news/202526-annual-research-programme-updated-criteria-now-live

### Drain-water heat recovery — field-performance warning

A 2025, 14-month in-situ study of a commercial shower drain-water recovery plate heat exchanger reported 66% overall exchanger efficiency after fouling and unsteady-use effects, an 8% reduction in nominal efficiency from fouling, and 52% overall system efficiency after shower-to-drain heat loss was included. These are results from that study and configuration, not universal limits or Aqua results.

The design implication is to compare useful purchased-energy displacement at the bathing-system boundary, not a component brochure's exchanger efficiency.

Bouvenot and Beaudet, *Energy and Buildings* (2025), DOI: https://doi.org/10.1016/j.enbuild.2025.115341

### Repairability and disassembly

ISO 20887:2020 gives design-for-disassembly and adaptability principles, but it does not prescribe universal numeric performance levels. EN 45554:2020 provides generic methods for repair, reuse and upgrade of energy-related products; direct applicability to passive plumbing fixtures must not be assumed. Aqua's product-specific serviceability trials remain a separate method unless the responsible standards authority confirms otherwise.

ISO 20887: https://www.iso.org/standard/69370.html

## 7. Competitive experiment and result presentation

For the first AQS-1 comparison, retain separate arms rather than attributing all improvements to one feature:

1. Exact incumbent shower and washstation installation.
2. AQS-1 passive spray geometry.
3. AQS-1 with independently qualified task-selective boost.
4. AQS-1 with dry-side service spine.
5. AQS-1 with optional indirect DWHR, separately metered.

A feature's contribution is not established by comparing only the full Aqua configuration against the incumbent. Use ablation / incremental arms where practicable: passive geometry alone, boost alone, service spine alone and DWHR alone.

The result is a vector, not one score:

| Dimension | Primary comparison |
|---|---|
| Service quality | Completion, task-specific comfort and independently assessed non-inferiority |
| Spray | Applicable standardized force and coverage method, plus Aqua spatial diagnostics |
| Hydraulics | Measured dynamic pressure/flow envelope, exact datum and uncertainty |
| Thermal | Delivered temperature stability and useful heat at the declared boundary |
| Water | Fresh litres per qualified completed task, including cleaning and failures |
| Energy | Purchased kWh per qualified completed task, including auxiliaries and standby |
| Hygiene / exposure | Independent water, surface/biofilm and aerosol assessments where relevant |
| Accessibility | Task completion and usability for the declared profile |
| Repairability | Qualified return-to-service, destructive access, parts, tools, labour and downtime |
| Lifecycle | Cost and material burden per 1,000 qualified completed services |
| Resilience | Behavior under declared supply, power, sensor, part-availability and recovery faults |

Publish the vector, uncertainty and Pareto frontier. A configuration is Pareto-dominated if another configuration is no worse on every declared metric and strictly better on at least one, under the same functional unit and comparable test conditions. A Pareto frontier does not waive hard acceptance gates.

### Candidate internal targets — still hypotheses

- ≥20% less fresh water per completed task versus the strongest exact comparator, only if all quality and hard gates pass.
- ≥20% lower purchased energy per completed task for the matched power-coupled configuration, if resource closure confirms it.
- 100% of declared routine wear-part jobs accessible without destructive finish removal.
- ≥20% lower whole-service lifecycle cost, without shifting burden to consumables, user cleaning, specialist labour, premature replacement or unqualified maintenance.

These are internal stretch targets, not standards, measured results or achieved performance. The 70% / 80% pressure-retention targets remain separate hypotheses and must be reported only at their defined pressure datums.

## 8. Adversarial / negative-control cases

The software contract should have synthetic coverage for at least:

- missing one required domain;
- duplicate evidence for a domain;
- an assessment outside the selected claim profile;
- wrong evidence authority;
- malformed evidence digest;
- evidence method from another protocol revision;
- evidence conditions from another installation / water / room profile;
- record profile, protocol or conditions differing from the frozen claim;
- failed endpoint despite all other endpoints passing;
- indeterminate endpoint despite all other endpoints passing;
- a failed and indeterminate endpoint in the same record;
- candidate passes while control fails;
- candidate passes while control is indeterminate;
- candidate fails or remains indeterminate;
- same run submitted as control and candidate;
- water-savings claim without water-resource evidence;
- water-and-energy claim with only one resource domain;
- accessible claim without accessibility assessment;
- lifecycle claim without serviceability or cleaning burden;
- missing local safety, water-quality or hygiene/exposure decision.

Synthetic tests validate the evidence gate only. They cannot prove physical performance, microbiological safety, user comfort, accessibility, scald protection or product reliability.

## 9. Integration and release boundaries

- The module consumes decisions and receipts; it does not replace Mycelix, Symtropy, independent laboratories or competent professionals.
- The result is not certification or regulatory approval.
- No remote/cloud process may authorize a valve, heater, disinfectant dose, electrical circuit or other safety-critical actuator.
- A missing or indeterminate required safety / quality assessment means the corresponding claim is not qualified.
- No human-exposure or occupied-installation test is authorized by this protocol.
- Any future physical work requires independently reviewed hazard controls, suitable isolation, calibrated instrumentation, competent installation and applicable local approvals.

## 10. Source-level implementation status

The Rust module defines seven claim profiles, thirteen evidence domains, explicit authority mapping, exact digest binding, three-state decisions and matched control/candidate qualification.

It currently contains **31 authored unit tests**. This is a source inventory, not an execution pass. No cargo test, rustfmt, Clippy or repository CI pass is claimed for this branch until actually observed on its exact head.

## 11. Decision gate

Proceed to an unoccupied AQS-1 bench only after:

1. the claim profile and complete task definition are frozen;
2. the incumbent SKU and all Aqua configurations are exact and traceable;
3. the standard-method and supplementary measurement procedures are reviewed by their competent owners;
4. the resource accounting boundary includes startup, use, cleaning, failed runs and return-to-ready;
5. the independent assessment authorities and acceptance margins are agreed before confirmatory testing;
6. safety, hygiene and measurement-system prerequisites are independently reviewed.

The next major Aqua achievement should be an independently reproducible matched result that qualifies the service first, then demonstrates resource or lifecycle improvement without sacrificing any mandatory domain.
