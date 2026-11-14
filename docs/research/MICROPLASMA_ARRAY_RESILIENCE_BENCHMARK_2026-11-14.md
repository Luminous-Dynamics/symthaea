# MICROPLASMA-ARRAY-RESILIENCE-001 — fault containment and degraded service benchmark

**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12199  
**Parent:** [MICROPLASMA-001 #11780](https://github.com/Luminous-Dynamics/symthaea/issues/11780)  
**Date:** 14 November 2026  
**Status:** S0 synthetic contract and offline audit  
**Claim ceiling:** no physical plasma-array qualification, operating procedure, safety approval, or hardware-actuation authority

## 1. Executive thesis

The next step in numbered-up plasma is not simply to show that many cells can ignite. It is to establish what the array can truthfully claim after partial failure, shared-utility degradation, loss of observability, or coupling between cells.

The engineering object is therefore not just a cell or a nominal array. It is a **qualified service envelope over both nominal and degraded states**.

A system with five active cells is not automatically qualified when one cell fails. A system that reports the aggregate output while hiding an unobserved cell is not qualified. An array that has excellent nominal uniformity but loses it under shared-manifold or thermal interaction is not qualified.

This increment makes the missing failure-state contract executable without claiming to predict discharge physics.

## 2. Research anchors and competitive context

### Independently controlled microplasma cells

Martinet et al. report a five-cell atmospheric RF microplasma array whose cells can be individually ignited and quenched. The paper discusses possible extension to tens or hundreds of cells, but the demonstrated five-cell result is not evidence that a larger array has already been qualified.

- Martinet et al., *Individual ignition of RF microplasma array at atmospheric pressure*, Plasma Processes and Polymers 20, e2200131 (2023). https://doi.org/10.1002/ppap.202200131

### Large-area numbered-up array

Bhattacharya et al. report a 43-element LTCC cold-atmospheric-plasma array with a 10 × 10 cm nominal treatment footprint. Their paper describes 21 kHz operation, 1.55 kV RMS, approximately 130 L/min argon flow, per-element ballast resistors and gas-flow optimization. A specified biofilm-coupon experiment reports a 91% CFU reduction after 150 s at a 1.5 cm gap. These are architecture- and endpoint-specific published results, not universal treatment efficacy, adhesion outcomes, or array fault-tolerance qualification.

- Bhattacharya et al., *Fabrication and Characterization of a 10 × 10 cm Cold Atmospheric Pressure Plasma Array*, IEEE Transactions on Plasma Science 54(2), 496–502 (2026). https://doi.org/10.1109/TPS.2025.3646554

### Industrial process-control baseline

Tantec's PlasmaTEC-X advertises discharge-output monitoring, alarm signals, digital control interfaces, automatic air regulation and control of up to eight generator units from one remote interface. Its published product page lists a 425 W nozzle, 33 L/min air consumption, a 6.1 kg generator and 6–12 mm treatment width. These manufacturer specifications show that monitoring and production integration are incumbent capabilities; they are not a matched demonstration of cell-level degraded-service qualification.

- Manufacturer information: https://tantec.com/atmospheric-plasma-treatment-system-plasmatec-x/

**Competitive conclusion:** Luminous should not claim to invent plasma arrays, individual cell control, or process telemetry. The differentiator to investigate is a vendor-neutral, evidence-qualified degraded-service contract that joins cell identity, fault containment, useful output, product quality and complete measurement boundaries.

## 3. Paradigm shifts

### 3.1 From nominal availability to qualified degraded service

Report at least three separate states:
- nominal service: all declared cells healthy and observed;
- degraded service: a known fault is contained and the application service floor is still met;
- indeterminate service: telemetry, utility state, fault confirmation or neighbour response is missing.

Do not collapse these into one uptime percentage.

### 3.2 From cell redundancy to failure-domain independence

A cell array is not resilient merely because it has many cells. Shared power, gas, cooling, exhaust, control, timing, contamination and thermal structures can create common-cause failures. Each must be represented as a dependency/failure domain before any N-1 claim is accepted.

### 3.3 From aggregate output to cell-reconciled output

The aggregate useful-output observation must reconcile to the per-cell observation set. Unknown or isolated cells cannot silently contribute output. A failed cell must be identified, and the affected region and neighbouring response must be measured at appropriate spatial support.

### 3.4 From passive monitoring to verified containment

The benchmark distinguishes:
- fault detection;
- fault confirmation;
- isolation or failover;
- post-fault neighbour response;
- retained product/service output.

A fault alarm is not proof that the fault was isolated. A software state flag is not physical isolation evidence.

### 3.5 From one fault at a time to common-cause challenge

Future physical campaigns must include shared-utility faults and combinations of degraded conditions. Passing an independent cell fault does not establish survival of a power-supply, gas-manifold or cooling fault.

### 3.6 From uptime to lifecycle service economics

A module that fails less often but requires slow field replacement, extensive calibration, gas flushing or array-wide shutdown may not be better. Report useful accepted output, lost production, false isolation, maintenance time, replacement burden and energy/gas per accepted service.

## 4. S0 executable contract

The machine-readable fixture is:

- `data/research/microplasma_array_fault_fixture_v0_1.json`

The offline auditor is:

- `xtask/src/microplasma_array_audit.rs`
- command: `cargo run -p xtask -- microplasma-array-audit`

The contract binds:
- topology and service-contract identity;
- declared cell count and unique cell IDs;
- each cell's state and normalized useful output;
- diagnostic completeness;
- shared power, gas and cooling status;
- fault class and affected cell identities;
- fault-injection confirmation and containment/failover receipt;
- neighbouring-cell response;
- aggregate output and recomputed cell-output sum;
- a declared minimum useful-output floor;
- the product-quality endpoint;
- whether cell-output uniformity is actually a claimed requirement.

S0 uses normalized synthetic output units. They are intentionally not watts, joules, gas volume, treated area, bond strength, reactive-species dose or a physical plasma measurement.

### Dispositions

- **PASS:** internally consistent synthetic scenario meets its declared service contract.
- **BLOCKED:** required diagnostics, fault confirmation, containment/failover, utility state or neighbour observation is missing or unknown.
- **REJECTED:** known identity, isolation, service-floor, product-quality, shared-utility or neighbour-envelope requirements are violated.

PASS only means the synthetic contract evaluator behaved as designed. It does not qualify a physical device.

## 5. Frozen benchmark and negative controls

The fixture contains 28 deterministic scenarios: nominal service; N-1 and N-2 isolated-cell service; shared-power and shared-gas failover; an explicitly non-uniformity-claim case; below-floor degradation; unisolated failed cells; shared-utility excursions; thermal cross-talk; diagnostic/sensor loss; unknown cell/utility/neighbour state; missing fault confirmation; missing containment; aggregate-output mismatch; excessive cell-output CV; duplicate/foreign identities; incomplete manifest; isolated-cell nonzero output; claim/state mismatch; and failed product-quality endpoint.

The audit computes cell-output coefficient of variation from active cell observations. It never trusts a separately supplied summary statistic. The 5% ceiling is a synthetic benchmark target inherited from the parent MICROPLASMA-001 proposal; it is not a universal plasma standard. Where uniformity is not part of the declared service claim, CV is reported conceptually but is not silently promoted into a universal rejection criterion.

The negative controls explicitly test that:
1. unknown cells cannot be treated as healthy;
2. isolated cells contribute zero output;
3. duplicate or foreign cell IDs reject;
4. the aggregate sum reconciles to cell-level observations;
5. a faulted active cell cannot be counted as healthy;
6. missing diagnostics and missing neighbour observations block;
7. shared utility excursions reject;
8. a failed product-quality endpoint cannot be rescued by throughput;
9. a nominal-service claim cannot hide a fault;
10. a degraded-service claim cannot omit its fault class.

## 6. Prospective physical benchmark

These are future measurement goals, not demonstrated Luminous performance.

| Dimension | Required measurement |
|---|---|
| Nominal uniformity | Per-cell output distribution and spatial support |
| N-1 / N-k retention | Accepted useful output after a predeclared number/fraction of cells is disabled |
| Fault detection | Detection latency and missed-fault rate |
| Isolation | Containment latency, isolation success and false-isolation rate |
| Common-cause tolerance | Power, gas, cooling, exhaust and control failure-domain challenges |
| Cross-talk | Neighbour-cell electrical, thermal, fluid and process-output response |
| Product outcome | Actual selected application endpoint, not plume appearance or nominal cell power |
| Reliability | Cell survival, drift, starts/stops, service intervals and module replacement time |
| Economics | Total-system energy, gas, cooling, extraction, downtime and accepted output |

For a specific application and array topology, predeclare:
- the number and fraction of failed cells;
- which failures are independent versus shared;
- the minimum acceptable service output;
- the product-quality threshold;
- permitted neighbour response;
- measurement uncertainty and calibration;
- test duration, start/stop cycles and failure accounting.

Do not assert a universal N-1 service-retention percentage. A five-cell array, a 43-element array, and a 100-cell array have different topology, common-cause and output constraints. A physical scale-out claim requires a new array assembly and a held-out fault challenge, not interpolation from cell count.

## 7. Comparison protocol against SOTA and industry

The fair comparison is not “our microplasma has more cells.” Compare candidates under the same service contract and report:

1. nominal accepted output and quality;
2. output distribution across cells and treatment footprint;
3. output under the same N-1 / N-k challenge;
4. detection, isolation and false-isolation behavior;
5. common-cause and cross-talk response;
6. total-system energy and consumables at the same service output;
7. downtime, replacement and lifecycle cost;
8. uncertainty, failed trials and excluded specimens.

Use a conventional single-source industrial reference, an individually controlled microcell array, a large-area numbered-up array and the candidate modular architecture where the same endpoint can be tested. Do not compare source-reported antimicrobial results against an adhesive-bond endpoint.

Incumbent vendors already offer output alarms, digital interfaces, remote generator control and automatic air regulation. Luminous must add verifiable evidence composition and honest degraded-state claims, not merely reimplement their HMI.

## 8. Integration boundaries

- #11780 remains the parent taxonomy and miniature/modular source benchmark.
- #11718 and #11758 own energy and metrology boundaries; this audit does not add a second ledger or calibration registry.
- #5669 and #7588 own plasma regime/model applicability.
- #4866 and #4800 own physical observation and experiment provenance.
- Symtropy #1685 and #1467 remain the owners for a future deterministic array digital-twin/fault-injection scenario engine.
- The selected application benchmark owns the actual accepted product endpoint.

No new solver, plasma chemistry model, optimizer, hardware controller, HAL command path, sensor authority or physical operating procedure is introduced.

## 9. Validation record and limitations

Validation planned/performed for this increment:
- strict JSON parse and schema enforcement;
- independent JavaScript known-answer replay of the frozen fixture;
- Rust tests for fixture replay, missing diagnostics, isolated-cell output, faulted-active-cell rejection, aggregate reconciliation and product-quality rejection.

The current authoring environment has no Cargo, rustc, rustfmt or Nix. Rust compilation, tests, formatting and Clippy are therefore **not executed**. Hosted CI status is not assumed. No physical device, solver, HV/RF source, gas system, plasma cell or test coupon was energized or operated.

The evaluator is a structural contract audit. It does not authenticate raw evidence, establish sensor calibration, predict plasma physics, qualify safety, prove failure probability, or establish industrial superiority.

## 10. Research references

- Martinet et al. (2023), https://doi.org/10.1002/ppap.202200131
- Bhattacharya et al. (2026), https://doi.org/10.1109/TPS.2025.3646554
- Tantec PlasmaTEC-X manufacturer information, https://tantec.com/atmospheric-plasma-treatment-system-plasmatec-x/
- Parent microplasma benchmark and additional source register: https://github.com/Luminous-Dynamics/symthaea/issues/11780
