# Marine multi-domain test campaign and measurement integrity — V0.1

**Status:** marine campaign-binding adapter and synthetic fixtures; no instrument procurement, physical acquisition or qualification is claimed.  
**Parent:** [MARINE-TEST-EVIDENCE-001 #11064](https://github.com/Luminous-Dynamics/symthaea/issues/11064)

## 1. Objective

Bind the proposed LOS subsystem demonstrator's physical article, as-built revision, instruments, calibration state, clocks, raw observations, protocol deviations and analysis profile into one auditable campaign.

This is not another telemetry stack. It is a marine profile over existing canonical owners:

| Concern | Canonical owner |
|---|---|
| Instrument physical identity, decoder and raw-to-measurement boundary | ROB-INSTR #4866 |
| Prospective protocol and experiment run | ROB-EXP #4800 |
| Physical observation / raw observation identity | SE-OBS #3695 |
| Calibration identity, epoch and currentness | EXEC-ID #4620 |
| Quantity kind, unit, frame and conversion | SE-SEM #4861 |
| Physical article and as-built configuration | ROB-REALIZE #4859 |
| Mission and vessel design subject | MARINE-LIVEABOARD #10926 |
| RO water process | MARINE-WATERMAKER-BENCH #10994 |
| Electrical transients | MARINE-POWER-TRANSIENT #11028 |
| Multi-carrier service scheduling | MARINE-RESOURCE-OPS #11008 |
| Serviceability mock-up | MARINE-SERVICEABILITY #11049 |

The marine adapter holds references and marine-specific channel profiles. It must not create a second observation, calibration, quantity/unit, clock or raw-data ontology.

## 2. Paradigm shift: measurement-native engineering

Instrumentability must be a design constraint from the start:
- reserve sensor mounts, cable routes, probe access, reference datums and calibration access;
- define system boundaries and metering points before selecting an efficiency target;
- choose instrument bandwidth and timing from the physical claim, not convenience;
- preserve raw outputs, calibration, parser identity and derived-data lineage;
- make calibration, clock quality, sample integrity and uncertainty visible at decision time;
- retain failed, aborted and out-of-envelope runs, not just the clean traces.

A beautiful digital twin cannot compensate for a poorly instrumented physical article. A high-rate sensor cannot compensate for an unknown calibration. A calibrated channel cannot repair a mismatched as-built configuration.

## 3. Evidence theorem

Keep these distinctions explicit:

```
HAL/runtime telemetry != calibrated observation
calibration identity != calibration correctness/currentness
sample index equality != temporal simultaneity
raw sample != decoded quantity != derived metric
simulated/replayed trace != new physical acquisition
protocol planned != protocol executed
run completed != analysis eligible
analysis eligible != system qualified
```

The campaign verifier checks identity and evidence prerequisites. It does not certify an instrument, approve water, qualify a power system or authorize equipment operation.

## 4. Campaign subject and identity

Each campaign binds:
- exact prospective ROB-EXP protocol ref and digest;
- exact physical article identity at the strongest honestly supported level;
- exact as-built configuration and installation generation;
- exact mission/profile/conditions where the test claim is mission-specific;
- exact subsystem/system boundary;
- exact instrumentation endpoints, mounts, quantity/unit/frame refs and calibration refs;
- exact acquisition and parser/normalizer profiles;
- raw observation bundle and raw artifact digests;
- exact analysis profile and acceptance-criteria owner;
- protocol deviations, aborts, missing samples and data-quality status.

A replaced instrument, moved sensor, modified pipe/wiring, changed module revision or changed calibration epoch creates a new measurement context. Historical raw observations remain historical; do not rewrite them to match the new configuration. Each measurement channel binds its own as-built configuration digest and each campaign declares a maximum synchronization-uncertainty budget for the claimed analysis. The verifier compares these values; it does not assume a channel inherits the article's identity or that clocks are aligned because samples share an index.

## 5. Channel profile and sampling adequacy

Every channel carries:
- physical endpoint identity strength and instrument revision;
- measured quantity kind, unit and frame/location ref;
- mounting/transform identity;
- range, resolution and saturation behavior;
- actual sample rate, method-declared minimum sample rate, analogue bandwidth and claim bandwidth;
- sensor response/lag where material;
- anti-alias filter identity and acquisition chain;
- calibration identity/epoch/currentness and uncertainty profile;
- source/DAQ/receive clock identities and synchronization uncertainty;
- raw observation ref/digest and decoder/parser identity.

**No universal sample-rate constant is prescribed.** The required bandwidth and sampling plan depend on the exact observable, event dynamics, sensor/probe response, DAQ, anti-aliasing and analysis. The responsible test engineer must issue a channel-specific sampling-adequacy review and declare the method-specific minimum sample rate. The semantic verifier rejects a sample rate below that minimum; it also rejects analogue bandwidth below the claimed observable bandwidth. A slow aggregate cannot support a fast-transient absence claim.

Preserve distinct timestamps:
- device/source timestamp;
- DAQ acquisition timestamp;
- host receive timestamp;
- command or event timestamp where applicable;
- monotonic campaign clock;
- synchronization offset/drift/jitter and uncertainty.

Do not interpolate away missing data for a hard acceptance claim. Retain gaps, clipping, duplicate/reordered samples and clock discontinuities.

## 6. Marine channel families

### RO / watermaker
Define the whole-system electrical boundary and separately observe seawater feed, permeate, concentrate/brine, pressures, temperatures, conductivity/salinity, tank level, pump/valve state and process phase. ISO 25175:2026 may be used for applicable RO/NF membrane-element test methods; it is not a complete installed watermaker or potable-water test. Use a separately approved water-quality method. Reconcile feed/permeate/concentrate using synchronized flow, salinity/density and uncertainty; naive volume closure is not enough.

### Electrical / propulsion / transient
Define each bus and load boundary. Capture voltage/current waveforms, source state, protective event markers and applicable AC or DC observables. The measurement bandwidth must support the exact transient claim. The electrical architect/test engineer sets safe limits, isolation and measurement topology. Do not infer fault-current, protection, EMC or transient compliance from a battery-monitor log.

### Thermal / habitability
Bind each temperature/humidity/motion sensor to its physical location, frame, mounting, calibration and response. Distinguish air, surface, coolant, battery case and inferred internal temperatures. A case temperature is not winding/core temperature unless a validated model establishes that inference.

### Mechanical serviceability
Use event-based phase markers and calibrated dimensions/mass where relevant. Preserve mock-up revision, tools, module ID, access configuration, trial participant/crew count and completion events. Video is optional, consent-governed and not required for time/clearance evidence.

## 7. Current standards and method boundaries

- **ISO 16315:2026 Edition 2** was published 6 February 2026. Its scope covers design/installation of AC/DC electric propulsion and the electrical part of hybrid propulsion on small craft up to 24 m LH, with stated voltage scope below 1,500 V DC and up to 1,000 V single-/three-phase AC. The 2026 edition clarifies overcurrent protection and revises fault monitoring for isolated DC systems. Confirm exact applicability and review the licensed standard with a qualified marine electrical engineer; record applicable clause refs without reproducing copyrighted text. Official record: https://www.iso.org/standard/84203.html
- **ISO 16315:2026 Edition 2**: ISO's public metadata notes that the 2026 edition clarifies overcurrent protection and revises fault monitoring for isolated DC systems. For LOS electric/hybrid propulsion, record the exact applicable edition and reviewer-approved requirements; capture relevant pre-commissioning test evidence and protection/fault-monitoring observations in the qualified electrical test plan. Do not infer compliance from a sensor log or reproduce licensed standard text. https://www.iso.org/standard/84203.html
- **ITTC Recommended Procedures register, Revision 10** includes resistance-test and uncertainty-analysis procedure families. Use current applicable ITTC procedures for towing-tank/model-test campaigns; do not improvise instrumentation or uncertainty methodology. https://ittc.info/downloads/quality-systems-manual/recommended-procedures-and-guidelines/
- **ISO 25175:2026** covers specified RO/NF membrane-element tests, not installed watermaker service or potable-water acceptance. https://www.iso.org/standard/89252.html
- **Bordbar et al. 2026 shipyard production planning review** finds hybrid DES–optimisation more operationally credible than many architecture-rich but longitudinally unvalidated DT/AI systems; it reinforces the need for calibrated data, execution-layer connections and long-run KPIs. https://doi.org/10.3390/jmse14040396

## 8. Frozen synthetic corpus and adversarial controls

The companion JSON contains five synthetic campaign fixtures:
1. RO multi-stream campaign: well-formed reference lineage, but synthetic only.
2. Electrical transient: required sampling-adequacy review missing; no transient admission.
3. Thermal/habitat: calibration stale; raw history remains but calibrated analysis is ineligible.
4. Serviceability timing: clock disagreement exceeds the declared synthetic timing budget; temporal ordering is indeterminate.
5. Mixed subsystem: one channel binds a different as-built configuration; do not merge into one article/campaign claim.

The values are synthetic and deliberately not vessel design limits or sensor specifications.

## 9. Executable semantic verifier

The branch now includes the standalone tool workspace `tools/marine-campaign-evidence`, a small Rust verifier over `serde_json::Value`. It consumes the reference-only campaign record and checks cross-record semantics that JSON Schema alone cannot establish:
- exact article ↔ mission ↔ per-channel as-built configuration binding;
- prospective protocol frozen before the run;
- run start/end ordering and run disposition / abort consistency;
- instrument identity strength and monotonic measurement range;
- measured analogue bandwidth versus claimed observable bandwidth;
- channel-specific sampling-adequacy review presence;
- calibration currentness and required canonical calibration/epoch/qualification/uncertainty references;
- sample integrity counters and raw-observation digest presence;
- source/DAQ/receive synchronization uncertainty against the declared analysis budget;
- acquisition/evidence-class boundaries for synthetic, measured, replay, HIL and PHIL inputs.

Disposition is fail-closed: `Invalid`, `NotAdmissible`, `Indeterminate`, `SyntheticOnly` or `EligibleForIndependentReview`. The last state is not an engineering or regulatory pass.

Ten Rust unit tests are authored, including the five frozen campaign fixtures plus negative mutations for article/channel mismatch, stale calibration, missing sampling review, clock-budget breach, missing samples, unfrozen protocol, weak instrument identity, safety-aborted run and reversed measurement range. The tool is intentionally isolated from the root Symthaea workspace and its Cargo lock, avoiding unrelated root-workspace checks; it pins `serde_json` and generates its own tool-local lock on first invocation. That lock must be committed after a real Rust run before reproducibility is claimed. **These tests have not been executed** because Cargo/Rust is unavailable in the current execution environment; exact-head CI remains the required compilation/test evidence.

## 10. Benchmarks — targets, not results

- 100% admitted channels bind exact instrument endpoint, quantity, unit, location/frame, calibration status and timebase.
- 100% runs bind exact protocol, physical article/as-built revision, mission/condition where applicable, raw bundle and analysis profile.
- zero stale/unknown calibration admitted where the analysis requires qualified calibration.
- zero fast-transient absence claims from a slow or bandwidth-inadequate record.
- zero missing/duplicate/reordered/clipped samples silently erased.
- 100% derived outputs link to exact raw parent and parser/filter/conversion identity.
- 100% deviations, aborts, saturation, stale states and failed runs are retained.
- deterministic fixture generation/canonicalization under pinned profile/toolchain.
- independent verifier re-derives campaign disposition; no self-asserted `valid=true`.
- uncertainty and timing budgets are claim-specific; no universal percent target before a calibrated measurement chain and observable are selected.

## 11. First physical test sequence

1. Freeze the demonstrator's exact article and measurement boundary.
2. Select one low-risk watermaker or inert mechanical test case first; do not combine unrelated tests into one physical claim.
3. Choose instruments with suitable range, bandwidth, response, environmental rating and calibration availability.
4. Define channel-by-channel uncertainty and time-synchronization budgets.
5. Complete electrical/water/process risk review and approved test procedure.
6. Capture a dry run and verify raw/decoded data lineage, clock health, saturation and abort path.
7. Execute the supervised test, preserving all runs and deviations.
8. Have a reviewer independent of the data-acquisition implementation assess the result.
9. Only then bind measurements into the watermaker, transient, serviceability or mission evidence owners.

## 12. Claim ceiling

A valid manifest or synthetic corpus PASS does not establish instrument accuracy, metrological traceability, water potability, electrical safety, power quality, transient stability, habitability, hull performance, seaworthiness, class/flag compliance or authority to operate. Each physical claim is limited to its exact article, configuration, instrument, calibration, method, uncertainty, conditions and competent review authority.
