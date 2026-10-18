# MINE-COMMS-001 — Resilient Underground Communications Benchmark V0.1

**Status:** research contract / design draft. No physical-layer implementation or field qualification is claimed.  
**Tracking:** [Symthaea issue #8010](https://github.com/Luminous-Dynamics/symthaea/issues/8010).  
**Composition:** [Symtropy MINE-SYSTEM-001 #1933](https://github.com/Luminous-Dynamics/symtropy/issues/1933), [FLOOD-EGRESS-001 #1963](https://github.com/Luminous-Dynamics/symtropy/issues/1963).  
**Claim ceiling:** passing synthetic tests means only that this prototype meets its defined software contracts. It does not certify mine radio coverage, permissible equipment, personnel tracking, evacuation, or rescue.

## 1. Research question and paradigm shift

Can a heterogeneous underground communication fabric preserve a minimal lifesaving message service after ordinary network paths are lost, while adapting routing and message handling to measured channel conditions without suppressing alarms or overstating delivery?

The paradigm shift is **not one universal radio**. It is a multi-layer system that degrades predictably:

1. **Primary network:** existing wired/fibre infrastructure, mine telephone systems, leaky-feeder or distributed radio where engineered and appropriate.
2. **Local access:** approved fixed, portable or mobile radio relays; explicit node power, coverage, topology and interference bounds.
3. **Emergency through-the-earth (TTE) overlay:** independently powered magnetic-induction or electrode-based distress/text link where empirical link measurements justify it. Treat voice or high-rate data as unproven until measured for the specific geology and geometry.
4. **Optional localization overlay:** treated as a separate estimator with its own error distribution, unavailable state and confidence. A stale location must never masquerade as current.
5. **Delay-tolerant store-carry-forward:** nodes can carry an authenticated distress message until contact returns. Message origin, creation time, expiry, duplicate identity and relay history stay intact.

Each additional layer must add a distinct failure mode and a measurable service benefit. Two radios that share the same cable, switch, shaft, power feed, route or flood exposure are not automatically independent redundancy.

## 2. What Symthaea should—and should not—do

Symthaea should initially be a bounded network-state estimator and decision-support layer, not an invented RF/MI physics engine or a sole safety controller.

Candidate responsibilities:
- Track measured per-link quality, noise/interference, packet loss, latency, retry cost, node battery, topology confidence and time since last successful acknowledgement.
- Predict degradation only using features actually available at decision time; compare against fixed routing and a simple statistical baseline.
- Select among already-qualified paths, retry budgets and store-carry-forward actions under explicit energy and expiry limits.
- Apply declared message priorities: distress / evacuation / emergency alert, personnel status, voice, telemetry, then bulk data. Lower-priority traffic must not starve life-safety traffic.
- Emit audit receipts that distinguish **created → queued → transmitted → relayed → received → authenticated → acknowledged**. Never treat a successful local send call as proof of remote delivery.
- Continue with a small deterministic fallback when Symthaea, its learned estimator, shared memory or upstream services fail.
- Make HDC/CfC an ablation, not a premise: compare it to static policies and simpler statistics on the same frozen traces. If it does not improve a decision-relevant metric without safety regressions, omit it.

### 2.1 Reuse established delay-tolerant messaging protocols

Do not invent a proprietary store-carry-forward protocol until standards fit has been tested. Evaluate **Bundle Protocol Version 7 (BPv7, RFC 9171)** for bundle identity, lifetime/age, hop-count protection and optional reception/forwarding/delivery/deletion status reports: https://www.rfc-editor.org/rfc/rfc9171.html. Evaluate **Bundle Protocol Security (BPSec, RFC 9172)** for end-to-end integrity and confidentiality where the deployment's trust and key-management model supports it: https://www.rfc-editor.org/rfc/rfc9172.html.

Important constraints from the standards:
- Bundle lifetime/age and hop-count semantics help make delayed messages and routing loops explicit, but do not turn an old personnel position into current truth.
- BPv7 status reports can multiply traffic along a path. RFC 9171 says status-report generation is disabled by default because excessive reports can degrade operation. Enable reports selectively for critical service classes with bounded rate, queue and expiry rules; do not create an alarm storm by acknowledging every telemetry bundle.
- BPSec protects bundle data integrity/confidentiality; it does not secure a compromised endpoint or make local keys safe automatically. Do not silently drop a distress message solely because an optional identity backend is offline; carry the authentication state as explicit evidence.
- Keep the Rust harness transport-agnostic. Before adopting BPv7/BPSec, compare an interoperable implementation with the typed lifecycle contract and define how local receiver acknowledgement differs from bundle reception, forwarding and application-level acknowledgement.

Physical layer design, antenna construction, electrical permissibility, installed mine maps, radio licensing and mine emergency procedures stay with the appropriate hardware, mine and regulatory authorities. No software-only benchmark can approve a real installation.

## 3. Research evidence and limitations

- **Ngwenyama & Webber-Youngman (2024), “A Review of the Applications of Through-the-Earth (TTE) Communication Systems for Underground Mines.”** Reviews electromagnetic magnetic-induction/electrode and seismic methods for post-accident communications. Range/data rate depend on antenna geometry, frequency, coupling and rock properties, including conductivity, moisture and mineralogy. The paper reports that TTE may sustain communications without depending on the usual shaft-to-workplace backbone, but also emphasizes attenuation and range/rate trade-offs. DOI: [10.1007/s42461-024-01056-5](https://doi.org/10.1007/s42461-024-01056-5).
- **Austin (2024), “Radio communications through rock strata — South African mining experience over 50 years.”** Documents South African work on propagation through rock and incidental coupling to conductors, illustrating that local experience and antenna constraints matter. DOI: [10.17159/sajs.2024/17301](https://doi.org/10.17159/sajs.2024/17301).
- **Ma et al. (2025), “Through-the-Earth Magnetic Induction Communication and Networking: A Comprehensive Survey.”** Reviews magnetic-induction channel modelling, relays and network architecture; identifies issues including fast fading and crosstalk as network density and topology grow. DOI: [10.1109/COMST.2025.3623258](https://doi.org/10.1109/COMST.2025.3623258); [arXiv record](https://arxiv.org/abs/2510.14854).
- **NIOSH, Medium Frequency Mine Emergency Communications.** An infrastructure-light emergency communications approach may be more survivable when a mine disaster damages the ordinary underground infrastructure. This is a design motivation, not a universal product-performance guarantee: [Technology News 546](https://www.cdc.gov/niosh/docs/mining/works/coversheet906.html).

**Interpretation:** TTE is a credible emergency-overlay research direction, not magic broadband through arbitrary rock. Use measured test corridors to learn coverage and rates. Do not transfer performance numbers across rock, depth, antenna, moisture, geometry or power assumptions.

### Literature-inspired workload profiles (not product guarantees)

The 2024 TTE review describes a set of previously reported magnetic-induction configurations with approximate service modes of 10 bit/s text-only, 100 bit/s text/data, 500 bit/s voice messaging and 2,500 bit/s real-time voice. It also summarizes reported field tests reaching depths up to 600 m and horizontal distances up to 1,500 m under particular antenna, site and frequency-band conditions (reported as 3–8 kHz). These figures are **literature-specific examples**, not a promise for a new design, an assurance of simultaneous range and rate, or a universal mine operating envelope. Source: [Ngwenyama & Webber-Youngman (2024), section 3](https://link.springer.com/article/10.1007/s42461-024-01056-5).

Use them only to define test traffic profiles and questions:
- At 10 bit/s, can a short SOS packet plus framing/authentication overhead fit within the declared time and energy budget?
- At 100 bit/s, can compact status/location uncertainty records be sent without overflowing the queue?
- At 500 bit/s, what latency and duty cycle apply to a voice-message service?
- At 2,500 bit/s, what additional link, antenna, power and safety assumptions are required, and are those assumptions actually supported by a measured installation?

Before comparing profiles, define whether rates are application payload or raw physical bitrate, include framing/retries/duplex overhead, and report successful payload goodput separately. The current Rust kernel does **not** simulate bitrates or airtime yet; these are inputs for the next harness increment, not results of the current CLI.

## 4. Frozen comparators

| ID | Comparator | Purpose |
|---|---|---|
| C0 | Fixed wired/fibre/leaky-feeder path where available | Operational baseline |
| C1 | Conventional radio or mesh with fixed routing | Wireless baseline |
| C2 | Heterogeneous network with deterministic static failover | Primary safety baseline for adaptive proposals |
| C3 | Bounded route/retry/replication/store-carry-forward policy | Adaptive systems hypothesis |
| C4 | HDC/CfC-assisted estimator, if implemented | Ablation; must beat C2/C3 on held-out traces |
| C5 | Exact oracle on small synthetic graphs only | Diagnostic upper bound, never a deployable baseline |

All candidates use the same topology, traffic, power budgets, initial link states, channel traces, damage/fault schedules and ground-truth message outcomes. Do not compare on disjoint random seeds. Split by entire topology/geology/fault families, not randomly selected packets, to avoid leakage.

## 5. Scenario manifest

Each case includes a versioned topology, units/coordinate frame, link/channel evidence class, node and energy constraints, message trace, message priorities/deadlines, failure schedule, location ground truth where applicable, simulator/model/verifier digests, expected disposition and claim ceiling.

| Case | Fault or question | Required result |
|---|---|---|
| MC-001 Baseline | Nominal multi-level drift/crosscut topology | Reproducible per-class metrics and acknowledgements |
| MC-002 Backbone severance | Fibre/leaky-feeder segment lost | Measure time to first emergency message via remaining path |
| MC-003 Power-domain loss | A node/switch/power feeder dies | No false “healthy”; route around only where a validated path remains |
| MC-004 TTE envelope | Sweep declared depth, conductivity, moisture and coil alignment | Emit measured-model operating envelope and out-of-domain status |
| MC-005 Interference / machinery | Noise and interference bursts | Report retries, delay and message class impact |
| MC-006 MI topology | Relay placement, crosstalk and density variation | Measure gains and degradations, including worst cases |
| MC-007 Flood / inrush | Local path and some fixed nodes fail | Critical alarm, power-state and alternate-route receipts remain truthful |
| MC-008 Store-carry-forward | No live path for a bounded interval | Deliver only unexpired messages with origin, age, replay/duplicate handling |
| MC-009 Stale location | Tracking path fails or miner moves after last fix | Expose uncertainty and age; do not publish as current position |
| MC-010 Compound outage | Backbone + power + one relay loss | Fail visibly; preserve only the services shown to remain physically reachable |
| MC-011 Alarm storm | Many simultaneous alerts plus bulk telemetry | Priority and bounded queue policy protect distress traffic |
| MC-012 Adversarial data | Replay, malformed, delayed, unauthenticated or conflicting packets | Explicit authenticity/availability status; no silent escalation or suppression |
| MC-013 Recovery | Infrastructure returns with inconsistent queues | No stale state overwrites new truth; reconcile duplicates and route generations |

Synthetic channel models begin at evidence class E0. A model matched to measured link traces may become calibrated-model evidence, but a held-out physical test is required before claiming measured out-of-sample behavior. Field qualification requires additional independently governed tests and mine approval.

## 6. Required metrics

Every metric declares unit, numerator/denominator, measurement window, topology/fault stratum, source, uncertainty and missing-data rules.

- **Delivery:** PDR by message class; successful authenticated acknowledgements; undelivered/expired messages; duplicate/replay acceptance; false healthy-link reports.
- **Time:** time-to-detect link loss; time from backbone failure to first SOS sent, received and acknowledged; p50/p95/p99 latency and outage duration.
- **Capacity:** goodput, queue age, retries, contention, crosstalk and starvation by class; keep byte/packet rates distinct.
- **Resilience:** connected components / path diversity, distinct failure domains, reachable miners/locations, restoration time and service remaining after each fault.
- **Energy:** joules per delivered critical message, idle/standby load, battery autonomy under declared send schedule, power-state uncertainty and expiry safety margin.
- **Localization (optional):** error distribution, confidence calibration, false-fix rate, availability and age of last valid fix.
- **Operations:** setup time, node mass/size, installation effort, inspection interval, replaceable parts and end-of-life behavior.
- **Robustness:** outcomes across topology, depth, material electrical properties, moisture, antenna coupling/orientation, machinery noise, changing geometry and traffic load.

Do not collapse these into one score. A throughput gain cannot compensate for reduced distress delivery, stale locations, hidden outage states or an unsafe power envelope.

## 7. Initial acceptance targets (laboratory hypotheses)

These are targets for test design; none are yet measured:
1. **Zero false healthy states** for failed, disconnected, stale or out-of-envelope critical links in the frozen negative-control corpus.
2. **Typed delivery semantics:** no successful end-to-end status without receiver-side evidence; no response becomes “delivered” solely because the local transmitter accepted a packet.
3. **Single-fault alert delivery:** aim for at least 99.9% delivery over the explicitly defined test set; publish numerator, denominator, confidence interval, conditions and every miss. This is not a guarantee of real-mine reliability.
4. **Connected local path:** initial lab target of p95 time-to-first-SOS ≤ 5 s, measured from event creation to receiver acknowledgement. Test TTE cold-start and long-range cases separately rather than assigning them an unrealistic shared deadline.
5. **Redundancy:** demonstrate critical messaging across at least two independent failure domains when feasible; run common-cause faults so a claimed redundant pair is not simply two nodes sharing a single vulnerable feed/route.
6. **No learned-policy regression:** C3/C4 may not reduce delivery or violate resource/fallback constraints compared with C2 on the frozen test set; improvement must hold on held-out topology/fault families.
7. **Energy/autonomy:** report a complete declared traffic schedule and measured energy; set a product-specific autonomy target only after hardware and rescue requirements are known.
8. **Local fallback:** if Symthaea or its prediction is absent/invalid, the baseline alerting and routing policy still operates within its separately measured envelope.
9. **Provenance:** every run is reproducible from frozen inputs, source digest, parameters, simulator revision and verifier identity.
10. **Evidence honesty:** zero real-mine coverage, survival or regulatory claims from synthetic-only evidence.

## 8. Security and human factors

Critical traffic needs origin/authenticity checks, replay protection, expiry and duplicate semantics. At the same time, the distress path should not silently discard an emergency because an optional identity service or cloud backend is unreachable. Separate message transport, origin authentication and claimed location so responders can see what is known and unknown. Define a safe degraded mode with mine safety personnel; do not invent authorization policy in this benchmark.

Field usability tests should cover gloves, hearing protection, noisy machinery, low light, low literacy/language assumptions, degraded batteries and accessible alarms. The design must be usable without a cloud account or a smartphone in the critical path.

## 9. Integration and sequence

- Link to Symtropy [MINE-SYSTEM-001 #1933](https://github.com/Luminous-Dynamics/symtropy/issues/1933) and [FLOOD-EGRESS-001 #1963](https://github.com/Luminous-Dynamics/symtropy/issues/1963) for shared underground fault scenarios, flood/inrush alarms and safe-egress messaging.
- Use Sol Atlas only as the existing spatial/topology evidence authority; do not create a duplicate authoritative mine layout or escape map.
- Use Mycelix for provenance and receipt distribution where interfaces are available, while preserving an offline local safety path.
- First freeze synthetic traces and implement a deterministic software harness. Then compare baselines and measure model error. Only after that should hardware-in-the-loop and supervised field testing be proposed with instrumented fixtures, electrical/RF review and independently approved plans.
- The first milestone is a reproducible answer to **which message services survive which failures, for how long, at what energy cost, and with what uncertainty**—not a claim that Symthaea has already invented a radio that penetrates every mine.

## 10. Non-goals

No new physical RF/MI propagation authority; no unsupported antenna or power claims; no real equipment actuation; no unsupervised evacuation decisions; no people-tracking certainty from stale sensors; no compliance certification; no production AI dependency for emergency delivery; no claim that HDC/CfC is better before ablation.
