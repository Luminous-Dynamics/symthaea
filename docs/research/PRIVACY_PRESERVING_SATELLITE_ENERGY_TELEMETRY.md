# Privacy-Preserving Electrical Usage Telemetry over Satellite

**Status:** Research design plus offline contract, authenticated-envelope, and SQLite delivery prototypes. No meter has been connected, no satellite transmission or field trial is claimed, and no production-ready cryptographic deployment is asserted.  
**Scope:** Aggregated electricity-use telemetry for remote feeders, rural sites, microgrids and resilience monitoring, with satellite as a transport option—not a privacy control by itself.  
**Threat model reviewed:** 2026-10-12.

## Executive decision

**Yes, electrical usage can be monitored over satellite while materially reducing household-data exposure.** The key design choice is to make raw readings local and make the satellite uplink an aggregate-only boundary. Encryption is necessary but does not, by itself, protect against an authorized recipient of decryptable per-meter records or all traffic-analysis risks.

Use the satellite link where coverage, resilience or site isolation makes it worthwhile. For dense urban locations, compare terrestrial cellular, fibre, RF mesh and power-line communications first; satellite airtime, antenna visibility, power, latency and terminal costs may make satellite a poor default. Satellite is the backhaul option, not the sensor and not the privacy scheme.

## Goals and non-goals

Goals:
- monitor aggregated Wh/kWh and broad demand trends for grid operations, microgrids, energy planning and outage awareness;
- allow household-scale readings to stay on-device or within the authorized local meter/billing boundary;
- expose only a coarse, sufficiently large cohort aggregate over satellite;
- support secure aggregation, differential privacy for research/public release, replay-resistant batches and auditable purpose/consent policies;
- work with intermittent low-bandwidth links by batching and store-and-forward.

Non-goals for v1:
- replacing revenue-grade meter certification, billing accuracy, utility safety systems or grid protection;
- uploading household-by-household interval curves for analytics;
- inferring individual appliance use, occupancy, employment schedules or household behaviour;
- remotely disconnecting meters, changing tariffs, dispatching DERs or controlling equipment;
- claiming legal compliance merely because an envelope passes the validator.

## Architecture

```text
Revenue meter / local energy sensor
    |  raw readings remain inside the authorized local boundary
    v
Trusted site gateway OR cryptographic secure-aggregation protocol
    |  validate meter freshness/calibration; aggregate eligible participants
    |  suppress undersized cohorts; apply DP for research/public release
    v
Aggregate-only envelope + detached signature / attestation
    |  end-to-end authenticated encryption; rotating radio/device identity where supported
    v
Satellite IoT / NTN uplink (batch + store-and-forward)
    |  receiver verifies signature, sequence/replay state and privacy policy
    v
Grid / microgrid analytics: aggregate demand, trend, resilience and outage summaries

Separate path: customer-authorized individual billing data -> authorized billing recipient.
No raw readings, customer identifiers, household location, or per-meter telemetry in this aggregate path.
```

A **trusted gateway** is a clear and comparatively simple first prototype where all participants agree to trust the gateway operator. If the gateway operator itself must not learn individual readings, use a reviewed secure-aggregation protocol so the gateway obtains only the sum (with dropout handling and cohort policy). Do not invent new cryptography. Secure aggregation is not identical to differential privacy: it hides individual inputs from the aggregator under a stated trust model, while differential privacy limits what a released statistic reveals about an individual's participation/value.

### What travels over the satellite link

The v1 contract is a single aggregate for a bounded time window and coarse distribution area/microgrid:
- random, batch-scoped 128-bit identifier for idempotency; never a meter or household identifier;
- UTC aggregation window and duration;
- opaque region token for a **group**, not a customer or individual meter; rotate/retire it under an explicit schedule to reduce long-term linkability;
- cohort size and a policy-defined minimum cohort threshold;
- sum of energy in integer watt-hours, coverage fraction, and a coarse measurement-quality state;
- the aggregation method and, for DP releases, the mechanism, per-release epsilon/delta and a reference to the budget-accounting receipt;
- a detached signature/attestation over the canonical payload and authenticated transport metadata.

The JSON contract intentionally has no fields for account/customer name, meter serial/ID, address, GPS coordinates, phone/email, appliance data, raw interval series or arbitrary extension objects. Unknown fields fail closed. A schema pass is a shape check; it does not authenticate a meter, prove the cohort count, verify a DP accountant, or turn untrusted data into true measurements.

## Data classes and trust boundaries

| Data class | Examples | Where it should live | Default allowed recipients |
|---|---|---|---|
| Individual operational readings | meter ID, exact interval Wh, voltage/current samples | Meter or authorized utility/billing system; short bounded retention | Customer and authorized meter/billing roles |
| Local analytics features | rolling load average, local anomaly score, voltage-quality flag | Customer/site gateway where feasible | Explicitly authorized local operator |
| Utility aggregate | cohort Wh sum, broad time bucket, quality/coverage | Aggregate service and restricted grid operations store | Grid operations under documented purpose |
| Research/public statistic | cohort aggregate with formal noise and release accounting | Separate publication pipeline | Approved research or public audience |
| Mycelix provenance | policy/consent receipt, signed aggregate receipt, aggregate manifest digest | Permissioned evidence/control plane | Purpose-limited authorized participants |

Do not place individual consumption readings, stable household identifiers, raw per-meter hashes or detailed energy time series in a distributed hash table, public ledger or immutable receipt. Hashes of low-entropy readings can sometimes be guessed by enumeration; hashing is not anonymization. Receipts should bind the policy and aggregate batch, not leak the value being protected.

## Privacy policy: layered controls

1. **Minimize at source.** Read only what is needed for the declared purpose. Keep detailed household histories local or in the authorized billing boundary; compute coarse windows at the edge.
2. **Aggregate before satellite egress.** Prefer one gateway uplink for a whole qualifying group over one satellite packet per household. When secure aggregation is needed, ensure the aggregation protocol does not reveal individual masked inputs to the aggregator.
3. **Suppress small cohorts.** Prototype policy default is at least 20 contributing meters per residential aggregate, with the final threshold raised where the sensitivity, geography, side information or release cadence warrants it. This is an engineering floor, not a universal proof of anonymity. Do not emit a small-cohort value under an alternate label.
4. **Prevent differencing.** Maintain a release/query ledger. Do not let recipients subtract overlapping cohorts, adjacent time windows or slightly different filters to reconstruct one participant's use. Coarsen windows, merge cohorts, rate-limit outputs or suppress results where needed.
5. **Use differential privacy for research and public reports.** Apply a reviewed mechanism to bounded per-household contributions, record the privacy parameters and account cumulative composition across every release. Prototype validator cap: per-release epsilon ≤ 1 and delta ≤ 1e-6. These values are conservative contract defaults, not a complete privacy guarantee. The real privacy budget and contribution bounds need a documented threat/utility analysis and an accountant; repeated releases consume budget.
6. **Encrypt end-to-end and authenticate.** Link encryption protects a network hop. Application-level encryption must continue to the authorized collector. Use reviewed standard protocols and managed keys, mutual authentication, rotation/revocation, signed batches and anti-replay state. Consider post-quantum migration planning for data whose confidentiality needs to last many years.
7. **Minimize metadata.** Randomize batch identifiers; avoid a stable meter pseudonym, precise point coordinates, customer-specific send cadence and unnecessary device identifiers in payloads. Satellite operators can still observe radio/session timing, network identifiers, approximate coverage geography and traffic volume. Padding/batching and a shared site gateway reduce, but do not erase, those observations.
8. **Bound retention and access.** State the purpose, recipients, retention period, deletion behaviour for queued batches, subject access/correction process, key custody and audit policy. Do not silently reuse demand data for eligibility, pricing, credit scoring or occupant surveillance.
9. **Keep monitoring separate from actuation.** V1 is telemetry-only. Any future control channel needs a separate threat model, authorization, safety case, local fail-safe, explicit customer/utility authority and replay protection. A signed telemetry record does not authorize a command.

### Billing accuracy is a separate requirement

Do not add differential-privacy noise to an individual bill or use a cohort sum in place of a required revenue-meter reading. Billing remains a separate authorized path with its own metrology, legal, dispute and correction requirements. The analytics path may deliberately reveal less than the billing path; users should be told which entity receives which data.

## Utility and performance model

Satellite telemetry is most promising when:
- sites are remote, off-grid, mountainous or poorly served by terrestrial communications;
- a microgrid needs periodic aggregate demand, generation or storage telemetry;
- a distribution site needs resilient backup communications when the primary backhaul is down;
- only compact, delayed, aggregate reports are required.

Prefer comparing terrestrial cellular, fixed wireless, fibre, RF mesh and power-line communications for dense urban smart-meter rollouts. A satellite connection does not increase meter accuracy and does not solve a missing sensor. Before selection, test site-specific coverage, terminal certification, line of sight/antenna placement, energy budget, message-size and message-frequency pricing, delivery latency, outages, retransmissions, roaming and provider metadata exposure. Do not claim commercial availability in South Africa for an untested operator or service plan.

Use store-and-forward batches for routine energy totals. Local safety and grid-protection functions must not depend on satellite latency or intermittent connectivity. On lost backhaul, retain only a bounded encrypted queue, report freshness/coverage honestly, expire stale items by policy, and avoid repeating the same aggregate with new identifiers.

## South Africa-specific governance

Electricity usage linked to a person, dwelling, account or stable location can be personal information. Before a field deployment, obtain local legal/privacy review and a purpose-specific processing basis, document the responsible party/operator roles, access rights, retention and security safeguards, and carry out a risk assessment. South Africa's POPIA establishes conditions for lawful processing of personal information; see the [National Treasury POPIA overview](https://ocpo.treasury.gov.za/POPIA/default.aspx). That overview is not a complete legal determination for this proposed system. The [EU smart-grid data-protection impact-assessment material](https://energy.ec.europa.eu/topics/markets-and-consumers/smart-grids-and-meters/data-protection-impact-assessment-smart-grid-and-smart-metering-environment_en) is a useful engineering risk-assessment reference, not South African legal authority.

## Symthaea / Mycelix role

- **Edge component:** schema validation, clock/freshness checks, bounded energy totals, cohort eligibility, quality/missingness checks and local anomaly detection.
- **Symthaea analytics:** forecast aggregate demand with uncertainty; distinguish missing/stale telemetry from low use; abstain when coverage is insufficient; evaluate model utility against a no-model baseline.
- **Mycelix evidence plane:** versioned purpose/consent policy, access/aggregation receipt, detached signer identity/credential and aggregate manifest digest. Keep raw meter readings out of shared DHTs and immutable logs.
- **Sol/energy coordination layer:** if later used for energy sharing or demand response, carry approved aggregate capacities and event status, not household histories. Interoperability with demand-response protocols such as [OpenADR](https://www.openadr.org/about-us) can be investigated later, but v1 has no actuation/control plane.

## Offline prototype contract

Files:
- `docs/research/energy_telemetry_envelope.schema.json`
- `tools/energy_telemetry/privacy_contract.py`
- `tools/energy_telemetry/release_guard.py`
- `tests/test_energy_telemetry_privacy_contract.py`
- `tests/test_energy_telemetry_release_guard.py`

The original deterministic parser and release-policy suites contain 51 cases: 33 envelope/parser cases and 18 release-policy cases. Later increments add a cryptographic protected-envelope suite, durable queue recovery tests, and in-memory delay-tolerant queue tests. The original `ReleasePolicyGuard` remains an in-memory reference. The `DurablePrivacyLedger` persists release reservations and cumulative participant budgets across restarts, using HMAC tags rather than raw participant tokens, and caps input cohort materialization at 10,000 participants. The new `QualifiedOutputVerifier` checks an Ed25519-signed canonical attestation against deployment-pinned trust and qualification policy, exact reservation context and output digest before routing to the internal output-binding primitive. It authenticates the attester's assertion; it does not implement noise generation, prove that the attester witnessed actual execution, or verify contribution clipping independently. Tests use a fixed test-only signing key. All added cases are implemented but have not been established as passing by the hosted exact-head CI.

The aggregate validator checks exact keys, UTC/time-window consistency, cohort floor, bounded integer energy value, finite coverage fraction, privacy-mechanism/recipient compatibility and per-release DP parameter limits. The signed attestation gate lives in `tools/energy_telemetry/qualified_output.py` and its contract tests are included in the durable privacy-ledger test module. The durable reservation implementation is documented in [DURABLE_DP_RELEASE_ACCOUNTING.md](https://github.com/Luminous-Dynamics/symthaea/blob/research/privacy-preserving-satellite-energy-telemetry/docs/research/DURABLE_DP_RELEASE_ACCOUNTING.md). The maintained-library selection and formal qualification gate is documented in [DP_MECHANISM_QUALIFICATION_GATE.md](https://github.com/Luminous-Dynamics/symthaea/blob/research/privacy-preserving-satellite-energy-telemetry/docs/research/DP_MECHANISM_QUALIFICATION_GATE.md). The separate authenticated-envelope prototype encrypts event metadata and payload with ChaCha20-Poly1305 and signs its canonical outer envelope with Ed25519 using the maintained `cryptography` library. The durable receiver inbox now verifies sender signatures, commits exact ciphertext and a replay record to SQLite, and only then creates a signed receiver ACK. Trusted-key provisioning, key rotation/revocation, durable nonce-use policy and production receiver operations remain outside the prototype.  A dedicated [pinned GitHub Actions workflow](https://github.com/Luminous-Dynamics/symthaea/blob/research/privacy-preserving-satellite-energy-telemetry/.github/workflows/energy-telemetry-privacy-tests.yml) compiles and runs the offline tests on relevant branch/PR changes. It also rejects duplicate JSON keys, non-standard `NaN`/`Infinity` constants and malformed container types without crashing. It does not establish cryptographic authenticity, hardware provenance, meter calibration, the reality of claimed cohort membership or a mathematical privacy proof.

## Research and standards base

- [NIST IR 7628 Rev. 1 — Guidelines for Smart Grid Cybersecurity](https://csrc.nist.gov/pubs/ir/7628/r1/final): risk analysis for smart-grid security and privacy.
- [ITU-T X.1332 — Security guidelines for smart metering services](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-X.1332-202003-I%21%21PDF-E&lang=e&type=items): threats from invalid measurements, system misuse and third-party access.
- [The privacy cost of fine-grained electrical consumption data (Scientific Reports, 2025)](https://www.nature.com/articles/s41598-024-78285-7): supports treating fine temporal detail as sensitive.
- [DP²-NILM (2024)](https://doi.org/10.1016/j.rser.2023.114091): distributed load-monitoring with differential-privacy and utility trade-offs.
- [NIST — Analyzing Data Privacy for Edge Systems](https://www.nist.gov/publications/analyzing-data-privacy-edge-systems): local/edge privacy mechanisms evaluated on smart-meter streaming data.
- [3GPP NB-IoT/eMTC support for Non-Terrestrial Networks](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3747): standards work supporting non-terrestrial connectivity; this is not a claim of a particular local service's availability or price.
- [OpenADR](https://www.openadr.org/about-us): interoperable demand-response messaging to consider only if a later, separately secured control layer is approved.

## Acceptance gates before any live pilot

1. Threat model names who must not learn individual readings: satellite carrier, gateway operator, utility analytics staff, researcher, public and other customers.
2. Data-flow test proves no individual reading or identifying field leaves the local boundary in aggregate mode.
3. Cohort size, geography precision, output cadence and overlapping-query controls are validated adversarially.
4. Differential-privacy accountant composes budgets across releases and records auditable receipts.
5. The prototype has contract tests for signature and AEAD tampering, protected-envelope persistence/replay, receiver commit-before-ACK, duplicate acceptance, lost-ACK retries, tampered transmissions, offline queue behavior and stale-data gating. Production key rotation, durable nonce/key usage state, crash/power-loss proof, and production receiver anti-replay operations remain open implementation and verification gates.
6. Meter measurement accuracy, clock uncertainty, calibration and dropout rates are reported separately from privacy.
7. Link budget, coverage, unit economics and alternative terrestrial channels are assessed at the actual site.
8. Local legal review, participant notice/consent where applicable, retention schedule and incident procedure are approved.
9. Tests, schema and threat model pass exact-head CI; independent review is completed before deployment.

## Claim ceiling

This is an architecture and a set of offline research prototypes. It does not prove anonymity, regulatory compliance, production cryptographic correctness, safe key management, energy savings, meter accuracy, satellite coverage, cost effectiveness, availability in Johannesburg, or safe control of the electrical grid. The next step is simulation and adversarial testing with synthetic values—not live household collection.
