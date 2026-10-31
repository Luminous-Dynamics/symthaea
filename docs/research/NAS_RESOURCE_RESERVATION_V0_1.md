# NAS resource reservation ledger v0.1

Tracking: [NAS-RESERVE-001 / #9485](https://github.com/Luminous-Dynamics/symthaea/issues/9485)

Implementation branch: \`research/nas-resource-reservation-v0.1\`

## 1. Purpose

The #9483 preflight verifier recomputes the value-of-computation action, objective/hardware identity, anytime confidence stream, and resource frontier. It is read-only. Two callers could both pass preflight against the same available budget and subsequently oversubscribe a device or an exclusive instrument.

This module adds a reference compare-and-swap (CAS) reservation state machine. It consumes an in-process \`VerifiedMeasurementReservation\` capability that can only be obtained by calling the complete measurement preflight. The capability binds the exact preflight receipt, action-scope digest, resource cost vector, budget/registry identities, and reservation-ledger head seen by preflight.

## 2. Resource lifecycle

Each reservation is keyed by the exact action-scope SHA-256 and follows:

\`UNSEEN -> RESERVED -> COMMITTED\`

or:

\`UNSEEN -> RESERVED -> RELEASED\`

or:

\`UNSEEN -> RESERVED -> EXPIRED\`

- \`RESERVED\` holds every resource dimension in the exact cost vector.
- \`COMMITTED\` is terminal and permanently consumes the full reserved quantity. It is deliberately conservative: even when a downstream process crashes or the physical action fails, the reservation is not refunded automatically.
- \`RELEASED\` is allowed only before commit and frees held capacity.
- \`EXPIRED\` is a logged transition once the trusted logical epoch reaches the lease expiry; it frees only uncommitted capacity.

A committed reservation cannot be released. A new attempt requires a new action scope, and therefore a new idempotency key.

## 3. Compare-and-swap protocol

The reservation ledger has a revision head SHA-256. Before preflight, the action scope binds the exact current reservation-ledger head. The preflight result is wrapped in an unforgeable-in-process (non-serializable, private-field) capability by \`prepare_measurement_reservation\`.

The reserve call requires:

- caller's \`expected_head_sha256\`;
- private verified preflight capability;
- expiry epoch within the maximum lease duration.

It verifies that the expected head equals the current ledger head and that the capability's observed head is the same. Only the first competing request against a given head can mutate the ledger. Any other distinct request from that same head fails stale and must refresh the ledger and repeat preflight. This matters because a stale preflight must not spend resources using an old budget snapshot.

The same action-scope digest with exactly the same action/receipt/cost is an idempotent retry: it returns the existing reservation state without appending another event, even after the caller lost the original response. The same idempotency key with different action, preflight, or cost evidence is rejected.

## 4. Budget accounting

The ledger genesis binds the run ID, exact resource registry, initial budget, and maximum lease duration. For each resource dimension, spendable headroom is:

\`total - base_already_spent - protected_reserve - active_reserved - committed_consumption\`

All arithmetic is integer and checked for underflow/overflow. Resource identity, coordinate order, unit, cost-model digest, and registry digest must match exactly. At least one resource quantity must be positive.

The existing resource frontier and preflight perform benefit eligibility and Pareto non-dominance checks. This ledger is the final component-wise capacity reservation; it never converts unlike resource types into a weighted scalar.

## 5. Event-chain integrity

Every transition records:
- contiguous event sequence number;
- reservation/action identity;
- preflight-receipt digest;
- exact resource-cost-vector digest;
- event kind and epoch;
- exact lease expiry;
- reason code;
- previous event digest;
- domain-separated event digest.

The record digest binds current reservation state, complete preflight receipt, and latest event pointer. The top-level ledger digest binds the full configuration, all records, events, and current epoch.

The verifier:
1. validates resource registry and starting budget;
2. recomputes the genesis digest;
3. verifies every reservation record and cost vector;
4. replays the event chain and checks contiguous sequence, monotone epochs, parent hashes, immutable reservation identity, and legal state transitions;
5. reconstructs final record state and latest-event pointers;
6. recomputes per-resource active/committed usage and rejects budget/reserve breaches;
7. rejects any RESERVED lease whose expiry epoch has elapsed without an EXPIRED transition;
8. verifies the final ledger digest.

JSON structs deny unknown fields. Round-trip verification is included.

## 6. Strict production limitation

**This crate module is not a distributed atomic transaction service.** A \`&mut self\` reference serializes mutations only within one process. It does not protect concurrent processes, machines, or services; it is not durable, has no database isolation, no write-ahead log, no trusted lease clock, no authentication, and no crash-recovery daemon.

The expected-head contract defines the operation that a production adapter must implement in one durable serializable transaction or compare-and-swap database update. The production service must also:
- authenticate the preflight capability and action caller;
- atomically compare ledger head and reserve every resource coordinate;
- persist the updated head, record, and event before acknowledging success;
- handle lost responses through the scope-digest idempotency key;
- recover leases using a trusted monotonic clock and durable schedule;
- commit the reservation immediately before external action, with conservative non-refund after execution begins;
- prevent duplicate device/instrument execution, not merely duplicate accounting.

Until such an adapter exists and passes multi-process crash/replay tests, this module is a reference protocol and research prototype only. It must not be presented as production concurrency safety.

## 7. Qualification tests

Fifteen Rust test declarations cover:
- genesis and empty ledger replay;
- exact-head reserve and resource headroom;
- stale-head competition (first reservation succeeds, second fails);
- idempotent same-scope retry and conflicting same-key cost;
- overcommit/protected reserve rejection;
- irreversible commit;
- pre-commit release and capacity restoration;
- monotonic expiry and capacity reclamation;
- rejection of post-expiry commit;
- registry unit/cost-model mismatches;
- event/record/top-level digest tampering;
- rejection of an expired-but-still-reserved record;
- JSON round-trip and unknown-field rejection;
- lease-horizon rejection.

These tests are authored, not passing tests. The current authoring environment has no \`cargo\`, \`rustc\`, or \`rustfmt\`, so this code has not been compiled or executed.

## 8. End-to-end path and next gates

The #9483 end-to-end synthetic measurement preflight fixture now also:
1. creates a resource reservation ledger against the same registry and initial budget;
2. binds the ledger head into the exact measurement scope;
3. verifies and obtains the private in-process capability;
4. reserves against the expected head;
5. commits the reservation;
6. verifies the resulting event/record/ledger chain.

The next gates remain:
- implement #9388 private exact-scope \`VerifiedVoCQualification\` capability before candidate promotion;
- atomically debit #9329/#9104 run-wide pruning risk alongside pruning resource reservations;
- implement a durable transactional adapter with concurrent multi-process tests;
- implement seed custody and independent randomized-QMC replicate manifests;
- benchmark reservation throughput/replay at 1, 10, 100, 1,000, and 100,000 records before choosing a persistent representation.

## 9. NAS quality benchmark and industry context

The action/reservation plumbing is a safety and reproducibility contribution; it does not by itself improve architecture quality or search efficiency.

For NAS quality, compare at matched task/search-space/objective/constraint protocols against:
- ConNAS (ICLR 2026), which reports architectures within 0.14% of the optimal feasible NATS-Bench architecture under its tested constraints: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7c9f851081499ed5c60c51e312d3235f-Abstract-Conference.html
- EdgeVolution (Communications Engineering, June 2026), which performs hardware-in-the-loop search and measures latency/power on target MCUs throughout optimization: https://doi.org/10.1038/s44172-026-00708-2
- BoTorch qEHVI/HVKG, random search, regularized evolution/NSGA-II, and multi-fidelity/predictor NAS baselines.
- NATS-Bench-MCU v2, with 15,625 architectures, 14,281 physically profiled nRF5340 architectures and 1,344 flash-infeasible designs: https://doi.org/10.5281/zenodo.21644470

The 2026 RQMC paper by Jain, Hickernell, Owen and Sorokin, *Empirical Bernstein and betting confidence intervals for randomized quasi-Monte Carlo*, is especially relevant for later sample-efficiency work: https://doi.org/10.1093/imaiai/iaag003. Under the paper's bounded-integrand and RQMC variance assumptions, the optimal total-budget split between independent randomizations and points per replicate depends on the observed convergence exponent. Our hypervolume-dominance integrand is discontinuous, so this rate must be measured rather than assumed. That research is tracked in #9482.

Primary NAS outcomes should include held-out hypervolume/portfolio-class regret, feasible-architecture recall, false-feasible rate, physical constraint satisfaction, total independent measurement count, and total vector-resource cost to reach a fixed held-out regret target. Reservation throughput and failure recovery are separate system metrics.

Aspirational quality targets (not achieved): >=95% simultaneous sequential coverage, <=5% family-wise false decisions at a preregistered 5% budget, >=20% fewer independent measurement replicates than Hoeffding after a qualified variance-adaptive method is added, and >=10% lower all-in vector-resource cost to a fixed held-out Pareto-regret target than the strongest matched-budget baseline without deployment/calibration regression.

## 10. Claim ceiling

No Rust build/test/CI pass, multi-process atomicity, physical deployment, hardware data acquisition, SOTA improvement, or production authorization is claimed. The evidence chain is more complete; its operational and empirical qualification is not complete.
