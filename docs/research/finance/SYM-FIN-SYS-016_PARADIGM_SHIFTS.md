# SYM-FIN-SYS-016 — Paradigm shifts for socioeconomic interoperability

**Status:** research proposal / design note  
**Claim ceiling:** synthetic research only; no policy endorsement, production authority, real-world outcome claim, or settlement permission.

## Thesis

The field does not need another universal currency, another opaque welfare score, or a larger central ledger by itself. The more useful paradigm is a **federated economic protocol**: independently governed communities can coordinate resources and commitments through explicit semantic adapters, while evidence provenance, authority, rights, physical capacity, and settlement finality remain separate and inspectable.

The goal is not one winning economic regime. It is to discover which mechanisms remain viable under stated constraints, for whom, under which shocks, at what distributional cost, and with what exit options.

## Six shifts

### 1. From one universal ledger to federated semantics

A shared vocabulary can make heterogeneous systems interoperable without requiring them to share one database, currency, governance regime, or settlement authority. The official ValueFlows vocabulary distinguishes:
- **Intent:** desired or proposed economic flow;
- **Commitment:** promised/planned flow;
- **EconomicEvent:** a flow that actually happened in the represented domain; and
- **Claim:** a future reciprocal event claim triggered by a prior event.

Do not serialize Mycelix records into similarly named ValueFlows records by name alone. Declare, version, and test the mapping; preserve distinctions and profile extensions; reject unsupported fields rather than silently dropping them.

Reference: [ValueFlows specification](https://www.valueflo.ws/specification/vfspec/) and [model explanations](https://www.valueflo.ws/specification/model-text/).

### 2. From a single “verified” bit to provenance-typed evidence

“Schema-valid,” “cryptographically intact,” “issued by an identified party,” “independently observed,” “estimated,” and “synthetic-only” are different properties. A simulation receipt must never gain real-world authority merely because it parses, matches a schema, or is signed.

The Mycelix reference separates `EvidenceStatus` from `EvidenceOrigin`: a result can be `Verified` as a simulation while remaining `SyntheticSimulation` in origin. Authorization and obligation gates require both `Verified` status and `LiveDomain` origin, and delivery verification plus dispute reassessment fail closed for synthetic-origin records. This is an explicit state-machine barrier, not cryptographic authentication: origin is caller-supplied. A production design still needs issuer identity, signature/key lifecycle, revocation, evidence freshness, policy-specific trust roots, and an independent verifier.

Relevant general model: [W3C Verifiable Credentials Data Model 2.0](https://www.w3.org/TR/2025/REC-vc-data-model-2.0-20250515/) describes claims exchanged among issuers, holders, and verifiers, and includes credential status concepts. A credential representation or integrity check does not alone establish the real-world truth of every underlying claim.

### 3. From transaction-centric thinking to explicit cross-domain finality

A physical resource reservation, a service commitment, a delivery assertion, independent verification, and financial settlement are not one atomic event when they span different systems. Record each as a separate transition with identity, evidence, time, and failure semantics.

The current reference deliberately consumes verified capacity before financial posting: the physical capacity stays consumed if settlement cannot post. That may require reconciliation or dispute handling, but it does not fabricate capacity availability. Future adapters must represent unknown outcomes as **indeterminate**, not infer success or failure from a timeout.

This aligns with the useful portion of the [BIS unified-ledger discussion](https://www.bis.org/publications/aer-2025/next-generation-monetary-financial-system): programmability may reduce reconciliation friction, but that is not an argument that every domain must be collapsed into one ledger. Federation and explicit finality profiles remain essential.

### 4. From a single welfare score to a constrained outcome vector

Do not declare one regime “best” from a weighted scalar that hides distributional harm. Report a vector with explicit units, scope, missingness, uncertainty, and affected populations. At minimum keep separate:
- essential access and unmet need;
- physical capacity, maintenance, and ecological overshoot;
- distribution of purchasing power, assets, housing, authority, and verification power;
- bridge concentration, dependency asymmetry, substitutability, and exit accessibility;
- stock/flow consistency, liabilities, and intergenerational effects.

Use Pareto comparisons and publish the trade-offs. A rights or ecological limit should be a constraint that cannot be offset by a gain in an unrelated metric. Metric weights, if used for an experiment, must be declared and sensitivity-tested rather than hidden in the evaluator.

### 5. From winner-seeking simulations to falsifiable mechanism experiments

A simulator should try to break each mechanism, not merely produce a persuasive trajectory. Predeclare the causal question and test it against:
- multiple random seeds and initial institutional conditions;
- held-out shocks, parameter perturbations, mutation-order permutations, and replay determinism;
- no-bridge, bridge-removal, adoption-ablation, and random-mutation controls;
- independent stock/flow and power-concentration oracles;
- adverse selection, verifier capture, bridge capture, capacity shocks, and asymmetric exit costs.

Report failed runs and counterexamples. Distinguish implementation correctness, internal model validity, causal evidence within the simulation, and external validity; success at one level does not establish the next.

### 6. From reputation as authority to least-authority capabilities

A recommendation, reputation score, proof of past reliability, or institutional label should not itself grant a payment, capacity reservation, policy override, or privilege over someone’s essential needs. Explicitly authorize a narrow action for a specific scope, amount, resource, time window, and policy generation; re-check at the point of effect; expire and revoke capabilities; and log denials and appeals as well as approvals.

The intended architecture is a **capability floor with bounded actions**, not a globally trusted optimizer. Human/community governance and legal authority remain separate from simulation results.

## Current implementation slice

- Mycelix keeps evidence status and origin orthogonal; five authored regression tests cover synthetic-origin policy, obligation, delivery-claim, verification-result, and dispute-reassessment barriers. Caller-supplied origin remains an explicit provenance limitation.
- Delivery claims must name the provider fixed by the request. This checks logical identity in the reference model, not a cryptographic signature.
- Symthaea receipt loading rejects duplicate JSON object keys and Python's non-standard `NaN` / `Infinity` constants before schema or semantic validation.
- Eleven standard-library unit tests cover strict JSON parsing and metric-catalog semantics, including unknown metric IDs and schema-valid but incorrect metric dimensions. The validator pins all 26 negative-control IDs with a 10/16 schema/semantic split; it recomputes the scenario-registry SHA-256 from exact repository bytes and checks both scenario IDs and metric ID/dimension pairs against that registry. Other digest-shaped receipt fields still need binding to their source artifacts.
- Neither track is qualified by this note. Authored tests are not passing tests. CI must run against the exact final head.

## Acceptance gates before integration

1. Execute all eight pinned Python unit tests and the receipt validator on the exact PR head; record the SHA and full result.
2. Execute Rust formatting and all 33 authored reference tests with the pinned toolchain; record the SHA and job logs. Until the exact-head job completes, these remain authored, not passed.
3. Add a conformance corpus for any ValueFlows adapter: positive mappings, unsupported-field failures, claim/intent/commitment/event distinction, and stable round-trip behavior where representable.
4. Extend exact-byte digest binding to every receipt field that purports to identify an input artifact; currently only `scenario_registry_digest` is recomputed from exact bytes, and `scenario_id` is checked against the registry's declared IDs. The other digest-shaped fields are not proof that their referenced inputs match.
5. Add model-level invariants for conservation, stock/flow reconciliation, capacity non-overcommit, liability accounting, and no authority-bearing transition from synthetic evidence.
6. Run adversarial simulations and publish a failure/counterexample register before making comparative policy claims.

## References

- ValueFlows, [VfSpec](https://www.valueflo.ws/specification/vfspec/) and [model explanations](https://www.valueflo.ws/specification/model-text/).
- W3C, [Verifiable Credentials Data Model v2.0, Recommendation, 15 May 2025](https://www.w3.org/TR/2025/REC-vc-data-model-2.0-20250515/).
- Bank for International Settlements, [The next-generation monetary and financial system](https://www.bis.org/publications/aer-2025/next-generation-monetary-financial-system).
- JSON Schema, [Specification](https://json-schema.org/specification) (current version listed as 2020-12).

This note is a testable research direction, not a claim that the proposed system is fair, correct, production-ready, or proven to improve real economies.

## Benchmark targets

The research lab now has a staged qualification contract: receipt controls; exact-byte artifact provenance; deterministic replay; accounting/capacity/authority invariants; cross-regime common-seed experiments; adversarial recovery; and pinned-runner performance budgets. See [SYM-FIN-SYS-016_BENCHMARK_TARGETS.md](SYM-FIN-SYS-016_BENCHMARK_TARGETS.md). The 12,000-run baseline/shock campaign, 16,800-run directed interoperability campaign and performance thresholds are proposed future targets, not completed benchmarks.

The artifact manifest intentionally excludes itself and the receipt fixture to avoid circular hashing. The receipt records the exact-byte manifest digest; the manifest hashes source inputs used to validate the receipt and records unavailable run-artifact digests as unverified. This is an identity/completeness check, not a signature, issuer-authentication or physical-truth proof.
