# AR-5B5: Native EventTape Vector and Noisy Replicate Replay v0.2

**Status:** implementation draft; tests authored but not executed.  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/9692, https://github.com/Luminous-Dynamics/symthaea/issues/9663, https://github.com/Luminous-Dynamics/symthaea/issues/9469.  
**Claim ceiling:** native typed objective actions, costed replicate attempts, byte/identity receipt reconstruction, and independent EventTape/AnytimeHistory replay. No evaluator correctness, signed verifier authenticity, independent-noise proof, matched-policy parity, benchmark win, or SOTA claim.

## 1. Design change

The old scalar EventTape encoded candidate evaluation as a single score/uncertainty pair. That representation is insufficient for a multi-objective campaign: it hides objective ordering, directions, reference-point semantics, per-replicate sampling assumptions, and the distinction between a failed measurement and an absent one.

EventTape v0.2 introduces a frozen typed objective contract and two separate action paths:

- **DeterministicVector:** one complete finite objective vector per candidate evaluation.
- **NoisyReplicates:** each replicate attempt is a separate event with its own ID, seed, outcome, cost receipt, and (on success) vector plus artifact/manifest/byte-receipt digests.

The scalar v0.1 representation remains readable. Typed vector/noisy actions require a v0.2 header with an objective contract. A typed multi-objective tape rejects the legacy scalar CandidateEvaluation action to prevent mixed-score semantics.

## 2. Frozen objective contract

Schema: symthaea.ar5b4.event-tape-objective-contract v0.1.0.  
EventTape header version: v0.2.0.

The header freezes:

- ordered objective IDs and maximize/minimize directions;
- raw hypervolume reference-point coordinates in that exact order;
- deterministic-vector or noisy-replicate mode;
- for noisy mode, certified single-observation lower/upper bounds and each bounds-certificate digest;
- sampling protocol and digest;
- noise model and digest;
- family-wise alpha, maximum candidate family, and minimum replicate count.

The objective contract is content-digest-bound by the EventTape header. Objective order is semantic, not cosmetic. The noisy replay adapter deterministically converts the frozen contract to the same NoisyObservationPlan schema consumed by the confidence analyzer; the resulting plan digest is separately retained in each byte-verification manifest/receipt and preregistered in AnytimeHistory.

The confidence analyzer currently supports two through sixteen objective coordinates. AnytimeHistory's current exact-HV history header supports two through four coordinates; therefore exact joint reconciliation is currently limited by the narrower AnytimeHistory contract, even though EventTape and the noisy confidence report accept up to sixteen.

## 3. Event semantics

### Deterministic vector evaluation

Action intent: EvaluateCandidateVector.  
Action outcome: CandidateMultiObjectiveEvaluation.

A Verified outcome requires a complete finite vector of the frozen dimension, a frozen objective-contract digest, verifier identity/receipt, and evidence, manifest, and byte-receipt digests. Non-verified outcomes cannot expose a partial objective vector or successful evidence and require an explicit failure reason.

### Noisy replicate attempt

Action intent: ObserveCandidateReplicate.  
Action outcome: CandidateNoisyObservation.

The intent binds candidate digest, decision digest, replicate ID, sample seed, direction, and objective-contract digest. The outcome carries one of:

- Succeeded
- Failed
- Invalid
- TimedOut
- VerifierRejected

Every attempt is a normal EventTape event, so its resource receipt and normalized cost are charged whether it succeeds or fails. A successful observation requires a complete vector within the frozen per-observation bounds and all evidence/manifest/byte-receipt digests. Failed/invalid/timed-out/rejected observations require a reason and cannot publish objective values or success receipts.

Candidate decision identity cannot change within a campaign. Replicate IDs and sample seeds cannot repeat within a candidate. These constraints are provenance controls; they do not establish statistical independence between physical measurements or across candidates.

## 4. Byte-verified replay adapter

The new event_tape_noisy module consumes a map of caller-resolved evidence bundles keyed by EventTape event ID. Each successful replicate requires exactly one bundle containing:

- replicate-scoped evidence manifest;
- exact raw observation artifact bytes;
- byte-verification receipt with verifier ID and build digest.

The adapter does not perform I/O. A content-addressed storage resolver must enforce locator namespace and byte ceilings before passing bytes to this function.

For each successful event, replay verifies:

1. EventTape intent/outcome contract digests equal the frozen header.
2. Manifest digest equals the EventTape manifest digest.
3. Receipt digest equals the EventTape byte-receipt digest.
4. Manifest artifact digest equals the evidence digest on the sample.
5. Receipt run/campaign, source commit, candidate, decision, replicate ID, seed, and plan hash equal the exact event.
6. Receipt verifier-build digest equals the verifier digest frozen in the EventTape header.
7. The raw bytes reconstruct the canonical JSON envelope, exact identity and exact ordered objective vector.
8. The replicate is then appended to a fresh NoisyParetoStudy.

Missing bundles, extra/unreferenced bundles, byte substitutions, vector substitutions, identity changes, and receipt mutations fail closed. Failed attempts remain counted and charged by EventTape but do not enter sample means or confidence intervals.

**Receipt boundary:** this proves byte and identity correspondence. The current receipt records verifier ID/build and is reconstructable, but is not a signed attestation. The separate generic EventTape verifier_receipt_sha256 remains an opaque digest in this tranche; no corresponding verifier-attestation bytes or signature are resolved by this adapter. A hash chain is not a signature, remote-store authentication, evaluator correctness proof, or statistical-independence certificate.

## 5. Independent replay against AnytimeHistory

reconcile_noisy_event_tape_with_anytime_history requires:

- EventTape campaign ID equals AnytimeHistory run ID;
- source commits are exact matches;
- the AnytimeHistory header preregisters the exact NoisyObservationPlan derived from the EventTape contract;
- objective IDs, order, and directions match;
- every replicate attempt matches on candidate, decision, replicate ID, seed, status, ordered objective vector, evidence digest, and failure reason;
- each independently reconstructed NoisyParetoReport is identical;
- attempted, successful, and failed counters reconcile.

The EventTape-only replay checks byte receipts before aggregation. The AnytimeHistory replay is a second reconstruction path over its own hash-chained observations and cost receipt. The returned reconciliation digest commits to both reports.

A verifier-rejected EventTape attempt maps to AnytimeHistory's Invalid outcome because the current AnytimeHistory status enum does not have a verifier-rejected variant. The exact failure reason is still compared. Add a distinct AnytimeHistory status in a future schema if downstream consumers need to distinguish these causes.

## 6. Remaining integration boundaries

This tranche does not yet implement:

1. An external content-addressed artifact resolver. The adapter receives bytes as input and must be placed behind a bounded resolver.
2. Signed verifier attestations or trusted verifier identity. Build digests are recorded and checked against the header, but are not cryptographically signed here.
3. A matched-campaign multi-policy adapter for vector/noisy replicate actions. The current matched-campaign reconciler still handles scalar evaluation and proposal actions only; it should not be used to claim matched noisy-MOO results.
4. A deterministic-vector Pareto/Anytime archive adapter. v0.2 validates and records deterministic vectors, while this adapter currently handles noisy Pareto replay.
5. A benchmark driver or measured comparison against external optimizers.
6. A statistical test proving sample independence, stationarity, or evaluator correctness.

These are explicit follow-on work, not implicit properties of v0.2.

## 7. Benchmarks and acceptance gates

### Public comparators

- COCO bbob-biobj, bbob-noisy, and bbob-constrained should be run as separate regimes; the archive supports large-scale tracks as well. Use exact archived instances and versioned wrappers: https://coco-platform.org/data-archive/index.html
- Include parametrized outlier noise, heteroskedasticity, shared-latent cross-objective correlation, serial dependence, drift, and informative missingness. COCO's 2025 workshop agenda explicitly identifies noisy, constrained, mixed-integer, multi-objective, and outlier-noise gaps: https://coco-platform.org/workshops/bbob2025.html
- Keep NSGA-II, MOEA/D-DE and SMS-EMOA as classical MOO baselines; choose pinned PlatEMO algorithms by regime rather than treating the catalogue as a single comparator.
- Use BoTorch qNEHVI/qLogNEHVI and qNParEGO when surrogate assumptions fit the noisy/batched problem. Its tutorial recommends qNEHVI for noisy or batched MOO: https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo
- Retain Sobol, random search, uniform replicate allocation, interval-width, interval-width-per-cost, racing/successive halving, and the exact finite-state EVSI oracle as controls.

### Metrics

Report per independent problem/task family:

- final and anytime hypervolume with frozen reference points, IGD+, additive epsilon, and attainment indicators where appropriate;
- target-attainment runtime and ECDF over target levels, not just a single final score;
- false-certain-dominance, false-feasibility, false-stop, and missed-target rates against known ground truth;
- empirical simultaneous coverage, interval width, uncertainty calibration, and assumption-violation detection;
- all-in host CPU, elapsed wall time, worker time, accelerator, peak memory, energy, evaluator/verifier calls, setup, acquisition, retries, and artifact retrieval;
- exact-byte resolution, manifest/receipt reconstruction, wrong-vector substitution, missing evidence, and EventTape/AnytimeHistory divergence rates.

COCO uses runtime-to-target and empirical cumulative-distribution performance assessment; retain that style for anytime optimization, but add evidence and failure accounting as separate dimensions: https://numbbo.github.io/coco-doc/perf-assessment/

### Proposed gates (not results)

- 100% of successful replicate events have one byte-resolved manifest and reconstructable byte receipt.
- Zero accepted candidate/decision/replicate/seed/plan/vector substitutions in adversarial tests.
- Zero EventTape/AnytimeHistory attempt, status, sample, or Pareto-report divergence in paired replay.
- Nominal 95% family-wise coverage is measured over independent campaigns and reported with confidence bounds under the declared IID model.
- Every out-of-model drift, dependence, missingness, or seed-collision regime either gets detected/downgraded or is explicitly labeled unsupported.
- Confirmatory performance claims use locked settings and held-out independent landscape/task clusters; a narrow synthetic benchmark cannot justify a general SOTA claim.

## 8. Exact-head validation commands

    cargo fmt --all -- --check
    cargo test -p symthaea-architecture-engineering event_tape_noisy -- --nocapture
    cargo test -p symthaea-architecture-engineering event_tape -- --nocapture
    cargo test -p symthaea-architecture-engineering
    cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings

The current environment does not include cargo, rustc, rustfmt, or clippy. Tests are authored but remain uncompiled and unexecuted. CI skipped is not a pass.
