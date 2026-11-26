# Evidence-gated joint science, common-cause, and device belief transitions v1

**Status:** bounded research reference extending EGBT; not production attestation or metrology.  
**Implementation:** [scripts/research/sdcb_common_cause_evidence.rs](../../scripts/research/sdcb_common_cause_evidence.rs)  
**Schema:** [SDCB_COMMON_CAUSE_EVIDENCE_V1.schema.json](SDCB_COMMON_CAUSE_EVIDENCE_V1.schema.json)  
**Example:** [SDCB_COMMON_CAUSE_EVIDENCE_V1.example.json](SDCB_COMMON_CAUSE_EVIDENCE_V1.example.json)  
**Expected transitions:** [SDCB_COMMON_CAUSE_EVIDENCE_V1.expected.json](SDCB_COMMON_CAUSE_EVIDENCE_V1.expected.json)

## 1. Purpose

The existing EGBT reference updates a scientific-hypothesis posterior. The joint instrument extension adds device-local states. The common-cause decision model adds an environmental/reference/batch state shared by multiple devices.

The evidence ledger must preserve all three dimensions:

\[
p(h,g,c_1,\ldots,c_D\mid E),
\]

where:
- \(h\) is the scientific hypothesis;
- \(g\) is a registered shared common-cause state;
- \(c_d\) is the local state of named device \(d\);
- \(E\) is the sequence of accepted, verified observations.

This reference combines those variables in one posterior and retains fail-closed update semantics. It is separate from, and does not alter, EGBT v1 or the prior device-only joint ledger.

## 2. Observation model and update

A proposal for device \(d\) declares an outcome likelihood:

\[
L(h,g,c_d,y)=P(y\mid h,g,c_d,e).
\]

The state index is hypothesis-major; the instrument-state code uses bit 0 for \(g\), and bit \(d+1\) for the local state of device \(d\).

After a valid receipt for outcome \(y\), the ledger computes:

\[
p'(h,g,c_1,\ldots,c_D)
=
\frac{p(h,g,c_1,\ldots,c_D)L(h,g,c_d,y)}
{\sum_{\tilde h,\tilde g,\tilde c}
p(\tilde h,\tilde g,\tilde c)L(\tilde h,\tilde g,\tilde c_d,y)}.
\]

A SharedReferenceCalibration likelihood must be independent of \(h\) conditional on the common state \(g\) and the named device's local state \(c_d\). It may update \(g\) and \(c_d\); with a correlated prior, that can legitimately change the scientific marginal indirectly. The calibration likelihood itself cannot leak scientific-hypothesis information.

## 3. Evidence binding and versions

A proposal and its receipt bind the exact:
- run, task, system, ordered scientific-hypothesis set, device registry, and common-cause registry;
- common-cause ID and common-cause evidence version;
- device ID and device-local evidence version;
- experiment ID, proposal digest, and observation digest;
- calibration epoch, calibration-context digest, and capture interval;
- global ledger version, outcome, cost, verifier identity, and remaining budget.

An accepted observation increments three version layers atomically:
1. the global scientific/joint ledger version;
2. the shared common-cause evidence version;
3. the selected device's local evidence version.

This means a proposal cannot silently reuse an older common-cause state or an older device-specific evidence state. The global version also protects against stale proposals across all devices in the same episode.

All binding checks, likelihood validation, posterior calculation, and version-overflow checks happen before mutation. Wrong common-cause IDs, stale shared-cause or device versions, expired calibration windows, hypothesis-leaking calibration data, unverified, duplicate, and over-budget receipts reject without changing the joint posterior, any marginal, versions, budget, used-experiment set, or accepted count.

## 4. Frozen known-answer fixture

The fixture has two equally likely scientific hypotheses, one shared binary lab-environment state, and two device-local binary states. All device-local states begin known at zero. The shared state is initially uniform.

A 95%-accurate shared-reference calibration on shared-reference returns outcome 0. The exact update is:

- scientific marginal remains [0.50, 0.50];
- common-cause marginal changes from [0.50, 0.50] to [0.95, 0.05];
- both device-local marginals remain [1.00, 0.00];
- global version changes 0 → 1;
- common-cause version changes 0 → 1;
- shared-reference device version changes 0 → 1;
- budget changes 5 → 4.

A subsequent 95%-accurate scientific observation on science-a, also outcome 0, yields:
- scientific marginal [0.905, 0.095];
- common-cause marginal [0.95, 0.05];
- global version 1 → 2;
- common-cause version 1 → 2;
- science-a device version 0 → 1;
- budget 4 → 3.

These are exact finite Bayesian calculations, not physical sensor results. They show why a reference observation should update the shared measurement state before a science observation is interpreted through it.

## 5. Three trust layers remain separate

1. **Evidence authenticity:** did a separately governed verifier authenticate this exact source, device, proposal, and capture record?
2. **Measurement validity:** does the calibration chain, check-standard history, uncertainty budget, environment, common-cause model, and device state support the declared likelihood?
3. **Scientific validity:** is the observation model adequate, and is the mechanism identifiable under the declared interventions?

A valid receipt does not prove the instrument is accurate. A valid calibration chain does not prove a scientific model is correct. The independently_verified boolean and lowercase digest checks in this Rust reference are test seams only; they do not authenticate signatures or hash preimages.

## 6. Production integration path

The intended system boundary is:

**Symthaea proposes → authorized executor measures → independent verifier authenticates the receipt → joint belief service updates \(h,g,c_d\) → independent replay qualifies the transition.**

Production integration requires:
- authenticated, revocable verifier attestations;
- signed device, common-cause, calibration-chain, and source-custody records;
- cryptographic digest/preimage verification and trusted time;
- append-only accepted and rejected event storage;
- a governed likelihood/transition model with source-pinned parameters;
- metrological traceability and uncertainty evaluation;
- replay of every joint and marginal transition from the accepted evidence.

The ledger should not permit the agent that proposed an experiment to certify its own measurement.

## 7. Limits

The reference supports at most four hypotheses, three devices, one binary common-cause state, one binary local state per device, and at most four outcomes per experiment. It performs one Bayesian update per accepted observation; it does not propagate the Markov common-cause transition itself. The likelihood must already be conditioned on the declared capture epoch and state model.

Not implemented: cryptographic verification, custody, trusted time, revocation, continuous common/local states, multiple common factors, cross-device transition learning, calibration-reset actions, physical metrology, HDC/CfC integration, or production persistence.

**Evidence boundary:** ten Rust tests are authored but not compiled or executed in this environment. The JSON example and expected posterior updates are independently parsed and recomputed. No physical validation or external benchmark result is claimed.
