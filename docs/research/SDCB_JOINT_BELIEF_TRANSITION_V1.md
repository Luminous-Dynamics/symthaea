# Evidence-gated joint scientific and instrument belief transitions v1

**Status:** bounded research reference extending the EGBT transition contract; not production attestation.  
**Implementation:** [scripts/research/sdcb_joint_belief_transition.rs](../../scripts/research/sdcb_joint_belief_transition.rs)  
**Schema:** [SDCB_JOINT_BELIEF_TRANSITION_V1.schema.json](SDCB_JOINT_BELIEF_TRANSITION_V1.schema.json)  
**Example:** [SDCB_JOINT_BELIEF_TRANSITION_V1.example.json](SDCB_JOINT_BELIEF_TRANSITION_V1.example.json)  
**Expected result:** [SDCB_JOINT_BELIEF_TRANSITION_V1.expected.json](SDCB_JOINT_BELIEF_TRANSITION_V1.expected.json)

## 1. Why extend EGBT rather than duplicate it?

The existing EGBT v1 reference applies a verified observation to a posterior over scientific hypotheses. The instrument-aware and drift-aware decision references now distinguish scientific uncertainty from device-specific state uncertainty. A scientific evidence transition should therefore update their **joint** belief rather than throw away the measurement-channel state.

This reference preserves the EGBT fail-closed pattern while adding:
- A frozen ordered device registry and registry digest.
- A per-device evidence version in addition to the global belief version.
- Exact device identity on both proposal and receipt.
- Calibration epoch and calibration-context digest.
- A calibration validity interval and observation capture interval.
- A joint posterior over scientific hypotheses and binary device states.
- Before/after scientific and device marginals in the transition record.

This is a separate reference file; it does not modify or replace the existing \`sdcb_belief_transition.rs\` implementation.

## 2. Joint update contract

Let \(h\) be the scientific hypothesis and \(c_1,\ldots,c_D\) the binary state of each registered device. The joint ledger stores:

\[
p(h,c_1,\ldots,c_D).
\]

An experiment on device \(d\) supplies \(L(h,c_d,y)=P(y\mid h,c_d,e)\). After the independently verified outcome \(y\), the reference computes:

\[
p'(h,c_1,\ldots,c_D)
=
\frac{p(h,c_1,\ldots,c_D)L(h,c_d,y)}
{\sum_{\tilde h,\tilde c_1,\ldots,\tilde c_D}
p(\tilde h,\tilde c_1,\ldots,\tilde c_D)
L(\tilde h,\tilde c_d,y)}.
\]

For a calibration-role experiment, likelihoods must be independent of \(h\) when the named device state \(c_d\) is held fixed. Calibration can still inform the scientific hypothesis indirectly when the *prior* has a genuine correlation between scientific and instrument states; it cannot leak hypothesis information through the calibration likelihood itself.

The transition record stores:
- the complete joint prior and posterior;
- scientific-hypothesis marginal before and after;
- named-device state marginal before and after;
- global ledger version before and after;
- per-device evidence version before and after;
- predictive probability, receipt/proposal digests, calibration epoch, capture interval, and budget transition.

This supports exact independent replay of the finite Bayesian update. It does not prove that the likelihood model is scientifically correct.

## 3. Receipt binding and atomicity

An accepted transition requires exact agreement on:
- run, task, system, hypothesis-set, and device-registry identity;
- ordered hypothesis and device IDs;
- experiment ID and proposal digest;
- device ID and its current device-evidence version;
- global prior/ledger version;
- calibration epoch and calibration-context digest;
- observation digest and verifier identity;
- cost and remaining budget;
- observed outcome;
- capture interval inside the declared calibration-validity interval.

An accepted observation increments both the global ledger version and the named device's evidence version. This prevents a previously prepared proposal for the same device from being accepted after the device evidence state has changed.

All checks and posterior calculations occur before mutation. A rejected receipt leaves the joint posterior, global version, all per-device versions, budget, used-experiment set, and accepted-transition count unchanged.

## 4. Frozen fixture

The example has two equally likely hypotheses and two named devices. Device A begins with an unknown binary offset; device B has an independent 0.10 prior probability of a flipped offset. A calibration measurement on device A reports outcome 0 with 0.95 accuracy.

The prior is factorized across the scientific hypothesis and the two device states. Therefore a valid calibration observation changes device A's state marginal from \`[0.50, 0.50]\` to \`[0.95, 0.05]\` while leaving the scientific marginal exactly \`[0.50, 0.50]\`.

The accepted calibration increments the global ledger version from 0 to 1, device A's evidence version from 0 to 1, and reduces the example budget from 5 to 4. This is a known-answer analytic fixture, not evidence from a physical sensor.

A subsequent science observation on device A can then update the scientific marginal using the now-informed joint state. The expected file records the calibration-only transition; the Rust tests also cover the subsequent science update.

## 5. Three trust questions must remain distinct

1. **Receipt authenticity and integrity:** did an authorized verifier attest to these exact bytes, device, proposal, and capture interval?
2. **Metrological validity:** was the device traceable, calibrated for this measurand, operated within its valid range, and appropriately characterized for drift and uncertainty?
3. **Scientific validity:** does the likelihood/model correctly describe the phenomenon, and is the requested mechanism identifiable from the available interventions and observations?

A valid signature or provenance receipt can support the first question. It does not answer the second or third. The \`independently_verified\` boolean is only a test seam and must never be set by an untrusted agent.

Production integration still requires authenticated verifier identities, signed source/custody records, digest-preimage verification, revocation and key lifecycle, append-only accepted/rejected event storage, calibrated sensor references, and independently governed metrology. The reference currently checks digest *shape*, not cryptography.

## 6. Integration path

The intended boundary remains:

**Symthaea proposes and predicts → authorized executor performs → independent verifier authenticates evidence → joint belief service applies the update → independent replay qualifies the transition.**

The existing EGBT v1 hypothesis-only ledger remains unchanged. A future integration should:
1. Add the device registry and joint prior to the episode contract.
2. Extend proposal/receipt wire schemas with device identity, calibration context, calibration epoch, capture interval, and per-device evidence version.
3. Replace the untrusted boolean test seam with a typed, authenticated verifier attestation.
4. Bind the likelihood model and any instrument-state transition model to a source-pinned digest.
5. Persist accepted and rejected transitions in an append-only ledger.
6. Replay both scientific and device marginals from the same accepted evidence.

## 7. Limits

The Rust reference supports at most four hypotheses, three binary-state devices, and at most four-outcome observations. It performs one Bayesian joint-state update; it does not propagate Markov drift between observations itself. The caller must supply likelihoods valid for the captured device state and declared calibration epoch. The preceding drift-aware decision reference models finite state transitions but is not yet connected to this ledger.

Not implemented: cryptographic signature verification, source custody, trusted time, calibration-certificate revocation, continuous sensor state, Markov transition propagation in the ledger, common-cause model learning, physical metrology, HDC/CfC integration, or production persistence.

**Evidence boundary:** ten Rust tests are authored but not compiled or executed in this environment. The JSON example and expected fixture are independently parsed and checked. No physical validation or external benchmark result is claimed.
