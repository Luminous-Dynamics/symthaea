# EQC Trusted Observer Architecture v0.1

**Status:** Architecture proposal. Not implemented, independently audited, or a security certification.

**Tracking issue:** [#14189](https://github.com/Luminous-Dynamics/symthaea/issues/14189).

## 1. The trust gap

EQC v0.1 deliberately refuses to qualify a system from its own adapter observation. The adapter can report that it used the right identity, made no unauthorized mutation, obeyed network restrictions, and independently replayed its result. Those claims are useful contract inputs, but they are not independent evidence.

A digest establishes byte identity, not truth. A signature establishes that a key signed a statement, not that the signer observed the correct process, that its clock/counters were correct, or that the runner was uncompromised.

The benchmark therefore needs a second measurement plane, separate from the system under test (SUT).

## 2. Separation-of-duties model

`task owner -> observer -> SUT -> artifact -> independent verifier -> qualification report`

The roles must have distinct authority:

| Role | Permitted responsibility | Must not control |
|---|---|---|
| Task/policy owner | Freeze task, authority snapshot, inputs, allowed mutations, budgets and verifier policy | SUT process, observer signing key |
| SUT | Attempt the task through declared interfaces | Hidden tests, observer memory/key, verifier secrets, host network policy |
| Observer | Launch/isolate SUT, capture external facts, hash artifacts, measure resource use and sign an envelope | Change task expected result, rewrite hidden tests, grant authority |
| Independent verifier | Re-run artifact in a fresh environment; compare expected state and hidden tests | SUT workspace, adapter-supplied trust decisions |
| Qualification policy | Verify observer trust identity, verifier identity, evidence completeness and non-compensatory gates | Be selected or overridden by the SUT |

The observer is not another callback inside the agent process. It must run outside the SUT's process/container boundary and own the measurement channel. A wrapper that merely asks the agent to report its own counters does not qualify.

## 3. Observer envelope

The proposed signed envelope has two layers:

1. **Payload:** canonical, versioned JSON describing one execution and its observed evidence.
2. **Attestation:** signature and signer identity over the exact canonical payload bytes.

Minimum payload fields:

- envelope schema, run ID, fresh nonce, task ID and task digest;
- SUT source commit, binary/container image digest, model/agent/harness identity and tool versions;
- frozen input digests, authority/policy digest, expected mutation manifest;
- observer ID/version, workflow identity, runner class, operating system and architecture;
- pre-state and post-state snapshot digests, allowed mutation manifest digest, action/event log digest and final artifact/result digest;
- network policy, enforced allowlist, observed endpoints, bytes sent/received and enforcement outcome;
- elapsed wall time, CPU time, peak memory, tool calls, human interventions, measured energy and cost;
- verifier ID/version/digest, replay status and replay artifact digest;
- calibration/method metadata, timestamp source, monotonic duration and any missing/unknown measurements.

Unknown measurements remain `null`; never silently encode missing energy, cost, or resource counters as zero.

The signature covers all security-relevant fields and is calculated over a defined canonical encoding. The signature field itself is excluded from the signed bytes. Schema validation alone is not a signature verification.

## 4. Signature and identity policy

The evaluator must receive accepted observer trust roots separately from the SUT and receipt. An envelope cannot nominate its own trusted public key.

Two possible first implementations:

- **Pinned-key mode:** a verifier configuration contains one or more trusted Ed25519 public keys and key IDs. The observer signs the canonical envelope; the verifier checks the key, signature, algorithm, and all run bindings.
- **Workload-identity mode:** a short-lived OIDC identity is constrained to an exact repository, workflow, ref, environment and audience. A verifier validates the attestation chain against a separately pinned issuer policy.

Do not treat a key included only inside the receipt as trusted. A valid signature is necessary but not sufficient: the policy must separately authorize the signer and its measurement method.

## 5. Anti-replay and anti-forgery rules

A receipt is rejected or left unqualified when any of the following occurs:

- duplicate or reused run nonce;
- unknown, expired, revoked or out-of-policy observer identity;
- signature mismatch, unsupported algorithm or malformed canonical payload;
- task, policy, source, environment, input, action-log or result digest differs from the frozen run manifest;
- pre/post state evidence is missing or the observed mutation set exceeds the permitted set;
- host-side network counters disagree with the declared network policy or allowlist;
- resource measurement is absent, inconsistent or exceeds a hard budget;
- verifier/replay identity is the SUT or shares its mutable workspace;
- the same receipt or signature is attached to another run/task;
- observer and verifier are not independent under the declared threat model.

A system cannot recover a qualified label by setting `independent=true` in its own observation.

## 6. First pilot: Linux-only, fail-closed

Start with a deliberately narrow measurement envelope on a pinned Linux runner:

1. Run the observer on the host; run the SUT as an unprivileged user/container with separate PID, mount and network namespaces.
2. Mount task inputs read-only; provide one bounded writable output volume. Snapshot it before and after execution.
3. Use cgroup v2 for CPU, memory and process accounting; use monotonic host clock for elapsed duration.
4. Enforce deny-by-default egress through a host-owned network namespace/firewall. Capture actual bytes and endpoint decisions outside the SUT.
5. Store action logs and output artifacts in observer-owned storage; hash them before the SUT is destroyed.
6. Launch a fresh verifier environment with only the task, frozen inputs, bounded artifact and policy manifest. Do not mount the SUT's mutable workspace.
7. Sign the envelope with a short-lived CI workload identity or a dedicated test key whose public key is pinned in verifier policy.
8. Preserve the full raw log, observer version, source commit, runner image, envelope, signature and verifier report.

This pilot should initially support only `network_policy=none` and deterministic software-state fixtures. Unsupported operating systems, hardware counters or network modes must return unqualified rather than being inferred from self-report.

## 7. Threat model and explicit limitations

The first version trusts the host kernel, runner image, CI identity provider, observer binary/build, pinned verifier configuration and signing-key custody. A compromised host kernel, malicious hosted runner operator, compromised identity provider, or malicious observer binary can forge measurements. This design does not eliminate those risks.

Mitigations to evaluate in later versions include reproducible observer builds, external transparency logs, dual observers operated by independent parties, hardware-rooted measurements/TPM quotes, confidential-computing attestations, independent network taps and physical power meters. None should be claimed until the actual threat model, supported hardware, quote verification and failure modes are tested.

The observer must not collect secrets or private clinical content unnecessarily. Prefer hashes, synthetic datasets and minimum event metadata. The protocol is not permission to publish patient data, credentials, tokens, or private source records.

## 8. Mutation tests for the observer itself

The verifier's test suite must contain deliberately forged or weakened evidence:

- flip each signed field and prove signature failure;
- substitute an untrusted key and prove identity rejection;
- reuse a nonce and prove replay rejection;
- change the SUT commit, environment digest, task digest or policy digest after signing;
- replace the output artifact after signing;
- claim zero network bytes while host counters show traffic;
- claim zero state mutations when the independent before/after oracle detects a write;
- claim independent replay while the verifier reused the SUT's mutable workspace;
- omit CPU, memory, elapsed-time or energy measurements and verify unknown remains unknown;
- make the observer report a false positive/false negative in a known negative-control task.

Mutation sensitivity is a validity property of the observer and verifier too. The system should not receive a pass merely because the signature verifies: the measurement channel must also pass its own negative controls.

## 9. Promotion gate from EQC v0.1

EQC v0.1 remains a reference-contract and schema-validation seed. It must continue to report `reference_contract_only_not_system_under_test` and never claim independent system qualification.

Do not enable `qualified=true` for external systems until all of the following are true:

- observer envelope schema and canonical encoding are frozen;
- signature verification and out-of-band signer policy are implemented;
- nonce/replay and all observer mutation tests pass;
- before/after state is captured independently of the SUT;
- resource and network measurement are collected outside the SUT;
- clean-environment replay is independently executed;
- the observer/verifier threat model and calibration are externally reviewed;
- a red-team exercise attempts to forge each evidence channel.

This is the next architectural step, not a feature currently delivered by EQC v0.1.
