# EQC v0.2 Trusted Observer and Replay Boundary

**Status:** Architecture proposal. Not implemented, not a certification, and not yet a claim of trusted execution.

**Tracking issue:** [#14076](https://github.com/Luminous-Dynamics/symthaea/issues/14076)  
**v0.1 implementation:** [PR #13997](https://github.com/Luminous-Dynamics/symthaea/pull/13997)

## 1. Why v0.2 exists

EQC v0.1 deliberately separates adapter-reported contract checks from independent qualification. Its Rust evaluator verifies frozen task inputs, fixture digests, expected decisions, fail-closed controls, and consistency of an adapter's observation. But an adapter can lie about resource use, network access, state changes, and replay. A JSON field saying \`independent=true\` is only a claim.

Therefore v0.1 never sets \`qualified=true\` from adapter-submitted observations. v0.2 must add a separately provisioned observer and a replay environment that does not inherit the SUT's mutable workspace or credentials.

The governing rule remains:

> The system under test may submit an artifact and claims about its run. It may not choose the observer, measurement source, trust root, replay environment, verifier, or qualification policy.

## 2. Trust-level model

| Level | Evidence source | Permitted status | What it does not prove |
|---|---|---|---|
| L0 — self-reported | SUT/adapter JSON | \`contract_checks_pass\` or \`unqualified\` only | Truth of execution, resource use, network, state or replay |
| L1 — host-observed | Separately provisioned supervisor and isolated replay worker, with keys/policy provisioned outside the SUT | May qualify only after observer identity, measurements, state evidence and independent replay all verify | Security against a compromised host kernel, supervisor or cloud control plane |
| L2 — platform-attested | L1 plus verified platform measurement (for a supported TPM/TEE/confidential-computing deployment) and a verified deployment policy | May qualify under a versioned platform policy | Universal trust, correctness of the application, or properties not covered by the attestation |

L1 is a useful, implementable first target. It is not adversary-proof. A normal hosted CI runner should not be described as a hardware root of trust. L2 is an optional deployment-specific extension, not a requirement for every benchmark episode.

The v0.2 observer receipt schema is evidence input, not a self-contained authorization to qualify. Signature verification, key allow-listing, nonce state, observer policy, and replay binding happen in an evaluator that has its trust policy from outside the SUT.

## 3. Architecture and data flow

\`\`\`text
Frozen task corpus + policy snapshot + fresh nonce
                    |
                    v
          Trusted evaluation controller
             /                 \\
            v                   v
  SUT sandbox / namespace   Observer policy + key
            |                   |
            v                   |
  submitted artifact + logs     |
            |                   |
            +-------> Host observer
                         |
                cgroup / namespace /
                filesystem / state measurements
                         |
                         v
                 Signed observer receipt
                         |
                         v
             Independent clean replay worker
                 receives artifact only
                         |
                         v
              replay receipt / verifier result
                         |
                         v
            Trust-policy + signature verifier
                         |
                         v
          EQC dimensions and qualification status
\`\`\`

### 3.1 Controller responsibilities

Before SUT launch, the controller:

1. resolves the immutable task version, task digest, input digests, actor/subject/authority snapshot and policy digest;
2. creates a cryptographically random nonce with a bounded validity interval and records it in an evaluator-owned consume-once store;
3. provisions the observer key ID and verifier policy from an external configuration channel;
4. creates a clean execution environment and records its immutable image/runtime identity;
5. ensures the SUT receives only the task, allowed inputs and narrowly scoped capabilities.

The SUT must not receive the observer signing key, evaluator policy, replay credentials, CI write token, container-runtime socket, host filesystem, or mutable replay workspace.

### 3.2 Host-observer responsibilities (L1)

A separate supervisor—not code running inside the SUT process—records:

- start/end wall time and process exit status;
- cgroup CPU usage, peak memory and process count;
- tool-call count and tool identities from the host-side tool broker;
- network namespace policy, bytes in/out, connection endpoints and policy-enforcement result;
- filesystem/state before/after digests, with explicit \`unchanged\`, \`mutated\` or \`unknown\` semantics;
- submitted artifact bytes and SHA-256;
- observer binary digest, observer policy digest, key ID and task/nonce binding.

Unknown or unavailable measurements remain unknown. They must never be coerced to zero, empty, false or success. Filesystem change events alone do not prove the full state of an external service; external state transitions need a separately instrumented service/database audit source.

For Linux, cgroup v2 and network namespaces are plausible L1 primitives, but the supported kernel/runtime configuration must be pinned and tested. An implementation must specify which counters are kernel-enforced and which are broker-derived. The observer should fail closed if it cannot read the required counters.

### 3.3 Independent replay worker

The replay worker runs after the SUT exits and receives only:

- immutable task bundle and permitted public/licensed input artifacts;
- submitted patch/action/simulation artifact;
- the versioned replay environment and independent verifier.

It must not receive the SUT's working directory, caches, generated files, mutable tool state, credentials, process memory, or observer private key. It applies the artifact to a pristine image and records a separate replay-run identifier, environment digest, verifier digest, result digest, exit code, and outcome.

For non-deterministic numerical/physical tasks, replay agreement uses a preregistered metric, tolerance and uncertainty interval—not forced bitwise equality. The task contract records that policy before the run.

## 4. Signed receipt contract

The machine-readable proposal is [\`eqc-observer-receipt-v0.2.schema.json\`](eqc-observer-receipt-v0.2.schema.json).

The supervisor signs a canonical payload that binds:

- task ID/digest, nonce, issuance/expiry times and one-time observation ID;
- SUT identity (agent/model/harness/tool versions, source head, artifact image);
- authority/policy and input digests;
- observer identity, binary digest, policy digest and key ID;
- resource, network and state measurements plus their measurement sources;
- exact submitted artifact digest;
- replay worker identity, clean environment digest, verifier digest, replay run ID and result;
- optional platform attestation evidence at L2.

The signature covers a canonical JSON/CBOR representation with a declared canonicalization algorithm. Do not sign arbitrary JSON text whose key order or number formatting may vary. The verifier recomputes all digests; it never trusts a \`payload_sha256\` supplied without recomputation.

The receipt schema intentionally has **no \`qualified\` boolean**. A SUT or observer receipt cannot declare itself qualified. Qualification is computed by a separate evaluator from a locally pinned trust policy.

A valid signature only proves that a holder of a key signed particular bytes. It does not by itself prove that the host measurement is correct, that the host was uncompromised, or that the replay was independent. Those are trust assumptions that must be named and reviewed.

## 5. Threat model and required negative controls

| Threat | Required control | Evidence |
|---|---|---|
| SUT reports zero network/CPU/mutations while using resources | Measurements from host cgroup, network namespace and state audit sources | Observer receipt and independent negative-control result |
| SUT supplies \`independent=true\` and matching hashes | Ignore adapter independence claims; replay on a different worker | Separate replay job identity and clean environment digest |
| SUT replaces observer key/verifier | Key allow-list and policy digest injected outside SUT | Evaluator's pinned observer policy |
| SUT reads supervisor file, CI token or runtime socket | Dedicated UID, minimal mounts, no write token, no runtime socket | Negative-control probe and host access logs |
| SUT forks child process after parent exits | cgroup/process-tree lifetime tied to episode | Peak process count and cgroup final state |
| Network egress hidden behind an allowed endpoint | Default-deny egress, broker-mediated endpoints, DNS/IP logging | Namespace policy and observed connection set |
| SUT mutates cache or workspace then restores it | Immutable input mounts, filesystem diff/overlay audit and clean replay | Before/after content digests and event record |
| Receipt or nonce replay | Evaluator-owned consume-once nonce store and expiry check | Nonce state transition and observation ID |
| Observer policy or key expires/rotates | Versioned trust bundle, validity interval and revocation check | Evaluator-resolved key/policy identity |
| Host cannot observe state | \`unknown\` state; no qualification | Explicit missing measurement |
| Numerical result differs by platform | Predeclared tolerance/uncertainty policy | Replay metric, tolerance and observed residual |

The threat model assumes an uncompromised host kernel and supervisor at L1. A privileged adversary able to control the host kernel can falsify host-local observations. L2 can add measured-platform evidence, but only for a specific supported implementation.

## 6. Qualification state machine

The evaluator should return distinct states:

- \`contract_checked_untrusted\`: v0.1 or adapter-only evidence was internally consistent;
- \`observer_signature_verified\`: receipt signature, key, nonce and policy binding are valid;
- \`measurements_verified\`: required host measurements are present, within the task budget and from accepted measurement sources;
- \`state_verified\`: state-before/after evidence satisfies the task contract, with no unresolved unknown;
- \`replay_verified\`: the independent replay job is separate, pristine and meets the declared comparison policy;
- \`qualified_l1\`: every required EQC dimension passes under a pinned L1 policy;
- \`qualified_l2\`: the L1 conditions pass and the platform attestation is verified by a supported L2 policy;
- \`unqualified\`: any required check fails, is absent, stale, unverifiable, or contradictory.

A status transition is not monotonic toward success by default: revocation, nonce expiry, key rotation, task invalidation or discovery of contradictory evidence can downgrade an episode to unqualified.

## 7. Implementation phases

### Phase A — Receipt verifier and threat-model tests

- Validate the v0.2 schema with an offline Draft 2020-12 validator.
- Implement canonicalization and signature verification over a fixed test vector.
- Pin accepted observer keys/policies outside the observation payload.
- Add nonce consume-once and expiry checks.
- Verify that adapter fields cannot overwrite observer measurements.
- Test malformed signature, wrong key, stale nonce, replayed receipt, wrong task digest, unknown state and wrong replay environment.

### Phase B — Host-only observer prototype

- Prototype in a dedicated Linux worker with a dedicated UID and cgroup v2.
- Deny network by default; record namespace-level counters and brokered endpoints.
- Use immutable task/input mounts and capture artifact/state digests.
- Keep the observer signing key outside the SUT namespace.
- Make the replay job a separate worker/image with no inherited credentials.
- Publish the L1 trusted-computing-base statement and exact kernel/runtime configuration.

### Phase C — Independent review and measured comparison

- Have an external reviewer attempt to forge usage counters, bypass egress restrictions, mutate state, steal the observer key and contaminate replay.
- Publish per-threat-family false acceptance/rejection and unresolved rates.
- Only then label a benchmark track \`qualified_l1\`.
- Add L2 only for explicitly enumerated hardware/platform combinations with verified attestation chains.

## 8. Non-goals and claim boundary

This proposal does not implement a sandbox, a cryptographic attestation system, a remote attestation verifier, or a production clinical/financial authorization system. It does not make the current EQC v0.1 seed qualified. It must not be used to claim a shared hosted CI runner is tamper-proof.

The v0.1 behavior—self-reported observation cannot set \`qualified=true\`—remains in force until the L1 trust chain exists and passes adversarial review.

## References

- [Linux cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html) — resource accounting/control primitives.
- [Linux network namespaces](https://man7.org/linux/man-pages/man7/network_namespaces.7.html) — network isolation boundary.
- [in-toto Attestation Framework](https://github.com/in-toto/attestation) — signed, typed statement/envelope model; it does not by itself establish truthful measurement.
- [SLSA provenance](https://slsa.dev/spec/v1.1/provenance) — build provenance concepts and trusted builder identity.
- [TPM 2.0 Library Specification](https://trustedcomputinggroup.org/resource/tpm-library-specification/) — platform trust primitives for a future, narrowly scoped L2 track.
