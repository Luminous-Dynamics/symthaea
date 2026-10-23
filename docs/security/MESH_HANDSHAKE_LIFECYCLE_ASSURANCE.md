# Mesh Handshake Lifecycle: Bounded Assurance Model

**Status:** research artifact; not protocol approval  
**Scope:** abstract session-admission logic related to PR #8412  
**Model:** `docs/security/models/MESH_HANDSHAKE_LIFECYCLE.tla`

## Why this is separate from nonce uniqueness

Nonce uniqueness and session establishment are independent safety properties. A correct nonce allocator cannot prove peer authentication, transcript binding, forward secrecy, hybrid key-exchange integrity or key confirmation. Conversely, a valid handshake cannot prevent nonce reuse if record counters and restart semantics are defective.

This model isolates the API-level question: **under what conditions may candidate key material become an established application session?** It does not model X25519, ML-KEM internals, a cryptographic combiner, wire bytes or an implementation's side channels.

## Modeled properties

- **EstablishedSound:** every established session has candidate secret material, a usable backend, bound peer identity and transcript, initiator and responder confirmation, and a policy-allowed negotiated profile.
- **FailedStateErasesCandidate:** an aborted in-progress handshake has no retained candidate secret in the abstract state.
- **UnavailableBackendNeverEstablishes:** absence of the required backend never permits session establishment.
- **TypeOK:** state variables remain in their declared finite domains.

Five mutant/legacy modes are designed to violate the properties:

1. **One-sided confirmation:** allow session admission after initiator confirmation without responder confirmation.
2. **Stub backend:** let the state machine create candidate material and establish a session despite the backend being unavailable.
3. **Downgrade:** select a classical-only profile when policy requires hybrid, despite no authenticated downgrade authorization.
4. **Unbound peer identity:** permit session establishment while the selected key is not bound to the claimed peer identity.
5. **Unbound transcript:** permit session establishment when the negotiated transcript is not bound to the candidate session.

The strict configuration models the expected policy gate. It is an oracle for desired behavior, not evidence that Rust uses the same gate or implements the cryptography correctly.

## Running the model

The workflow at `.github/workflows/mesh-handshake-model.yml` now downloads an exact upstream JAR and verifies its SHA-256 before parsing or checking the model. It requires strict configurations to preserve their invariants and treats a legacy run as an expected counterexample only when TLC reports the specified invariant violation—not merely when the command fails.

The current workflow pins upstream `v1.8.0` with SHA-256 `dae639a1f51a9e13ce60c369b15f32124cf1a78017aca366b6de0ae2b5169439`. **This release is marked prerelease at the time of writing.** Its run is suitable for experimental model validation; before relying on it as a production release gate, review the upstream release status and repeat the checks with an approved stable version.

The upstream TLA+ tools repository documents Java 11+ command-line use. For local execution, use a pinned tool artifact and record its digest:

```sh
java -cp /path/to/tla2tools.jar tla2sany.SANY \
  docs/security/models/MESH_HANDSHAKE_LIFECYCLE.tla

for cfg in docs/security/models/mesh_handshake_*.cfg; do
  java -cp /path/to/tla2tools.jar tlc2.TLC \
    -config "$cfg" docs/security/models/MESH_HANDSHAKE_LIFECYCLE.tla
done
```

The five `*_legacy.cfg` configurations are expected to fail on `EstablishedSound` or `UnavailableBackendNeverEstablishes`. The strict configurations are expected to preserve the declared invariants in the bounded state graph. These expected outcomes have **not** been run here; run and record the real tool output before claiming model-checking evidence.

## Implementation mapping required

After the model is parsed and model-checked, map each abstract transition to the real API and assert that the behavior matches:

| Abstract transition | Required Rust-level conformance evidence |
|---|---|
| `SelectProfile` | negotiated profile observed from the actual secure-channel provider, not just a requested setting |
| `BindIdentityAndTranscript` | full canonical peer identity plus protocol/version/suite and transcript hash bound by the chosen protocol |
| `ConfirmInitiator` / `ConfirmResponder` | both directions execute the selected protocol's required key-confirmation behavior |
| `candidateSecretPresent` | candidate material cannot be read through application-facing APIs |
| `Establish` | active session only after all requirements for the configured profile pass |
| `Abort` | timeout/error erases candidate state and does not register local trust or allow a downgrade |

This mapping must include all relevant feature combinations and failure cases: missing provider, stub provider, malformed KEM key/ciphertext, low-order X25519 input, wrong peer identity, transcript mutation, downgrade, timeout between confirmations, repeated peer IDs, cancellation, and concurrent handshakes. The state model does not represent secret zeroization guarantees, constant-time execution, side-channel behavior, or correctness of the underlying primitives.

## External design baseline

RFC 9954 defines a hybrid key-exchange framework **for TLS 1.3**. RFC 10024 specifies named PQ/T hybrid groups for TLS 1.3, including X25519MLKEM768 combined with ephemeral classical ECDH. They are appropriate baselines when using a conforming TLS 1.3 provider; they do not define a drop-in custom raw-mesh handshake.

For any hybrid profile, the implementation must demonstrate which group was actually negotiated and bind profile selection to the authenticated handshake. A source flag or build feature is not negotiation evidence. A compile-only/all-zero ML-KEM stub is not a cryptographic backend.

References:
- RFC 9954: https://www.rfc-editor.org/rfc/rfc9954.html
- RFC 10024: https://www.rfc-editor.org/rfc/rfc10024.html
- NIST FIPS 203 (check current errata/revisions): https://csrc.nist.gov/pubs/fips/203/final

## Benchmark targets

These are proposed thresholds, not achieved results.

| Metric | Target |
|---|---:|
| Pre-registered one-sided-confirmation, stub-backend and downgrade mutants rejected by strict model and implementation | 100% |
| Established sessions with missing/unusable required backend | 0 |
| Established sessions before required mutual confirmation | 0 |
| Established sessions with unbound identity or transcript | 0 |
| Unauthorized downgrade acceptance | 0 |
| Candidate secrets accessible to application callers before establishment | 0 |
| Valid handshakes dropped on frozen reference corpus | 0 |
| Resource/state overflow from bounded malformed-handshake input | 0 |
| Interoperability vectors across independent implementations | 100% agreement |

Benchmark the same frozen trace corpus across a hand-written state-machine test suite, ordinary property tests, formal model checking, one tool-using agent, and independent generator/evaluator agents. Record mutant kill rate, valid-control retention, false positive count, counterexample minimization, execution cost, human triage time, and exact model/source/tool digests. A bounded model pass cannot substitute for implementation conformance or independent cryptographic review.
