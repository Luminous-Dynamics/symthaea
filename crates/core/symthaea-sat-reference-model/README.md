# Symthaea SAT Reference Model

An independent, dependency-free Rust oracle for a narrow subset of the
Sovereignty Assurance Tournament (SAT): purpose-bound effect scopes covering
action, resource, sink, purpose, parameter digest and payload labels; delegation
attenuation, lineage revocation and validity windows.

## Run

From the Symthaea workspace:

```sh
cargo test -p symthaea-sat-reference-model
cargo clippy -p symthaea-sat-reference-model --all-targets -- -D warnings
cargo fmt --all -- --check
```

The crate is included by the root workspace's `crates/core/*` member glob.

## Important boundary

This is a **reference model for tests**, not production authorization code.
It does not authenticate root issuers, verify signatures, canonicalize or hash
effect parameters, establish payload-label provenance, perform declassification
or content inspection, persist revocation state, obtain trusted time, fence an
external sink, or provide process isolation. Root grants are trusted fixture
inputs. Production authority and evidence adapters must remain independently
implemented and must be checked against this oracle rather than delegating
their trust decisions to it.

The authority API has no model-intelligence, confidence, consciousness,
collective-score or reputation parameter. A grant binds action, resource, sink,
purpose and an opaque digest of canonical operation parameters; payload labels
must be contained by its scope. Labels remain caller-supplied evidence, not
verified facts.

This label-set check is not the Luminous information-flow or disclosure policy.
Compose with Mycelix OPSEC-000 (#2746), Symthaea CogSec (#214), INTX-010
(#4453), INTX-010A (#4543), INTX-017 (#4504), and INTX-013 / INTX-013A
(#4491 / #4594). Those owners cover source classification/derivation, exact
InteractionIntent-bound egress, multi-connector lineage and policy-decision
evidence. This crate does not propagate DataLineage, make declassification
decisions, or create an egress receipt. An egress AllowCandidate remains
evidence and cannot create disclosure capability, DispatchPermit or effect.

Closest research comparator: [DelegateGuard (2026)](https://arxiv.org/abs/2610.16005).
This reference model is intentionally a narrow Rust conformance oracle, not a
claim to originate purpose-bound capability enforcement. The paper's key
limitation also applies here: a capability check cannot decide whether
sensitive content is safe to disclose through an otherwise authorized,
externally observable sink. That requires separately qualified content
governance or declassification.

Unknown external mutation outcomes and retry/reconciliation semantics remain
owned by Symtropy MISSION-SUPERVISOR-001 (#6498) and the authority-linearized
native mutation boundary (#6669). This crate deliberately does not implement a
competing mutation state machine.

No SAT benchmark result, isolation qualification, deployment qualification,
or general alignment guarantee follows from compiling or passing this crate.
