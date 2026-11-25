# Symthaea SAT Reference Model

An independent, dependency-free Rust oracle for a narrow subset of the
Sovereignty Assurance Tournament (SAT): exact-effect capability scope,
delegation attenuation, lineage revocation and validity windows.

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
effect parameters, persist revocation state, obtain trusted time, fence an
external sink, or provide process isolation. Root grants are trusted fixture
inputs. Production authority and evidence adapters must remain independently
implemented and must be checked against this oracle rather than delegating
their trust decisions to it.

The authority API has no model-intelligence, confidence, consciousness,
collective-score or reputation parameter. An exact grant binds action,
resource and an opaque digest of canonical operation parameters.

Unknown external mutation outcomes and retry/reconciliation semantics remain
owned by Symtropy MISSION-SUPERVISOR-001 (#6498) and the authority-linearized
native mutation boundary (#6669). This crate deliberately does not implement a
competing mutation state machine.

No SAT benchmark result, isolation qualification, deployment qualification,
or general alignment guarantee follows from compiling or passing this crate.
