# Bounded candidate/evaluator wire protocol v0.1

Tracking: [#9851](https://github.com/Luminous-Dynamics/symthaea/issues/9851)  
Implementation branch: `research/temporal-candidate-protocol-v0.2`  
Parent stack: [HARTH stream parser PR #9847](https://github.com/Luminous-Dynamics/symthaea/pull/9847) and [active-sensing simulator PR #9845](https://github.com/Luminous-Dynamics/symthaea/pull/9845)

## Objective

The active-sensing simulator's Rust policy trait is useful for fast deterministic algorithm tests, but it is not a secure candidate/evaluator boundary. Candidate code and hidden oracle data in one process remain exposed to shared address space, accidental references, unsafe code in dependencies, process-global state and runtime interference.

This protocol creates a small, explicit wire contract that can later be attached to a separately launched candidate process. It is only one layer of the solution; OS isolation and evaluator ownership remain mandatory.

## Frame format

Every message is a fixed 56-byte little-endian header followed by a bounded payload:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | Magic: `SYMTEDGE` |
| 8 | 2 | Protocol version (1) |
| 10 | 2 | Message type: 1 observation, 2 action |
| 12 | 8 | Monotonic request/response sequence |
| 20 | 4 | Payload length |
| 24 | 32 | Domain-separated SHA-256 over type, sequence and payload |

Maximum payload length is 128 bytes. The observation payload encodes only:
- current public step;
- optional last sensor measurement (sample step, sensor tag, positive/negative bit);
- optional age of that measurement;
- sample counters.

The action payload encodes only:
- alarm bit;
- sensor choice (none, low-power, high-fidelity);
- next wake interval;
- candidate-declared compute cycles, transmitted bytes and memory-write bytes.

No scenario seed, event schedule, hidden label, shift parameters, oracle digest or evaluator metric is part of the candidate observation codec.

## Codec behavior

`research/temporal-bench/src/candidate_protocol.rs` provides typed `Read`/`Write` functions:
- evaluator: `write_observation`, `read_action`;
- candidate: `read_observation`, `write_action`.

Every frame validates magic, version, type, exact expected sequence, bounded length and payload digest. Typed decoding validates optional tags, booleans, sensor values, timestamp/age consistency, sample-counter consistency, action wake limits and exact payload consumption. No JSON parser, serde dependency, unbounded frame allocation or dynamic schema negotiation is introduced.

The digest is a corruption/replay commitment, not an authenticator. It is not a signature, MAC, trusted timestamp, attestation or proof that a candidate did not collude with an external process.

## Required process-isolation architecture (not yet implemented)

The confirmatory runner must launch a separate candidate process through a pinned, fail-closed sandbox supervisor. A direct unsandboxed execution fallback is forbidden.

Initial Linux profile requirements:
- candidate and declared runtime closure are read-only inside a private mount namespace;
- the candidate receives only framed public observations over stdin and returns typed actions over stdout;
- the evaluator's oracle, seed schedule, raw labels, result store and signing keys are absent from candidate mounts, environment, inherited file descriptors and IPC namespaces;
- network namespace disabled; capabilities dropped; syscall surface reduced;
- bounded temporary filesystem and no writable host bind mounts;
- cgroup-enforced memory, CPU, process-count and wall-clock limits;
- bounded stdout/stderr and strict frame/sequence/deadline handling;
- candidate termination on malformed output, overrun, timeout, memory exhaustion or sandbox failure;
- separate evaluator process validates candidate output and computes metrics from oracle state;
- system measurement process binds device telemetry and raw trace digests separately.

For NixOS/reproducibility, pin the sandbox supervisor, candidate artifact, runtime closure, system profile and launch policy by digest. A missing/unsupported namespace or resource-control primitive must fail closed. Do not silently retry with fewer restrictions. The exact bwrap/systemd/cgroup profile and kernel feature probe require a dedicated Linux runner implementation and review before being treated as qualified.

## Receipt chain

At minimum, later runner implementation should create separately typed artifacts:
1. **Candidate launch receipt:** candidate binary/artifact digest, runtime closure digest, protocol version, launch-policy/supervisor digest, resource-limit profile and host/kernel identity.
2. **Candidate output receipt:** request stream digest, action stream digest, bounded stderr digest, exit code and termination reason.
3. **Evaluator receipt:** candidate output binding, private oracle commitment, evaluator code digest, exact metric denominators and per-scenario strata.
4. **System receipt:** physical target, firmware/compiler/runtime, raw latency/energy traces, calibration/uncertainty receipt and device session.

Candidate output receipts must not contain the oracle. Evaluator receipts must not be writable by the candidate. The evaluator must replay the candidate trace against a separately committed task and oracle.

## Tests authored

Nine synthetic tests cover observation/action round trips, sequence mismatch, payload tampering, oversized length rejection before allocation, truncated header/payload, malformed action tags, inconsistent observation counters/ages, wrong message types and invalid sensor tags.

**Execution status:** tests are authored but have not been compiled or run. No claim of OS sandboxing, process isolation, attestation, model quality, physical energy or SOTA is made.
