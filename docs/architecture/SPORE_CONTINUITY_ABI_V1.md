# Spore Continuity ABI v1

Status: draft wire contract implemented by `symthaea-spore-continuity`

## Purpose

`spore-continuity-v1` carries a tiny amount of presentation-safe semantic state from one independently isolated lifecycle renderer to the next.

It is **not**:

- a boot success receipt;
- authentication state;
- a session credential;
- a lock authority;
- a framebuffer-sharing protocol;
- a general desktop IPC protocol;
- persistent machine identity.

If this ABI is unavailable, invalid, stale, or unsupported, the receiving surface starts a safe built-in visual scene and the operating-system lifecycle continues normally.

## Rust owner

```text
crates/core/symthaea-spore-continuity
```

The crate forbids unsafe Rust and currently depends only on workspace `serde` and `serde_json`.

## v1 fields

The canonical state contains:

```text
version
continuity_lineage[16]
handoff_sequence
scene_digest[32]
visual_seed[32]
visual_plan_digest?[32]
phase_micros
world_age_ticks
transition { from, to }
health
quality
motion
contrast
```

The complete encoded JSON payload is bounded to 2048 bytes.

## Digest semantics

Every 32-byte digest in v1 is BLAKE3.

`scene_digest` identifies exact scene semantics or an agreed built-in scene definition.

`visual_plan_digest`, when present, binds the handoff to the exact visual plan/genome semantics that produced the current scene state.

Digests are content identifiers, not credentials.

All-zero digests are reserved as invalid so missing/uninitialized identity cannot accidentally look meaningful.

## Visual seed privacy

`visual_seed` exists only to preserve deterministic visual identity.

It must:

- never be used as a cryptographic key;
- never authorize anything;
- remain local to the visual/lifecycle stack;
- not be uploaded as telemetry by default;
- not encode a username, hostname, serial number, MAC address, SSID, peer identity, or other direct identifier.

If derived from a longer-lived local visual seed, consumers must treat it as pseudonymous presentation material rather than public metadata.

## Ephemeral replay lineage

Every active lifecycle chain receives a fresh non-zero 128-bit `continuity_lineage`.

The lineage exists to distinguish unrelated chains and stale payloads. It is not a secret and carries no authority.

Within one lineage, `handoff_sequence` starts above zero and strictly increases.

A successor is accepted only when:

```text
same continuity_lineage
AND new_sequence > previous_sequence
AND new_world_age >= previous_world_age
AND previous.transition.to == next.transition.from
```

A producer/consumer coordinator owns explicit adoption of a new lineage. Consumers must not silently reinterpret an unrelated lineage as a successor.

## Lifecycle surfaces

v1 uses bounded enum values:

```text
Boot
Greeter
Session
Lock
Suspended
Recovery
Shutdown
```

The current conservative transition graph permits:

```text
Boot -> Greeter | Session | Recovery

Greeter -> Session | Suspended | Recovery | Shutdown

Session -> Greeter | Lock | Suspended | Recovery | Shutdown

Lock -> Greeter | Session | Suspended | Recovery | Shutdown

Suspended -> Greeter | Session | Lock | Recovery

Recovery -> Greeter | Session | Shutdown
```

Direct `Boot -> Session` supports intentionally configured autologin without requiring a fake greeter transition.

`Session -> Greeter` and `Lock -> Greeter` cover logout/user-switch behavior.

`Shutdown` has no outgoing v1 transition.

## Fixed-point scene phase

`phase_micros` is an integer normalized phase:

```text
0         = start
1_000_000 = end
```

The name denotes millionths of normalized phase, not wall-clock microseconds.

Persistent semantic state uses this integer representation so cross-runtime continuity is not dependent on floating-point serialization or GPU behavior. Renderers may convert it to local floating point for interpolation.

## World age

`world_age_ticks` is a monotonic semantic age for the visual world.

It is not Unix time and must not encode wall-clock identity.

A successor may keep age unchanged or advance it, but may not move backward within one continuity lineage.

Resume implementations should analytically advance world age/state rather than replaying hidden render frames.

## Health

The ABI exposes only the coarse presentation classes:

```text
Normal
Delayed
Degraded
Failed
Unknown
```

`Unknown` is distinct from `Normal`.

The continuity layer does not decide health. Boot health comes from the authoritative boot protocol/observer; later lifecycle surfaces use their own authoritative system/session source.

A renderer may interpret health visually but cannot promote or bless it.

## Accessibility and fidelity

v1 transports independent bounded policies:

```text
quality  = Calm | Standard | Rich
motion   = Reduced | Standard
contrast = Standard | High
```

They are deliberately separate:

- reduced motion should not imply low image fidelity;
- high contrast should not require disabling ambient rendering;
- Calm may reduce decorative complexity even when ordinary motion is allowed.

No important security or diagnostic state may exist only in animation/color.

## Encoding

The initial interoperability encoding is compact UTF-8 JSON produced in struct-field order by the Rust reference implementation.

The repository contains a canonical byte-for-byte fixture:

```text
crates/core/symthaea-spore-continuity/tests/fixtures/continuity-v1.json
```

and an integration test that requires the reference encoder to match it exactly.

If the project later adopts CBOR or another binary encoding, it must be introduced as an explicit version/encoding contract rather than silently changing v1 bytes.

## Transport and persistence

The ABI does not require one transport.

Recommended local lifecycle transport is a small `/run`-scoped handoff owned by the relevant trusted lifecycle coordinator.

Implementations should prefer **one-shot, target-specific consumption**:

```text
producer writes atomically
        |
        v
coordinator performs real lifecycle transition
        |
        v
consumer verifies target surface + lineage/sequence
        |
        v
consumer adopts state
        |
        v
handoff file is removed/rotated
```

Do not keep a single ancient `current.json` forever and let every future process trust it merely because it parses.

The actual OS state is always authoritative about which target surface is being entered.

For example, a Greeter should consume a `to=Greeter` handoff only when the host/session lifecycle actually says the Greeter is the current target.

## Atomicity

When file transport is used:

1. encode and validate in memory;
2. write a sibling temporary file;
3. apply restrictive permissions appropriate to the producer/consumer boundary;
4. rename atomically into the target-specific handoff path;
5. optionally sync only where durability is truly required.

Boot -> Greeter and ordinary session handoffs are ephemeral `/run` data and should not create unnecessary persistent write latency.

Shutdown history is a separate bounded persistent lifecycle mechanism; do not persist the entire continuity packet merely to make the next boot pretty.

## Security-boundary examples

### Boot -> Greeter

The boot renderer exports semantic state, then separately releases DRM. The display manager proceeds through host/systemd policy whether continuity succeeds or not.

### Greeter -> Session

Authentication remains owned by the normal greeter/PAM/session path. The newly authenticated user receives only validated presentation state after the session boundary is established.

### Session -> Lock

The unlocked desktop exports semantics only. Unlocked pixels/resources are not shared with the lock renderer. The compositor/session-lock mechanism remains authoritative.

### Suspend -> Resume

Continuity may preserve semantic age/phase, but failure to checkpoint can never veto suspend or resume.

## Failure behavior

Every consumer fails closed with respect to **continuity** and fail-open with respect to **OS availability**:

```text
bad continuity -> discard visual handoff
OS lifecycle   -> continues
visual result  -> safe built-in fallback scene
```

Reject:

- payload > `MAX_CONTINUITY_BYTES`;
- unsupported version;
- malformed JSON;
- all-zero lineage;
- zero sequence;
- invalid lifecycle transition;
- out-of-range phase;
- all-zero required digests/seeds;
- stale/equal sequence within an adopted lineage;
- world-age rewind;
- broken surface chain.

## Golden qualification

Before v1 is treated as stable:

1. Rust format/check/test/Clippy passes with the workspace-pinned toolchain;
2. canonical fixture remains byte-stable;
3. oversized/malformed/future-version cases are fuzzed or property-tested;
4. every legal transition has a positive test;
5. every illegal transition class has negative coverage;
6. replay and world-age rewind are rejected;
7. one-shot file transport is VM-tested at lifecycle boundaries;
8. visual consumers demonstrate fallback when the ABI is completely absent.

## Invariant

> Continuity may preserve the feeling of one living environment; it must never erase the fact that boot, greeter, session, lock, suspend, recovery, and shutdown are real independent system boundaries.
