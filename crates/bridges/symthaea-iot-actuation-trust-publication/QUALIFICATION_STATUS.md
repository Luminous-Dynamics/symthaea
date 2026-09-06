# Qualification status

Status: **candidate / not yet compiler-qualified**.

Required exact-head evidence:

- one canonical publication atomically commits the authoritative transport trust head, device-reality trust head + policy anchor, and interlock trust head + policy anchor for one device;
- publication generation, predecessor commitment, and publication time are store-assigned rather than caller-supplied;
- trust generations cannot roll back, mutate their digest at the same generation, or advance without changing digest;
- policy generations cannot roll back, mutate their digest at the same generation, or advance without changing digest;
- no-op successors are rejected;
- initialization never overwrites existing state;
- publication writes use pinned descriptor-relative Linux paths, `O_NOFOLLOW`, bounded canonical encoding, file fsync, atomic rename, directory fsync, and exact read-back;
- existing publication state is opened only against an independently retained exact `ActuationTrustPublicationHead`;
- `publish_successor` consumes its store handle so the new head must be retained independently before another authoritative publication operation;
- `CurrentActuationTrustFence<'a>` holds both local and cross-process publication locks for its complete lifetime;
- a concurrent successor publisher is demonstrably blocked while the current fence is held and proceeds only after it is dropped;
- production contains no cryptographic verifier/provider, current transport/device/interlock cryptographic fence, final actuator permit, reusable JIT/HAL lease, network/process execution or unsafe surface;
- Rust 1.94 check/tests/package-local strict Clippy and Rust 1.96 formatting pass on the exact head;
- no unreviewed external sourced dependency drift is introduced.

The publication head is an atomicity/anti-rollback meta-root over existing trust/policy roots, not an additional source of actuation authority. Local workspace `Cargo.lock` bookkeeping remains to be intentionally frozen before promotion.
