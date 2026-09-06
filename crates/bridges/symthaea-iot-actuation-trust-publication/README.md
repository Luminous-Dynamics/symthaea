# Actuation Trust Publication v0.1

This crate provides the atomic publication meta-root for trust and policy state that is authoritative for privileged IoT actuation.

Transport trust, device-reality trust/policy, and controller/interlock trust/policy can be prepared independently, but they do not become current for actuation independently. A complete `ActuationTrustRootsV1` bundle becomes authoritative only after it is persisted as one canonical, crash-durable publication and its `ActuationTrustPublicationHead` is retained independently.

A successor publication is monotonic and fail-closed: trust generations cannot roll back, same-generation trust digests cannot be substituted, policy generations cannot roll back, policy digests cannot change without a generation advance, and a publication that changes nothing is rejected.

`CurrentActuationTrustFence<'a>` holds the same local and cross-process publication locks used by `publish_successor`. While that fence exists, another process or store instance cannot make a successor trust/policy bundle authoritative. This turns the three independently managed trust/policy families into one stable current view for a later actuation linearization attempt.

The publication root is not an additional authorization root. It is an atomicity/anti-rollback root over the already required transport, device-reality and interlock trust/policy roots. It performs no cryptographic verification and grants no final permit, JIT lease, HAL capability or physical-effect authority.

The later final boundary must still hold the trust-publication fence, durable admission-reservation fence and durable semantic-head fence; compare all of them with `ComposedActuationEvidence`; obtain all three owner-local cryptographic current fences; enforce one final common wall-clock deadline and short monotonic handoff ceiling; and consume exactly one HAL attempt.
