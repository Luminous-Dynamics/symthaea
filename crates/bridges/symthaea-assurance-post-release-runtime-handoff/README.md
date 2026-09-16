# Post-release runtime handoff

Creates the supervisor half of an exact post-release runtime re-entry theorem without relying on timestamps.

The bridge is intentionally two-stage. Before release, `retain_post_release_runtime_handoff_channel` borrows the exact syscall-confinement qualification and matching executable-mapping-continuity qualification that the release constructor will later consume by value. It rebinds those qualifications to the original opaque launch handoff/report, then creates a private `F_DUPFD_CLOEXEC` duplicate of the supervisor write end and verifies that the duplicate resolves to the same Linux pipe object at retention time.

After a successful `ReleasedBootstrapVerifier` exists, `issue_post_release_runtime_challenge` consumes that retained-channel capability. It requires the release to name the exact confinement and mapping qualifications seen at retention, independently reconstructs the consumed confinement qualification from its retained public report, independently re-verifies the exact #3509 bootstrap-ready wire, and requires the ready wire/checkpoint bytes to match the identities already frozen before release.

Only then does it re-observe the private retained descriptor, generate a fresh nonzero 256-bit OS-CSPRNG challenge, and write one bounded `PostReleaseRuntimeHandoffTicket` through that retained descriptor. The ticket binds the retention digest, exact release, exact pre-release parent qualifications, original launch handoff, ready wire/checkpoint, tracee PID, signed runtime process-instance ID, launch attestation, runtime policy/verifier/backend/static identity, checkpoint-one challenge/observation, checkpoint-one counter and the fresh post-release challenge. It fixes the intended successor sequence to checkpoint 2 with checkpoint 1 as predecessor.

The causal statement is therefore capability-based rather than clock-based: the channel duplicate existed while the exact pre-release qualifications still existed; those exact qualifications were consumed into the successful release; and the fresh challenge is generated only by the post-release constructor that consumes the retained channel.

This does **not** yet establish tracee consumption, causal OS-PID-to-runtime-process binding from the wire alone, checkpoint-two live observation, signed checkpoint-two inclusion, exclusive pipe-reader/writer authority, resistance to arbitrary same-process descriptor sabotage, trusted time, global replay resistance, mapping continuity after release, or physical authority. Those remain separate child theorems.
