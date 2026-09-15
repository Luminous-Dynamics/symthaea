# Bootstrap syscall confinement

Linux x86_64 supervisor theorem for the interval from the existing `PTRACE_EVENT_EXEC` stop to a signed checkpoint-one ready token. Every post-exec syscall entry is mediated with `PTRACE_SYSCALL`; only a code-defined bootstrap-safe profile is allowed, except for one exact ready-token write. The supervisor independently verifies that token against the reviewed runtime policy and the opaque launch challenge it issued, confirms the exact bytes arrived on the peer pipe, and re-observes the critical exec-preserved authority before producing a still-stopped release capability.

The theorem does not establish instruction-by-instruction mapping continuity, trusted time, exclusive pipe-writer authority, generic syscall safety outside the fixed profile, or physical authority.
