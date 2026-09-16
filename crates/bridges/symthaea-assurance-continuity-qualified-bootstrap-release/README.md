# Continuity-qualified bootstrap release

Consumes the exact syscall-confinement qualification and the matching file-backed executable-mapping continuity qualification before releasing the ptrace-stopped verifier. The release path rechecks that the tracee is still in a tracing stop owned by the current supervisor immediately before `ptrace::detach`.

This establishes a stronger preferred release gate. It does not by itself remove the weaker public release helper still present in the frozen lower #3514 layer; that lower-layer bypass must be removed before the full stack is considered merge-ready. External privileged mutation, trusted time, kernel-pseudo mapping identity continuity, and physical authority remain outside this theorem.
