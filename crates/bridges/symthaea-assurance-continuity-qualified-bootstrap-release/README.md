# Continuity-qualified bootstrap release

Consumes the exact syscall-confinement qualification and the matching file-backed executable-mapping continuity qualification before releasing the ptrace-stopped verifier. The release path rechecks that the tracee is still in a tracing stop owned by the current supervisor immediately before `ptrace::detach`.

In this stacked lineage the lower syscall-confinement crate no longer exports a public detach helper. The continuity-qualified constructor is therefore the only reviewed public safe-Rust release path for the confined bootstrap capability. Success consumes both opaque qualifications; failure returns them intact.

This API-surface theorem does not claim that a privileged debugger, kernel actor, arbitrary external process, or unreviewed unsafe code cannot alter or release the tracee. Post-release runtime continuity, trusted time, kernel-pseudo mapping identity continuity, and physical authority remain outside the theorem.
