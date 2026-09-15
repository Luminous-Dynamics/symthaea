# Critical exec-preserved authority

Linux-only assurance bridge that observes the critical authority a confirmed static verifier inherits across exec while it remains stopped at `PTRACE_EVENT_EXEC`.

It binds an exact fd inventory, selected credentials/capabilities/signal/security state, cwd/root, namespace identity, resource-limit bytes and cgroup membership to reviewed policy. It deliberately does not claim all process state, seccomp-filter semantics, fd peer behavior, trusted time, or post-release continuity.
