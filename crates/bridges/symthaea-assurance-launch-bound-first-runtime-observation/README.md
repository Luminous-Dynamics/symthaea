# Launch-bound first runtime observation

Linux-only tracee-side assurance bridge that consumes one canonical launch handoff ticket from the reviewed inherited pipe, proves the current OS PID and runtime identity match that ticket, binds its challenge to checkpoint sequence 1 with no predecessor, and only then invokes the live fresh mapped-runtime observer.

This establishes ticket consumption by the launched OS process and challenge-before-observation ordering. Supervisor-issued-ticket authenticity, signed checkpoint inclusion, uninterrupted continuity since exec, exclusive pipe-writer authority, trusted time, and physical authority remain separate claims.
