# Deterministic Crash-Boundary Testing

The test build contains one-shot, thread-local failpoints at the persistence
boundaries that matter to recovery semantics. Production builds compile the
checks as no-ops and expose no fault-control API.

Covered boundaries:

| Boundary | Expected recovering-open result |
| --- | --- |
| append entry flushed, header not committed | trailing entry remains invisible and is not resurrected |
| delete status flushed, header not committed | metadata recovery reconciles the durable tombstone |
| batch journal synchronized, data untouched | complete batch rollback |
| batch data flushed, header not committed | complete batch rollback using journal records |
| batch header committed, journal not removed | committed batch validation and journal finalization |

The tests assert both the immediate fail-stop behavior of the original handle
and the state observed after dropping it and reopening through
`HdcStore::open_recovering`.

These failpoints model process termination between ordered persistence steps;
they are not a substitute for filesystem, kernel, power-loss, or hardware fault
testing. The same scenarios should eventually be repeated in a subprocess
harness that terminates the writer with `SIGKILL` and validates the store in a
fresh process.
