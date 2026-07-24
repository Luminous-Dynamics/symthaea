# Human Data Governance

Series 52 treats privacy as a runtime contract rather than a documentation promise.

Every processing operation is bound to a short-lived authenticated permit containing a pseudonymous wearer digest, session identity, consent record, allowed data classes, allowed purposes, and maximum retention.

The gate enforces the following invariants:

- direct identifiers and reversible identity linkage are forbidden;
- raw kinematics and raw biometrics remain local;
- cognitive-state data cannot be exported;
- network telemetry is limited to redacted, aggregated derived biomechanics;
- release evidence must be redacted and aggregated;
- research, model training, and analytics require separate policy approval;
- expiry or revocation blocks processing and requires purge.

This subsystem has no actuator-authority output. Safety operation must remain possible with data export disabled.
