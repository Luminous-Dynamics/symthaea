# Release-State Convergence

Qualification gates count only when they are simultaneously fresh and identically bound to the release, source tree, deployment, calibration, hardware, and SBOM under review. Failed or withdrawn gates latch rejection. Replayed attestations are rejected.

After all required gates remain stable for the configured dwell, two distinct authenticated roles may issue a short-lived release permit. The repository permit explicitly leaves human-worn authorization false.
