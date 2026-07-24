# Persistent Safety Journal

The safety journal prevents a restart from erasing accumulated risk or rollback state.

Each externally persisted entry is:

- sequence ordered;
- hash linked to the previous verified entry;
- authenticated by a deployment-supplied verifier;
- bound to deployment and calibration digests;
- monotonic in security epoch, build number, calibration revision, runtime, energy, reversals, faults, and deadline misses;
- constrained so stop, maintenance, standby, arm, boot, and shutdown records contain zero authority with actuators disabled.

Journal verification latches closed on any chain break, authentication failure, rollback, counter decrease, invalid event transition, or binding mismatch. Maintenance debt can only be cleared by a `MaintenanceCompleted` record carrying independent evidence.

The crate intentionally does not choose a signature algorithm or storage technology. Production deployment should use authenticated, power-fail-safe storage with a hardware-backed monotonic anchor where available.
