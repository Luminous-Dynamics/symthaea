# Safety Time Integrity

Safety deadlines are valid only when their clock domain is identified and continuous.

`ClockIntegrityMonitor` checks:

- fixed clock-source identity;
- a reboot/monotonic-reset epoch;
- monotonic source and host timestamps;
- monotonic sequence numbers and missing samples;
- sample age and impossible future receipt times;
- offset discontinuities and source-rate error;
- a configurable run of good samples before trust is granted.

A source substitution, reboot epoch change, replay, reorder, or backward timestamp latches rejection. Clearing that latch requires an independent inspection and explicit installation of the verified boot epoch. Synchronizing and rejected clocks grant zero authority; degraded clocks may only grant the configured restrictive ceiling.

This monitor validates time continuity. It does not provide PTP/NTP synchronization, oscillator qualification, or independent hardware clock redundancy.
