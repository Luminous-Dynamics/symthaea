# Cyber-Physical Command Firewall

Series 50 separates network availability from actuator authority.

Only the identified local real-time controller may submit an enabled actuator frame. Independent safety, remote-diagnostic, and maintenance ingress paths are structurally non-actuating. A remote-network partition therefore removes remote observability without abruptly invalidating an otherwise healthy local control loop.

The firewall validates:

- authenticated source identity;
- boot epoch and measured deployment binding;
- monotonic transport sequence;
- packet freshness;
- minimum and maximum command cadence;
- ingress-specific actuation capability.

Source substitution, replay, burst injection, stale traffic, deployment mismatch, or an enabled frame on a non-actuating ingress latches a power-disable decision. Clearing the latch requires external inspection and does not re-arm the system.
