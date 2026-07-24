# Operational Lifecycle

The canonical lifecycle is:

`Boot → SelfTest → Standby → Armed → Active`

`Active` may enter `Degraded`. Critical conditions enter `SafeStopLatched`. Scheduled or exposure-triggered servicing enters `Maintenance`.

## Non-negotiable transitions

- Power-up never enters an actuating state.
- Arming and activation are separate explicit operator actions.
- A safe stop never auto-rearms or clears itself.
- Inspection acknowledgement returns only to Standby, never directly to Armed or Active.
- Recovery from Degraded requires stable readiness for the configured dwell and a new activation request.
- Maintenance completion requires the maintenance cause to be cleared and an inspection acknowledgement.

The lifecycle combines independent authority ceilings by minimum. It may not raise any subsystem's limit.
