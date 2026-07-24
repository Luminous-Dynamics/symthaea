# Diverse Controller Arbitration

Series 49 requires a learned primary controller and an independently implemented monitor controller to produce fresh, identity-distinct candidates for the same control instant.

The arbiter is intentionally asymmetric:

- it may preserve the lower-magnitude agreeing command;
- it may reduce authority;
- it may return a fully backdrivable command;
- it may latch a stop;
- it may never select the larger candidate or invent a replacement gait.

Agreement covers torque direction, per-joint magnitude, stiffness, damping, sequence proximity, confidence, freshness, controller identity, and software diversity. Repeated disagreements consume a bounded budget and then latch zero authority until an external inspection ceremony.

This mechanism supplements, but does not replace, the final safety kernel, passivity supervisor, actuator diagnostics, independent power lease, or atomic actuation transaction.
