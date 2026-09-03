# symthaea-iot-actuation

Opaque type-state composition for consequential IoT/actuator effects.

Use this crate when code may eventually transmit a physical command. The lower `symthaea-iot-*` decision/checkpoint crates remain independently useful for audit, simulation, persistence and verification, but their serializable receipts are not actuation capabilities.

The intended runtime path is:

`validate_actuation -> prepare_actuation_dispatch -> persist combined checkpoint -> confirm_persisted -> revalidate_before_send -> ReadyActuationPermit`

`ArmedActuationPermit` intentionally exposes no command getter. Only a fresh `ReadyActuationPermit` exposes the exact command to an egress adapter.

A future Xenia egress should additionally require Xenia authenticated-session evidence. Connectivity/authentication never creates physical authority.
