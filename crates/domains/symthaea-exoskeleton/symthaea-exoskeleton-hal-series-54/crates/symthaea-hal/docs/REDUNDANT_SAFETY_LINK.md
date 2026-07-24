# Redundant Safety Link

Series 44 defines the communication contract for messages that may affect
actuator power. One packet on one bus is never sufficient.

A message is accepted only when two independently routed, authenticated packets:

- use the expected protocol version, identities, and boot epoch;
- have strictly increasing sequence numbers;
- are fresh and within a bounded validity window;
- arrive within the permitted inter-route skew;
- have different route identities; and
- agree on message kind, payload digest, payload length, and expiry.

Any replay, disagreement, repeated route, stale packet, or missing peer packet
latches the receiver. Clearing requires physical inspection and does not enable
power. This contract does not claim the two routes are physically independent;
board layout, wiring, transceivers, power domains, and EMC evidence must prove it.
