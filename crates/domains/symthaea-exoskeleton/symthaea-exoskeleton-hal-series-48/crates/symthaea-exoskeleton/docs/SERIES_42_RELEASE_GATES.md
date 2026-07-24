# Series 42 release gates

Series 42 closes several system-level gaps, but it does **not** authorize powered human-worn use.

## Architecture invariants

1. The application processor cannot directly maintain actuator power. A physically independent safety controller grants only short-lived, authenticated, command-bound enable leases.
2. Multi-channel actuation is atomic: every required channel is staged before a commit permit is issued. Partial acknowledgement, replay, stale acknowledgement, or a lease that expires too soon latches rejection.
3. Degraded stopping never generates gait and never injects positive mechanical power. Uncertain state, uncertain contact, unsafe attachment loads, or unhealthy power electronics immediately select a fully backdrivable output.
4. Release evidence covers time, sensors, actuators, power, transport, the safety co-processor, calibration, consent, wearer attachment, persistence, and software execution.
5. The release artifact is bound to its source tree, lockfile, compiler, build recipe, SBOM and reviewed dependency inventory. At least two reproducible builds are required.
6. Every safety-critical claim traces to requirements, hazards, fault-injection evidence and independent review.

## Mandatory external evidence

Before powered human-worn testing, the project still requires:

- a physically independent safety controller and hardwired power contactor;
- backend-specific cryptographic authentication;
- measured command-to-torque HIL tests on every channel;
- electrical, thermal, EMC and power-fail bench campaigns;
- endurance and attachment-load validation;
- supervised human-factors evidence;
- independent hazard and regulatory review;
- a signed residual-risk acceptance decision.

Repository tests, simulations, generated reports, or a passing Series 42 example are not substitutes for this evidence.
