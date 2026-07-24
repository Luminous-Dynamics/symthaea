# Safety co-processor and atomic actuation contract

The powered exoskeleton architecture uses two independent decisions:

1. The main controller produces and safety-filters a physical actuator frame.
2. An independent safety controller observes its own E-stop loop, watchdog, power stage, hardware identity and critical sensors, then grants a short command-bound power-enable lease.

The actuator backend stages all channels and returns an authenticated acknowledgement. A commit permit is valid only when:

- transaction, boot and frame sequences match;
- frame digest, channel mask and backend identity match;
- calibration and deployment bindings match;
- every required channel is staged without a fault;
- acknowledgement age is bounded;
- the commit window ends before the independent safety lease.

Any mismatch latches the transaction or safety-controller supervisor. Clearing a fault never restores power or replays a command; a new inspection, challenge, prepare and commit sequence is required.
