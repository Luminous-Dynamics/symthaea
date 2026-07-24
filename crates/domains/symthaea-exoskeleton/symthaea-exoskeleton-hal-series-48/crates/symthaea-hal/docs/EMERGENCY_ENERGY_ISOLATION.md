# Emergency Energy Isolation

Series 46 distinguishes a zero software command from actual removal of stored
actuator energy. The independent isolation supervisor evaluates:

- two hardwired E-stop loop channels;
- contactor command and independent auxiliary-contact feedback;
- residual DC-link voltage and discharge timing;
- discharge-circuit health;
- release of the mechanically backdrivable path; and
- an independent watchdog.

A contactor that fails to open or close, a charged DC link that does not decay,
an unhealthy discharge path, E-stop disagreement, or failure to release the
backdrive path latches a physical-inspection fault. Reset requires a physical
reset input while the contactor is open, voltage is below the safe threshold,
and the mechanism is backdrivable.

Real implementations still require appropriately rated safety contactors,
forced-guided feedback, creepage and clearance analysis, discharge-component
sizing, fault-tolerant wiring, and measured worst-case isolation latency.
