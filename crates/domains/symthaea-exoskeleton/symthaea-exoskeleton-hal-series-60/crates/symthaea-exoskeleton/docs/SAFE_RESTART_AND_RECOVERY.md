# Safe Restart and Recovery

A restart is a new safety session. The system must never restore:

- an armed or active lifecycle state;
- a prior wearer-consent permit;
- a prior actuator command;
- learned authority or a Phi-derived tier;
- an in-flight deployment permit.

A recovery checkpoint is accepted only when it is authenticated, current, monotonic, deployment-bound, calibration-bound, and records zero final authority with actuators disabled. Watchdog, brownout, panic, unknown, and unclean resets require a physical inspection covering the emergency release, actuator zero state, and attachment integrity.

Even a clean operator restart may only return the system to `Standby`. A fresh consent session followed by explicit Arm and Activate actions remains mandatory.

The checkpoint must also match the last authenticated safety-journal head and the expected reliability-state digest. A syntactically valid checkpoint cannot fork or replace persistent maintenance history.
