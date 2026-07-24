# Biomechanical Assistance Envelope

Series 47 adds a wearer-specific envelope downstream of learned assistance.
The profile is bound to wearer and calibration identities and may only narrow
the harder mechanical limits.

The envelope supervises:

- identified joint range of motion and soft margins;
- exoskeleton-to-human torque amplification;
- positive mechanical work per joint and stride;
- cumulative positive work over the session;
- bilateral assistance asymmetry;
- explicit wearer stop input; and
- a bounded discomfort signal.

Warnings reduce authority. Range violations, stale or invalid observations,
high discomfort, wearer stop, or exhaustion of the session work budget latch
zero authority. Reset requires wearer acknowledgement or a new session and does
not arm the device.

These quantities are engineering constraints, not medical diagnosis or clinical
prescription. Profile identification, clinical suitability, fatigue modelling,
and safe limits require qualified human-subject and biomechanics studies.
