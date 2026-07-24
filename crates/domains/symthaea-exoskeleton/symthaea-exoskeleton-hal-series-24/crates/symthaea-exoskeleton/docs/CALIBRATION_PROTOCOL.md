# Exoskeleton Calibration Protocol

Passing software tests does not make a calibration profile suitable for a
wearable device. This protocol defines the minimum evidence represented by a
profile and the conditions under which it may be loaded.

## K0 — profile identity

Every profile has a non-empty immutable identifier and a positive revision.
The identifier belongs in every hardware trace and evidence bundle.

## K1 — sensor polarity and zero

Each joint encoder is moved through a mechanically constrained positive motion.
The canonical sign, zero offset, and scale are recorded. Direction is exactly
`-1` or `+1`; ambiguous or near-zero polarity is rejected.

## K2 — actuator polarity and derating

Actuator polarity is verified at current-limited bench power before attachment
to a person. `actuator_scale` may derate the canonical command but may not
exceed `1.0`. A profile cannot increase the configured torque rating.

## K3 — travel envelope

Calibrated joint limits must be ordered, finite, narrower than one revolution,
and contain the neutral pose. Software limits remain inside mechanical stops.

## K4 — anthropometry

Wearer mass, thigh length, and shank length are recorded for simulation and
control-model selection. The broad software plausibility bounds detect obvious
unit or transcription errors; they do not validate fit.

## K5 — loading ceremony

A new profile may be loaded only while disarmed. Validation, profile identity,
hardware serial association, and operator acknowledgement are recorded before a
separate arm request is accepted.
