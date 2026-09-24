# symthaea-electroacoustics

Evidence-oriented electro-acoustic engineering primitives for Symthaea.

The initial EAC-001 surface is deliberately limited to provenance-bound transducer parameters and validation. It does **not** establish frequency response, enclosure behavior, numerical solver correctness, physical measurement, or perceptual quality.

Authority/source classes remain explicit:

- assumed;
- datasheet;
- analytically derived;
- numerically derived;
- directly measured;
- fitted from measurement.

Unknown optional quantities remain `None` and are never silently interpreted as physical zero.

The intended follow-on line is:

`EAC-001 parameters -> EAC-002 linear reference model -> enclosure/nonlinear models -> generic multiphysics -> FIELD measurement -> model discrepancy -> digital twin`.

Physical observation and calibration authority belongs to the FIELD architecture. Generic numerical orchestration belongs to `symthaea-sim-bridge`; this crate should not grow private replacements for either layer.
