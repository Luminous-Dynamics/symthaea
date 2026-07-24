# Series 66 Release Gates

Series 66 is admitted only when all earlier release gates remain green and the
following additional evidence is present:

1. final authority is quantized conservatively and invalid numbers remove all
   authority;
2. every cross-process safety contract is versioned, producer-bound,
   boot-bound, sequence-continuous, and payload-digest-bound;
3. the actuation cycle demonstrates bounded execution, stack, lock wait, heap,
   and blocking-I/O behavior under instrumented runtime measurement;
4. all safety domains agree on one release, deployment, calibration, hardware,
   journal head, boot epoch, and protocol version;
5. the pinned Rust toolchain builds every required feature and target with
   locked dependencies, formatting, Clippy warnings denied, tests, examples,
   dependency policy, vulnerability audit, Miri, and sanitizers;
6. repository evidence keeps `human_worn_authorized` false.

Passing these gates establishes compile and contract readiness only. It does not
replace HIL, bench, endurance, EMC, environmental, or supervised human-factors
qualification.
