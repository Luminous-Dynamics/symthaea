# Series 72 Release Gates

Series 72 is admitted only when all earlier release gates remain satisfied and:

1. canonical HAL headers and actuator-intent payloads pass exact-byte vectors;
2. checksum, reserved-field, version, producer, boot, sequence, freshness, and
   binding failures are rejected;
3. the declared cross-crate public API contract passes;
4. default, `symtropy`, `sensors`, `hal`, and all-feature compile probes pass;
5. every missing path dependency is explicitly declared as surrounding-workspace
   input rather than silently ignored;
6. no production Rust path contains `unwrap`, `expect`, `panic!`, `todo!`, or
   `unimplemented!`;
7. the exhaustive 64-case actuation protocol model has exactly one accepted
   case and rejects replay;
8. rustfmt, Clippy with warnings denied, locked tests, dependency policy, audit,
   Miri, sanitizers, HIL, bench, endurance, and human-factors gates pass in the
   complete workspace.

Series 72 does not authorize powered human-worn operation.
