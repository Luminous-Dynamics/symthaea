# SPINE-000B-P1R Exit Gates

P1R is complete only when all of the following are true on one exact frozen head:

1. The branch descends from `fa7b68645dd2109abb12c70f493e50e16c5740a4`.
2. Only the allowed P1R subject files differ from that base subject.
3. Fixture coverage contains every required tag enforced by `verify_spine_000b_p1r_contract.py`.
4. Python oracle execution is mandatory and its self-test passes.
5. Fixture regeneration is deterministic and produces no diff.
6. Production Rust `OutputCollector` matches the independent oracle for top-level integrated output.
7. Production Rust leave-one-out recomputation matches every oracle `integrated_without_subject` canonical field bit-for-bit.
8. Derived `changed_channels`, `uniquely_contributed_flags`, and `integration_changed` also match.
9. The targeted Rust equivalence test passes on the exact subject.
10. Postflight checkout immutability passes.
11. The evidence artifact binds exact HEAD/tree, relevant file hashes, Cargo manifest/lock hashes, and the non-authority claim boundary.

Until all eleven gates pass, runtime SPINE-000B proposal receipts remain blocked.
