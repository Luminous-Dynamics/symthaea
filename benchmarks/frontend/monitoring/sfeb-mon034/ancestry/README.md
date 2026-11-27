# MON-034 — PostgreSQL Timeline Ancestry Qualification

MON-034 makes PostgreSQL timeline history part of the recovery proof.

A timeline history file identifies the parent timeline, WAL switchpoint and reason for each fork. Each timeline has its own LSN interval. A timeline number is not a global clock, and an LSN from one timeline must not be numerically compared with an LSN from an unrelated branch.

## Checks

- base backup and recovered cluster system identifiers match the expected cluster;
- target TLI matches the supplied history file identity;
- non-root history bytes match the declared SHA-256;
- the base backup TLI is in the requested target's ancestry;
- the base backup LSN lies inside that source timeline's valid interval;
- target LSN lies after the requested target timeline's fork boundary;
- numeric LSNs are compared only when base and target refer to the same timeline;
- root timeline 1 can be represented without a history file.

Exact fork-boundary equality is indeterminate, not rounded into a branch decision. Invalid ancestry and out-of-interval LSNs are blocked.

## Use

```sh
cd benchmarks/frontend/monitoring/sfeb-mon034/ancestry
go run ./cmd/qualify --input INPUT-example.json
go run ./cmd/qualify --input INPUT-example-tli1.json
```

Exit codes: 0 qualified, 2 blocked, 3 indeterminate, 64 usage error, 65 invalid input.

The input, including raw timeline history text and digest, must itself be independently collected and custody-verified before MON-024/026/027 accepts the result. The module does not retrieve timeline history, archive WAL or restore a database.

## Validation

Run `bash CHECKS.sh` for JSON fixture sanity, Go vet, unit, race and benchmark checks. The benchmarks measure both 1,000-entry history parsing and full 1,000-timeline ancestry evaluation with allocation counts. They are local algorithmic microbenchmarks, not PostgreSQL or commercial backup-product performance comparisons. No PostgreSQL/Kafka service is started.
