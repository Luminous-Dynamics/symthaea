# MON-054 — Independent source inventory and failure-domain oracle

A standard-library-only Go reference model for independently corroborated source inventory and capture-coverage adjudication.

## Core model

A scope requires an exact preregistered witness set, at least two witnesses, trust-qualified and complete assertions, exact agreement on the inventory member set, and pairwise disjoint declarations across seven failure-domain dimensions:

- control plane
- identity root
- credential authority
- network path
- time source
- administrative path
- storage plane

A shared value in any dimension means `shared_failure_domain`. A missing value means `independence_unknown`. Only complete, disjoint declared vectors can qualify as independent within the declared model.

Each source member is reconciled against capture and capability evidence as `covered`, `missing`, `excluded`, `unsupported` or `unknown`. An authorized exclusion remains visible and is not counted as covered. Undocumented captured extras and excluded-plus-captured contradictions fail closed.

## Run

```sh
./CHECKS.sh
go test -run '^$' -bench . -benchmem -benchtime=1x ./...
```

## Limitations

Witness trust and completeness are upstream booleans in this reference model; this package does not cryptographically authenticate attestations, discover hidden common-cause paths, or certify production inventory completeness. Declared-domain disjointness is not proof of absolute independence. Benchmark figures are smoke observations, not stable performance comparisons or SOTA claims.
