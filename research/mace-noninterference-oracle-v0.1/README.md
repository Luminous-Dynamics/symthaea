# MACE-CoS paired-world noninterference oracle v0.1

A dependency-free Go research reference for one narrow question:

> When two executions have identical public inputs and differ only in protected input values, can any low-authorized sink observe a protected-dependent result without an exact, approved release rule?

This is a complementary oracle for MACE-CoS, not a replacement for its runtime validator or for the independent action-authority owner.

## Model

The fixture DSL supports deterministic `copy`, `concat`, `sha256`, `length`, and `constant` nodes. Derived nodes preserve the union of their source lineage. Sinks are observed only when their audience is listed in `Policy.LowAudiences`.

A paired-world comparison requires identical public inputs and identical protected-source key sets. Protected values may differ. Protected lineage reaching a low sink is rejected even when the two values happen to be equal, unless an exact, unexpired `ReleaseGrant` permits `length` of one named protected source to one audience for one purpose. The approved `length` operator must consume that protected source directly; a copied/aliased or merged intermediate is not accepted as the named source. The grant cannot be reused for arbitrary transforms, recipients, or purposes.

The evaluator receipt binds the exact program and policy digests plus sink identity, audience, purpose, whether protected lineage is present, and whether the exact release was authorized. It intentionally does not include output values or unsalted hashes of those values: a low-value hash can become a dictionary oracle for small or guessable domains. Keep the receipt itself in the evaluator trust domain; do not expose hidden labels or paired-world state to the agent. Protected input values are not hashed into the public receipt; the evaluator harness remains responsible for securely custodying and committing the exact hidden pair fixture.

## Run

Requires Go 1.23+ and the standard library only:

```sh
go test -count=1 ./...
go test -race -count=1 ./...
go vet ./...
gofmt -w model.go model_test.go
```

## What this demonstrates

- aliases and copies preserve source lineage;
- hashes do not erase protected provenance;
- merges union lineage across all inputs;
- an unrelated constant output is invariant to protected mutations;
- a narrow length release is accepted only under an exact, unexpired grant;
- release grants cannot be rebound through aliases;
- evaluator receipts do not hash low-visible output values;
- an unchanged protected value does not bypass release policy.

## Limits

This is an executable finite DSL, not a natural-language interpreter and not an LLM runtime monitor. It does not prove noninterference for arbitrary model hidden states, stochastic generation, timing, network side channels, tool implementations, or side effects. Its only declassification operator is byte length. The paired-world protocol must be adapted carefully for stochastic agents, with matched seeds, repeated trials, pre-registered equivalence margins, and independent adjudication.
