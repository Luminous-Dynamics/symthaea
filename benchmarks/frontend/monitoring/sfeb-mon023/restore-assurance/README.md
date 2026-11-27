# MON-023 semantic recovery reference

A standard-library Go evaluator for normalized backup/WAL, restore execution, MON-022 surface coverage, dependency, application probe and service acceptance evidence.

It keeps three claim levels distinct: artifact_validated, database_recovered and service_recovered. A requested lower claim never auto-promotes to a higher one. Database recovery requires an isolated environment, exact source/database identity, consistency/catalog checks, target/lineage proof, MON-022 coverage, conservative RPO and database RTO. Service recovery additionally requires all required dependencies, current read/write/invariant probes, verified rollback on the write probe, old-primary and stale endpoint fencing, and independent service acceptance.

It calculates conservative RPO/RTO upper bounds from a shared trusted clock domain and bounded uncertainty. It consumes normalized evidence; it does not execute restore or authenticate upstream evidence.

Run: go test -race -count=5 ./..., go vet ./..., go test -count=1 -cover ./..., go test -run=^$ -fuzz=FuzzRPOUpperBound -fuzztime=2s, go test -run=^$ -bench=BenchmarkEvaluateServiceRecovery -benchtime=3x.
