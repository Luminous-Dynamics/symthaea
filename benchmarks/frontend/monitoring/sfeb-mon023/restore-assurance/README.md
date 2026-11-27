# MON-023 semantic recovery reference

A standard-library Go evaluator for normalized backup/WAL, restore execution, MON-022 surface coverage, dependency, application probe and service acceptance evidence.

It keeps three claim levels distinct: artifact_validated, database_recovered and service_recovered. A requested lower claim never auto-promotes to a higher one. Database recovery requires an isolated environment, exact source/database identity, consistency/catalog checks, target/lineage proof, MON-022 coverage, conservative RPO and database RTO. Service recovery additionally requires all required dependencies, current read/write/invariant probes, verified rollback on the write probe, old-primary and stale endpoint fencing, and independent service acceptance.

It calculates conservative RPO/RTO upper bounds from a shared trusted clock domain and bounded uncertainty. It consumes normalized evidence; it does not execute restore or authenticate upstream evidence.

Run: go test -race -count=5 ./..., go vet ./..., go test -count=1 -cover ./..., go test -run=^$ -fuzz=FuzzRPOUpperBound -fuzztime=2s, go test -run=^$ -bench=BenchmarkEvaluateServiceRecovery -benchtime=3x.


Additional invariants: the declared MON-022 surface scope must match the recovery plan and all required surface counts must reconcile. Service recovery requires read, write/readback (rollback verified) and business-invariant probes; missing/unverified evidence is indeterminate while a proven failed invariant is denied. A verified unavailable required dependency is denied; absent/stale/unverified dependency evidence is indeterminate. Traffic enablement is ordered after database readiness and fencing, and service acceptance follows traffic enablement. Conservative RPO/RTO upper bounds include clock uncertainty and require a common trusted clock domain.


The evaluator separates proven failure from absent proof: a verified dependency that is unavailable or a required business invariant that fails is denied; missing, stale or unverified dependency/probe evidence is indeterminate. It requires an explicit MON-022 scope match and equal required/covered surface counts. For service recovery it requires read, safe write/readback (rollback verified) and invariant probes, then verifies old-primary fencing, stale endpoint fencing, traffic-enable time and independent service-ready time. RPO and RTO are conservative upper bounds including clock uncertainty.
