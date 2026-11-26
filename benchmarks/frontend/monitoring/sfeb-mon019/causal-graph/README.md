MON-019 causal-graph semantic reference

A standard-library Go research prototype for validating a bounded input-to-activity provenance graph. It checks typed node/edge relationships, unique identities, evidence references, causal DAG acyclicity, entity external origin/generation, per-activity bounded capture manifests, equality between declared, independently observed and graph-linked input sets, and normalized verifier reports bound to exact entity digests and pinned verifier ID/version/build/trust-store metadata.

It does not implement cryptography, transport authentication, a network egress collector, global completeness, factual truth, source accuracy, model internals, action authorization or destination effect. These must be supplied by independent adapters and MON-013–018.

Run: go test -race -count=5 ./..., go vet ./..., go test -cover ./..., go test -run=^$ -fuzz=FuzzDigestProfileIsolation -fuzztime=2s.
