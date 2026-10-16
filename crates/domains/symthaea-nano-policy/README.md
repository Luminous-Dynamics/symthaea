# symthaea-nano-policy

A small research-only Rust policy kernel for deterministic candidate selection from a frozen feature pool. It does not predict energy, spectra, stability, toxicity, synthesizability, or any other physical property.

## Contract

One JSON request per input line; one JSON response per output line. Example request:

    {"schema_version":"nano-policy-request-v1","policy":"farthest_first","budget":2,"seed":42,"feature_contract_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","candidates":[{"candidate_id":"c-a","family_id":"family-a","normalized_features":[0.0,0.0]},{"candidate_id":"c-b","family_id":"family-a","normalized_features":[0.2,0.2]},{"candidate_id":"c-c","family_id":"family-b","normalized_features":[1.0,1.0]}]}

Policies:
- random_seeded: deterministic pseudo-random sampling without replacement.
- farthest_first: greedy max-min distance over normalized features, starting from the lexicographically smallest candidate ID.
- group_balanced_farthest_first: seeded initial candidate, then prefers the least represented family before max-min distance.

## Schemas and benchmark criteria

- Request schema: schema/request.schema.json
- Response/error receipt schema: schema/response.schema.json
- Benchmark and go/no-go protocol: BENCHMARK.md

## Fail-closed contract

- Unknown fields are rejected. Target/label fields, optical target spectra, test membership and oracle observations must not be included.
- Candidate IDs must be unique; family IDs non-empty; vector dimension fixed; all feature values finite and within [0, 1].
- feature_contract_sha256 identifies the externally frozen feature and normalization definition.
- Request-line SHA-256 is returned with the result. It is a digest of exact input text, not proof that source features are true or legally usable.
- The selector performs no feature scaling or fitting. Fit transforms only on permitted training data and freeze them outside this program.
- No dataset, model checkpoint, external solver, network access, synthesis instruction, or production Python dependency is included.

Limits: 10,000 candidates, 256 features per candidate, 1,000 selected candidates, and 16 MiB per input line.

## Run and test

    cargo test -p symthaea-nano-policy
    cargo run -p symthaea-nano-policy < request.jsonl > response.jsonl

This is a baseline/adapter foundation. A Symthaea HDC representation or active-inference policy should be added as a separately measured arm after the feature contract and baselines are frozen. Descriptor-space diversity is neither physical novelty nor calibrated uncertainty.
