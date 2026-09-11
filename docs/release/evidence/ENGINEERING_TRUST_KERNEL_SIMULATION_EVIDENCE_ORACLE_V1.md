# Engineering Trust Kernel — Simulation Evidence Admission Oracle V1

**Status:** independent reference oracle; structural admission semantics only  
**Branch:** `engineering/etk-2a-simulation-evidence-oracle`  
**Base:** `main` at `3afeee3d40af0bae0b85e70869571e024c28f07b`

## Theorem boundary

```text
simulation converged
!= engineering evidence
!= admissible simulation evidence
!= discharged proof obligation
!= qualified design
!= manufacturing approval
!= physical actuation authority
```

This oracle freezes the first narrow Engineering Trust Kernel evidence-admission contract. It decides only whether one candidate simulation artifact is structurally eligible to be admitted as `Simulation` evidence for one exact proof obligation.

An `Admit` result is **not** an obligation discharge receipt. It does not establish solver correctness, physical truth, model validity outside the declared validity domain, formal verification, qualification, certification, manufacturing approval, deployment approval, or physical authority.

## Independence

`scripts/etk-simulation-evidence-admission-oracle.py` is standard-library Python and imports no Symthaea code.

It is based directly on `main`, not on the ETK-1 Rust implementation branch. A future production Rust implementation must reproduce this contract independently rather than calling this script as its authority source.

The oracle deliberately reuses the existing simulation bridge vocabulary instead of defining a competing execution-evidence model:

- `external_solver` is the only admissible execution mode;
- `dry_run` and unknown/untrusted execution are not engineering evidence;
- backend, solver version, rendered-input digest, raw-output digest, and parser version are all required;
- a non-empty normalized metric set is required;
- run-level and metric-level epistemic/aleatoric uncertainty are bounded;
- metric-level uncertainty overrides run-level uncertainty when present;
- convergence is necessary but not sufficient.

## V1 admission bindings

A candidate can be admitted only when all of the following are true:

1. schema and closed-world field sets are exact;
2. the obligation expects `Simulation` and the candidate is exactly `Simulation` evidence;
3. obligation ID and obligation revision match exactly;
4. engineering subject matches exactly;
5. design/twin revision matches exactly;
6. accepted requirement revision matches exactly;
7. evidence-policy ID is explicit;
8. simulation request ID matches exactly;
9. validity-domain ID matches exactly;
10. candidate currentness is exactly `Current` and a currentness-proof reference is present;
11. execution mode is exactly `external_solver`;
12. the solver result converged;
13. confidence is finite and within `[0, 1]`;
14. run uncertainty is well formed;
15. normalized metrics are present;
16. the required metric appears exactly once with the required unit;
17. the declared inequality predicate and uncertainty budget are valid;
18. the effective metric uncertainty is within that budget;
19. the metric value lies inside its declared interval when an interval is present;
20. the acceptance predicate holds at the conservative interval boundary when an interval is present;
21. backend, solver version, input digest, output digest, and parser version are all present;
22. observed input digest equals the expected rendered-input digest;
23. candidate artifact identity and source-lineage identity are present.

V1 intentionally supports only scalar inequality predicates: `<`, `<=`, `>`, and `>=`. More complex acceptance logic must receive a new version rather than being smuggled into free-form strings.

For a `<=` or `<` obligation, an uncertainty interval is checked at its upper bound. For a `>=` or `>` obligation, it is checked at its lower bound. Therefore a mean value cannot pass while the declared interval crosses the acceptance threshold.

## Deterministic decision

The oracle returns exactly one of:

```text
Admit { admitted_evidence_id, obligation_id, candidate_artifact_id,
        currentness, validity_domain_id }
```

or:

```text
Deny { ordered_reasons[] }
```

There is no scalar trust score and no optimizer override. Multiple simultaneous failures are returned in a frozen deterministic reason order.

The admitted-evidence identity is SHA-256 over a domain-separated canonical JSON preimage containing the exact schema, obligation, candidate, uncertainty declarations, and expected-input binding. This identifier is an audit/content identity only; possession of it grants no authority.

## Synthetic positive fixture

The built-in positive fixture binds:

- obligation `O-structural-stress-42:r3`;
- evidence policy `ETK-SIM-ADMISSION-V1`;
- subject `bracket-alpha`;
- design revision `design:G17`;
- requirement revision `REQ-STRESS:r5`;
- request `sim-static-G17-LC9`;
- validity domain `VD-static-G17-LC9`;
- required metric `max_stress_mpa <= 250 MPa`;
- uncertainty budget `epistemic <= 0.2`, `aleatoric <= 0.1`;
- observed stress `181.2 MPa` with interval `[175, 190] MPa`;
- external CalculiX `2.22` provenance;
- exact input/output/parser identities;
- a currentness-proof reference and source-lineage reference.

The frozen admitted-evidence vector is:

```text
sha256:7066c8509f0563484acc3a2d16d5b9b689606a2250389da56ad66560dbc83ff8
```

## Adversarial self-test coverage

The built-in self-test requires fail-closed denial for:

- dry-run execution;
- historically valid but non-current evidence;
- missing currentness-proof reference;
- rendered-input digest substitution;
- design/twin revision substitution;
- wrong evidence kind;
- validity-domain substitution;
- missing parser provenance;
- absent required metric;
- direct acceptance-threshold failure;
- a point estimate that passes while its uncertainty interval crosses the threshold;
- uncertainty exceeding the obligation's epistemic budget;
- malformed/inverted uncertainty intervals;
- non-converged simulation;
- request-ID substitution;
- unknown/shadow execution fields;
- non-finite confidence;
- deterministic ordered reporting of simultaneous independent faults.

## Exact local execution evidence

After the uncertainty hardening, the exact checked-in candidate bytes were executed with:

```text
Python 3.13.5
python3 /tmp/etk-oracle-gh.py --self-test
```

Result:

```text
ok sha256:7066c8509f0563484acc3a2d16d5b9b689606a2250389da56ad66560dbc83ff8
```

`python3 -m py_compile /tmp/etk-oracle-gh.py` also completed successfully.

Executed source SHA-256:

```text
48d6d9dc89cd64839e0ffb8e24695c9317360d6a9fc4b28ad6d3ba48f1931360
```

Executed source Git blob identity:

```text
256ba3e2777d44544b18df35742c02aa43d78e3c
```

GitHub reports the checked-in script with the same Git blob identity `256ba3e2777d44544b18df35742c02aa43d78e3c`. Therefore the locally executed candidate bytes and the checked-in oracle bytes are identical.

This local execution is useful implementation evidence, but it is **not repository qualification**. Exact-head CI remains a separate evidence boundary.

## Deliberate non-goals

V1 does not establish multi-source evidence independence, contradiction handling, supersession, semantic-staleness graphs, authenticated currentness, formal-proof admission, telemetry/test/standard admission, discharge receipts, safety-case closure, design qualification, or transition authority.

In particular:

- `source_lineage_id` is retained for later independence analysis, but distinct strings are not treated as proof of independence;
- `currentness_proof_id` is a binding/reference slot, not proof that the referenced currentness mechanism is itself trustworthy;
- the uncertainty budget is an admission constraint, not a claim that the uncertainty model is complete or calibrated.

## Production follow-up

The next production tranche should compose this frozen reference behavior with the existing `symthaea-sim-bridge` types and `symthaea-formal-safety` obligations:

```text
SimulationResult
-> candidate simulation evidence
-> ETK admission decision
-> AdmittedSimulationEvidenceV1
-> separately constructed ObligationDischargeReceiptV1
```

The current shortcut `converged simulation -> discharged Simulation obligation` should then be removed. Admission and discharge must remain different propositions.
