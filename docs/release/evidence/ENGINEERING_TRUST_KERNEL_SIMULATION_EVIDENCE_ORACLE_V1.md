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
- convergence is necessary but not sufficient.

## V1 admission bindings

A candidate can be admitted only when all of the following are true:

1. schema and closed-world field sets are exact;
2. the obligation expects `Simulation` and the candidate is exactly `Simulation` evidence;
3. obligation ID and obligation revision match exactly;
4. engineering subject matches exactly;
5. design/twin revision matches exactly;
6. accepted requirement revision matches exactly;
7. simulation request ID matches exactly;
8. validity-domain ID matches exactly;
9. candidate currentness is exactly `Current`;
10. execution mode is exactly `external_solver`;
11. the solver result converged;
12. confidence is finite and within `[0, 1]`;
13. normalized metrics are present;
14. the required metric appears exactly once with the required unit;
15. the declared inequality predicate is valid and satisfied;
16. backend, solver version, input digest, output digest, and parser version are all present;
17. observed input digest equals the expected rendered-input digest;
18. candidate artifact identity and source-lineage identity are present.

V1 intentionally supports only scalar inequality predicates: `<`, `<=`, `>`, and `>=`. More complex acceptance logic must receive a new version rather than being smuggled into free-form strings.

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

The admitted-evidence identity is SHA-256 over a domain-separated canonical JSON preimage containing the exact schema, obligation, candidate, and expected-input binding. This identifier is an audit/content identity only; possession of it grants no authority.

## Synthetic positive fixture

The built-in positive fixture binds:

- obligation `O-structural-stress-42:r3`;
- subject `bracket-alpha`;
- design revision `design:G17`;
- requirement revision `REQ-STRESS:r5`;
- request `sim-static-G17-LC9`;
- validity domain `VD-static-G17-LC9`;
- required metric `max_stress_mpa <= 250 MPa`;
- external CalculiX `2.22` provenance;
- exact input/output/parser identities;
- observed stress `181.2 MPa`.

The frozen admitted-evidence vector is:

```text
sha256:e18e040519bea87aaac413b385ef0c17362394c7d9bef0e8a31988b5ff23d536
```

## Adversarial self-test coverage

The built-in self-test requires fail-closed denial for:

- dry-run execution;
- historically valid but non-current evidence;
- rendered-input digest substitution;
- design/twin revision substitution;
- wrong evidence kind;
- validity-domain substitution;
- missing parser provenance;
- absent required metric;
- acceptance-threshold failure;
- non-converged simulation;
- request-ID substitution;
- unknown/shadow execution fields;
- non-finite confidence;
- deterministic ordered reporting of simultaneous independent faults.

## Exact local execution evidence

Before commit, the candidate source was executed with:

```text
Python 3.13.5
python3 /tmp/etk-oracle.py --self-test
```

Result:

```text
ok sha256:e18e040519bea87aaac413b385ef0c17362394c7d9bef0e8a31988b5ff23d536
```

`python3 -m py_compile /tmp/etk-oracle.py` also completed successfully.

Executed source SHA-256:

```text
a9856c33df96d8402e6a37fc61c2140e8abdf4069ba78eaad11d80c98aa23ae7
```

Executed source Git blob identity:

```text
4319077f551e47c8220bdab3bea2d63d43894fbb
```

After commit, GitHub reports the checked-in script with the same Git blob identity `4319077f551e47c8220bdab3bea2d63d43894fbb`. Therefore the locally executed candidate bytes and the checked-in oracle bytes are identical.

This local execution is useful implementation evidence, but it is **not repository qualification**. Exact-head CI remains a separate evidence boundary.

## Deliberate non-goals

V1 does not establish multi-source evidence independence, contradiction handling, supersession, semantic staleness graphs, uncertainty sufficiency, formal-proof admission, telemetry/test/standard admission, discharge receipts, safety-case closure, design qualification, or transition authority.

In particular, `source_lineage_id` is retained for later independence analysis, but V1 does not claim that distinct strings prove independent evidence.

## Production follow-up

The next production tranche should compose this frozen reference behavior with the existing `symthaea-sim-bridge` types and `symthaea-formal-safety` obligations:

```text
SimulationResult
-> candidate simulation evidence
-> ETK admission decision
-> AdmittedSimulationEvidenceV1
-> separately constructed ObligationDischargeReceiptV1
```

The current shortcut `converged simulation -> discharged simulation obligation` should then be removed. Admission and discharge must remain different propositions.
