# Kinetic Intervention Rate–Distortion point schema v0.1

This wire contract describes one candidate checkpoint measurement for #13420.

It records:
- physics cohort and initial-state identity;
- task contract, intervention-family digest, QoI registry, horizon and split;
- representation identity;
- serialized checkpoint payload and required decoding metadata;
- model/decoder assets, amortization rollout count and peak working-set memory separately;
- encode/decode/advance/observe/transition/fallback/training time and optional energy;
- separate microscopic, weak-observable, macro-QoI, intervention-response and sensitivity dispositions;
- physical gates, finite response-probe coverage / optional global-bound evidence and provenance.

## Validation layers

1. **Draft 2020-12 JSON Schema:** validates the record shape and local types.
2. **Rust semantic audit:** run `cargo run --bin kird_point_audit -- path/to/record.json`. It checks:
   - checkpoint-byte and persistent-asset arithmetic for overflow;
   - positive rollout counts and finite non-negative costs;
   - sealed-test acquisition prohibition;
   - synthetic-vs-physical execution authorization consistency;
   - valid SHA-256 syntax, and non-placeholder immutable provenance for physical records;
   - confirmed probe count not exceeding registered count;
   - `none`, `finite_empirical` and `certified_global` coverage consistency;
   - no global-bound claim from a finite probe set;
   - measured distortion PASS/FAIL consistency with normalized error and evidence;
   - response PASS requires error within tolerance, zero missing/misassociated branches and evidence;
   - physical-gate PASS requires every listed gate to pass; synthetic fixtures cannot pass physical gates.
3. **Independent scientific review:** verify actual digest targets, reference convergence, physical admissibility, branch association, split leakage and any claimed global response bound.

The semantic audit cannot prove that a digest resolves to an authentic external file, establish physical convergence, or turn a finite probe library into a continuous worst-case certificate. It is a fail-closed consistency layer, not a substitute for scientific evidence.

## Certified intervention cover (CIC)

The certified-global path is defined by [#13452](https://github.com/Luminous-Dynamics/symthaea/issues/13452). For a compact intervention metric space U, if the normalized response error e(u) has a verified global Lipschitz constant L and the confirmed probes form an r-net, then

`sup_U e <= max_P e + L*r + eta`

where eta is the sum of independently justified reference, candidate and measurement uncertainty budgets.

The KIRD semantic auditor requires a certified-global record to carry:
- metric identity, covering radius and verified Lipschitz constant;
- separate reference, candidate and measurement uncertainty terms;
- maximum sampled normalized response error and a conservative global upper bound;
- non-placeholder cover-set, bound-proof and branch-stability digests;
- an independent checker revision;
- explicit stable response-branch association.

It verifies the arithmetic `global_bound >= sampled_max + L*cover_radius + total_uncertainty`, checks that the declared sampled maximum is not below the worst confirmed response error, and refuses certification when branch association is unstable. A `PASS` response disposition under certified-global coverage additionally requires the upper bound to be at most one.

The checked-in `synthetic-kird-certified-cover-v0.1.json` is a manufactured mathematical example: U=[0,0.1], e(u)=0.5+2u, one probe at zero, r=0.1, L=2, and eta=0.05; the exact global upper bound is 0.75. This fixture tests the theorem and auditor only. It does not assert Lipschitz continuity of a Vlasov response map or certify any physical plasma model.


## Synthetic fixture

The included fixture uses placeholder all-zero digests, unknown distortion, unknown physical gates and `physical_execution_authorized=false`. Its byte and cost values are illustrative only and must not enter a scientific leaderboard.

## Status vocabulary

- `PASS`: the registered metric passed its frozen tolerance and evidence checks.
- `FAIL`: the registered metric or gate failed.
- `UNKNOWN`: the result is unresolved, unmeasured, under-resolved or below/near the noise floor.
- `UNSUPPORTED`: this representation does not produce the metric (for example, a fluid closure without distribution reconstruction on the microscopic tier).

A finite successful probe suite is `finite_empirical`, not `certified_global`.