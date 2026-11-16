# symthaea-operations-research

Operations research for Symthaea — a practical decision-science layer: inventory
optimization, queueing, and shortest paths.

Pure `std`, zero deps, no `symthaea-core` link. Closed-form + exact algorithms,
checked vs known values.

- `inventory::economic_order_quantity` (EOQ) + total cost.
- `queue::MM1` — ρ, L, Lq, W, Wq (Little's law).
- `graph::dijkstra` — single-source shortest paths.

```bash
cargo test -p symthaea-operations-research
```


## Workcell capability eligibility

`cell_eligibility::evaluate_cell_eligibility` evaluates one immutable production-operation requirement against one owner-resolved cell snapshot.

It returns exactly one of:

- `Eligible`: every evaluated hard requirement is explicitly satisfied by current evidence.
- `Blocked`: a known requirement is violated or known to be absent from a complete snapshot.
- `Indeterminate`: required evidence or owner facts are missing, unknown, or supplied through an incomplete snapshot.

The evaluator checks profile lifecycle and validity, profile qualification plus independent verification evidence, exact product/pattern/process/capability revisions, the requested operation and process configuration, asset state/capability/tooling/calibration, operator skill/availability/qualification, material-lot admission/envelope, utility units/capacity/availability, required clearances, and an evidenced inspection plan.

Reasons carry stable enum codes and subject references. Evidence references are sorted and deduplicated in the report. Duplicate matching owner references are rejected instead of allowing input order to select a favourable record.

This module consumes resolved facts. It does not parse the Atelier JSON profile, create canonical owner records, schedule production, verify source-document truth, authorize machine actuation, or certify facility safety. Callers must resolve evidence status and issuer trust through the responsible owners.

Run focused tests with:

```bash
cargo test -p symthaea-operations-research cell_eligibility
```

The unit tests are authored but have not been run in an environment with the Rust toolchain.


The B1 eligibility evaluator also checks the process configuration's own validity interval and exact parameter SHA-256 digest, rather than relying only on process-qualification evidence. Worker requirements carry a minimum qualified headcount; unresolved headcount is Indeterminate and a known shortfall is Blocked. Zero-person requirements and malformed digests are rejected.

The B1 eligibility evaluator also checks the process configuration's own validity interval and exact parameter SHA-256 digest, rather than relying only on process-qualification evidence. Worker requirements carry a minimum qualified headcount; unresolved headcount is Indeterminate and a known shortfall is Blocked. Zero-person requirements and malformed digests are rejected.


The crate now exposes an independent positive-claim verifier in `cell_eligibility_verifier`. It recomputes the hard predicates from the same immutable snapshot without calling the evaluator, and independently reconstructs the evidence-reference set. It is a soundness verifier for `Eligible` claims; it does not prove source-document truth, and it is not a second scheduler.
