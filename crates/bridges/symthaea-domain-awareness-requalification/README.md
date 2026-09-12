# symthaea-domain-awareness-requalification

Adapters from existing Symthaea assurance reports into the hysteretic `symthaea-assurance-requalification` gate.

The adapters deliberately convert upstream reports into **requalification samples**, not direct state changes. The gate still owns recovery semantics.

## Mappings

- model assurance: `Aligned -> Nominal`, `Restricted -> Restricted`, `Unsafe -> Unsafe`, `Incomplete -> Incomplete`
- an internally inconsistent `Aligned` model report with issues is treated as `Incomplete`
- common-cause diversity: any fail-closed/inconsistent/incomplete report -> `FailClosed`; otherwise `Nominal`
- common-cause-qualified track assurance: downgraded/internally inconsistent result -> `Restricted`; otherwise `Nominal`

A later clean report therefore becomes recovery evidence only. It cannot clear a previously latched restriction unless an explicit reviewed requalification authorization exists and the full recovery policy is satisfied.

This crate does not grant physical authority or implement actuation.

```bash
cargo test -p symthaea-domain-awareness-requalification
```
