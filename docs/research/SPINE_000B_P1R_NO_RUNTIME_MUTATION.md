# SPINE-000B-P1R No-Runtime-Mutation Rule

P1R may only harden measurement and qualification surfaces.

It must not change:

- subsystem scheduling;
- manager process outputs;
- `OutputCollector` non-empty integration equations;
- runtime state application;
- panic handling;
- Broca;
- epistemic authority;
- causal behavior.

The only allowed Rust change is to the P1 test surface inside `src/cognitive_loop/subsystem_trait.rs`.

If qualification reveals a production semantic defect rather than a test defect, stop P1R and open a new prerequisite lineage before modifying production behavior.
