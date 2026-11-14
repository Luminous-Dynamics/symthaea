# atelier-sewing-plan-core

Dependency-free deterministic validator for a proposed sewing-assembly operation graph.

## Validates

- Plan, source brief and source pattern digest shape and exact expected digest binding.
- Unique part, interface, machine-profile and operation identities.
- Pattern edge references against the exact declared part identities.
- Duplicate/missing part, edge, interface and operation references.
- Seam allowance and interface tolerance bounds in integer micrometres.
- Predecessor references, duplicate/self dependencies and cycles.
- Manual-versus-machine declaration consistency.
- Operation-kind support in the referenced machine profile.
- A deterministic topological order for a structurally valid acyclic operation graph.

## Does not validate

- Whether two physical edges are actually geometrically sewable or have matching lengths/notches.
- Fabric-specific needle, thread, stitch density, feed, presser-foot or tension suitability.
- Whether a machine-profile declaration is true or current; referenced profile evidence is owned upstream.
- Whether a construction sequence is ergonomic, safe, fast, economically viable or physically successful.
- Factory/resource scheduling, seam execution, robotics, machine control or actuation permission.

It is a planning contract validator, not a sewing compiler or machine controller. A valid graph means only that the declared identities, references, dependencies and machine-operation declarations are structurally consistent.

## Run

In a Rust 1.96.0 environment:

```bash
cargo test --manifest-path tools/atelier-sewing-plan-core/Cargo.toml --locked
cargo clippy --manifest-path tools/atelier-sewing-plan-core/Cargo.toml --locked --all-targets -- -D warnings
```

Twelve unit tests are authored, including unknown predecessor versus cycle classification. They have not been compiled or executed in the current authoring environment because the Rust toolchain is unavailable.

## Ownership

- Symthaea owns proposed plan synthesis, ambiguity/clarification and typed graph validation.
- Symtropy owns scheduling and physical process/robot execution.
- Mycelix owns canonical lot/process/inspection records.
- Sol-Atlas owns spatial geometry projection/uncertainty.
- A validated plan never authorizes actuation.
