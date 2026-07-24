# Series 66 Compile and Build Admission

The repository pins Rust 1.94.1 with `rustfmt` and `clippy`, defines a
machine-readable feature matrix, and provides a dependency-free structural
admission check that runs before Cargo.

Release verification must use locked dependencies. The complete matrix includes
both crates, default and all-feature builds, every exoskeleton integration
feature, all targets, examples, tests, Clippy with warnings denied, dependency
policy, vulnerability audit, Miri, and sanitizers. The latter two remain external
qualification requirements unless the local runner explicitly executes them.
