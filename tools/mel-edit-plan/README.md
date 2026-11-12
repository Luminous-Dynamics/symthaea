# MEL edit-plan compiler (CQEP-001)

A pure-standard-library Rust reference compiler for normalized audio/music edit requests. It selects only declared operators whose typed inputs/outputs, capabilities, locality, determinism, risk, preconditions and write scope satisfy the request. It requires independent evidence observers before dependent stages can execute.

## Run

```sh
cargo fmt --manifest-path tools/mel-edit-plan/Cargo.toml --all -- --check
cargo test --manifest-path tools/mel-edit-plan/Cargo.toml --locked
cargo clippy --manifest-path tools/mel-edit-plan/Cargo.toml --all-targets --locked -- -D warnings
```

## Claim boundary

- This is a plan compiler, not a natural-language parser, model router, audio processor or executor.
- Profiles are declarations, not proof that an operator truly has the advertised capability.
- The compiler never invokes an operator or authorizes a mutation.
- A compiled plan uses stable length-prefixed canonical bytes. SHA-256 is an explicit external hashing boundary; this crate does not ship a misleading non-cryptographic digest.
- Runtime execution must revalidate parent digests, operator binary/weight digests, preconditions, budgets, output artifacts and independent evidence before each mutating node.
