# Rust 1.92+ Compatibility Issues

**Date**: 2026-02-07
**Affects**: Holochain Sweettest Integration Tests
**Status**: Blocking - Awaiting Upstream Fixes

## Summary

Holochain Sweettest integration tests cannot run on Rust 1.92+ due to breaking changes in transitive dependencies. This affects all Mycelix hApps that use conductor-based integration testing.

## Affected Crates

| Crate | Version | Issue | Upstream Dependency |
|-------|---------|-------|---------------------|
| `mio` | 1.1.1 | `IoSource` trait methods not found | tokio → holochain |
| `wait-timeout` | 0.2.1 | `imp` module unresolved | criterion → benchmarks |

## Error Details

### mio 1.1.1

```
error[E0599]: no method named `register` found for struct `IoSource<std::net::TcpListener>`
  --> mio-1.1.1/src/net/tcp/listener.rs:215:20
```

The `IoSource` struct no longer implements the `Source` trait methods (`register`, `reregister`, `deregister`) in Rust 1.92+.

### wait-timeout 0.2.1

```
error[E0433]: failed to resolve: use of unresolved module or unlinked crate `imp`
  --> wait-timeout-0.2.1/src/lib.rs:68:9
```

The `imp` module (platform-specific implementation) is not being compiled correctly.

## What Works

| Component | Status |
|-----------|--------|
| Zome compilation to WASM | ✅ Works |
| DNA bundle packing (`hc dna pack`) | ✅ Works |
| Native Rust unit tests | ✅ Works |
| TypeScript SDK tests | ✅ Works |
| Sweettest integration tests | ❌ Blocked |

## Affected hApps

All Mycelix hApps with Sweettest configurations:
- mycelix-property
- mycelix-energy
- mycelix-knowledge
- mycelix-justice
- mycelix-identity
- mycelix-health

## Workarounds

### Option 1: Pin Rust Version (NOT VIABLE)

**Update 2026-02-07**: Pinning to Rust 1.81.0 does NOT work because:
- `sysinfo` 0.37.2 requires `edition2024` feature (Rust 1.85+)
- This creates a version conflict that cannot be resolved

The ecosystem is caught between:
- mio 1.1.1 / wait-timeout 0.2.1 → broken on Rust 1.92+
- sysinfo 0.37.2 → requires Rust 1.85+ (edition2024)

### Option 2: Update Test Dependencies

Force newer versions of affected crates in test `Cargo.toml`:

```toml
[patch.crates-io]
mio = { git = "https://github.com/tokio-rs/mio", branch = "master" }
wait-timeout = { git = "https://github.com/rust-lang/wait-timeout", branch = "master" }
```

### Option 3: Use TypeScript SDK Tests

The TypeScript SDK (`mycelix-workspace/sdk-ts/`) uses vitest and doesn't depend on Rust crates. These tests run successfully.

## Resolution Timeline

- **mio**: Waiting for tokio-rs/mio to release 1.1.2+ with Rust 1.92+ fixes
- **wait-timeout**: Waiting for rust-lang/wait-timeout to release 0.2.2+

## Testing Without Sweettest

Until resolved, test coverage can be maintained through:

1. **TypeScript SDK tests** - Full API coverage
2. **Zome unit tests** - Logic validation (native target)
3. **Manual conductor testing** - Using `hc sandbox`

## References

- [mio GitHub](https://github.com/tokio-rs/mio)
- [wait-timeout GitHub](https://github.com/rust-lang/wait-timeout)
- [Holochain Sweettest Docs](https://docs.rs/holochain/latest/holochain/sweettest/)
- [Rust 1.92 Release Notes](https://blog.rust-lang.org/2025/12/05/Rust-1.92.0.html)

---

*This document will be updated when upstream fixes are released.*
