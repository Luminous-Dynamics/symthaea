# Mycelix Finance -- Testing Guide

## Prerequisites

- Rust toolchain with `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
- Holochain CLI tools (`holochain`, `hc`) -- version compatible with HDK 0.6 / HDI 0.7
- `libclang-dev` (needed for datachannel-sys compilation)
- Set `LIBCLANG_PATH` if libclang is in a non-standard location

## Test Categories

### 1. Unit tests

`cargo test --workspace` -- pure Rust, no conductor needed. Tests type validation,
math functions, and integrity zome validation rules. Always works on any machine with
a Rust toolchain.

### 2. Stress tests

`cargo test --test economics_stress_test` -- pure Rust property/stress tests for
economic formulas. No conductor needed. Exercises demurrage convergence, fee
monotonicity, TEND zero-sum invariant, reserve ratio maintenance, and rate limit
enforcement across thousands of iterations.

### 3. Integration tests (sweettest)

`cargo test --test <name> -- --include-ignored` -- require a Holochain conductor.
Uses the sweettest framework to spin up an in-process conductor per test. Tests full
zome call chains including cross-zome interactions. Must run with `--test-threads=1`
because the conductor is single-threaded.

## Running Tests

```bash
# Pure tests (always works)
cargo test --workspace
cargo test --test economics_stress_test

# Full integration suite (requires Holochain)
./scripts/run-integration-tests.sh

# Single integration test file
cargo test --test bridge_test -- --include-ignored --test-threads=1
```

## Test Files

| File | Covers |
|------|--------|
| `sweettest_integration.rs` | End-to-end 3-currency model (MYCEL, SAP, TEND) across all zomes |
| `bridge_test.rs` | Collateral bridge deposits, redemption, rate limiting, DID validation |
| `tend_test.rs` | TEND mutual credit exchange, balance limits, disputes, quality ratings |
| `payments_test.rs` | Payment creation, double-spend prevention, fee calculation, demurrage |
| `treasury_test.rs` | Treasury lifecycle, allocation governance, commons pool, savings pools |
| `staking_test.rs` | Collateral staking, MYCEL-weighted stakes, slashing, crypto escrow |
| `recognition_test.rs` | MYCEL reputation, apprentice onboarding, passive decay, jubilee normalization |
| `economics_stress_test.rs` | Property-based stress tests for economic formulas (no conductor) |

## Constitutional Invariants Verified

- **Demurrage exempt floor**: small SAP balances below the floor are never taxed
- **Inalienable reserve (25%)**: commons pool always retains at least 25% of total SAP
- **TEND zero-sum**: every time exchange credits the provider exactly what it debits the receiver
- **Fee proportionality**: transaction fees scale with amount and MYCEL tier
- **Reserve ratio maintenance**: treasury allocations cannot breach the minimum reserve ratio
- **Rate limiting**: bridge withdrawals cannot exceed 5% of vault value per day per member
- **Balance limits**: TEND balances stay within +/-40 (adjusted by tier)
