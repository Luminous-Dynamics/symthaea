# Status: Maintenance-only

**As of 2026-07-02, mycelix-marketplace is frozen for new feature work.**

## Why

No cross-cluster consumers (0 files anywhere in the monorepo dispatch into
marketplace via `CallTargetCell::OtherRole`), and no product-side feature
momentum since April 2026 — recent activity has been security/quality fixes
(real arbitrator assignment + transaction-state auth checks, test coverage
20 → 288) and license/formatting chores, not new capability. See
`MYCELIX_TIER_3_4_PRODUCT_GATE.md` at the repo root for the full evidence
and reasoning behind this call.

This is a **freeze, not a retire**: the arbitration/escrow/dispute-resolution
domain logic here is real, tested, and correct as of this fix. It's kept in
the workspace rather than moved to `_retired/` because there's no concrete
reason to destroy working code — only no demonstrated demand to build on it
further right now.

## What this means

- **Accepted**: security fixes, bug fixes, dependency/toolchain bumps needed
  to keep it building alongside the rest of `mycelix-workspace/`.
- **Not accepted without a product decision first**: new features, new
  zomes, new bridge wiring, scope expansion.

## Reopening this

If a product owner steps forward with a concrete need (a consumer cluster,
a real deployment target, an external commitment), the freeze can be lifted
— revisit `MYCELIX_TIER_3_4_PRODUCT_GATE.md`'s "find an owner" option and
update this file.
