# Release Ledger

## Symtropy Tier 1 Release Lane

### Verification Results
| Gate | Result |
| :--- | :--- |
| `cargo check --workspace` | PASS |
| `cargo test --workspace` | PASS |
| `cargo test --workspace --examples` | PASS |
| `cargo publish (math) --dry-run` | PASS |
| `cargo publish (physics) --dry-run` | PASS |
| `cargo publish (bevy-core) --dry-run` | PASS |

### Excluded Examples
- `old_waterworks_micro_slice.rs`
    - **Reason**: Depends on non-Tier-1 / unstable scene scaffolding.
    - **Action**: Archived inside release lane (`crates/symtropy-bevy-core/examples/excluded/`), not deleted.

### Sync-Back Patch List (To Workshop)
*Files successfully propagated from `releases/symtropy-release/` back to `/srv/luminous-dynamics/symtropy/`:*
1. `crates/symtropy-physics/src/joints/*.rs` (Updated `solve_velocity` trait signatures)
2. `crates/symtropy-physics/tests/callback_contract.rs` (Implemented `record_work` mock)
3. `crates/symtropy-bevy-core/src/lib.rs` (Updated coordinate indexing fix `.coord(n) -> [n]`)

### Status
- **Symtropy Tier 1**: RELEASE CANDIDATE (Verified Clean)

---

## Workspace Integrity Ticket: Symthaea Muse Path Integrity

**Goal**: Resolve the `symthaea-muse` workspace/path leak without destabilizing the ecosystem.

**Next Steps**:
1. Locate references: `rg "symthaea-muse|crates/domains/symthaea-muse|../symthaea/crates/symthaea-muse" /srv/luminous-dynamics -n`
2. Confirm actual crate location: `find /srv/luminous-dynamics -path "*symthaea-muse/Cargo.toml"`
3. Classify each reference as: Valid, Stale, Leak, or Obsolete.
4. Patch path mappings (not APIs).
5. Archive/deactivate stale references; do not delete.
EOF

---

## Technical Debt: Mycelix API Migration
- **Target**: 
- **Issue**: API drift between  and .
- **Status**: Deferred (Legacy access patterns restored to maintain stable build).
- **Required Follow-up**: Deliberate refactoring of  and  to map deprecated profile fields to  API contract.
