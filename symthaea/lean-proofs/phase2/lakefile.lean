-- Phase 2 Lake project: Mathlib-aware Lean verification for
-- symthaea-lean-bridge's arithmetic fixtures.
--
-- Purpose: resolve Mathlib imports so that `.lean` files emitted by
-- `prove_fol_arith` can be externally verified via
--
--     cd lean-proofs/phase2
--     lake env lean ../../proofs/fol_arith/<fixture>.lean
--
-- First-time setup (~10 min cold, downloads Mathlib cache):
--
--     cd lean-proofs/phase2
--     lake exe cache get      # fetch prebuilt Mathlib olean cache
--     lake build              # verify the cache resolves cleanly
--
-- Subsequent `lake env lean <file>` calls use the cached Mathlib
-- olean files and complete in seconds.

import Lake
open Lake DSL

package «symthaea-phase2» where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

-- Mathlib version pinned for reproducibility. Update via `lake update` when
-- bumping (typically quarterly to track Mathlib's lean-toolchain bumps).
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

@[default_target]
lean_lib «SymthaeaPhase2» where
  -- Placeholder library target. Individual proof files at
  -- ../../proofs/fol_arith/*.lean are verified via `lake env lean <file>`.
  roots := #[]
