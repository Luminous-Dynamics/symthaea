-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-ZK-FV-001A
-- Application-level balance statement relation only.
-- This file does not model RISC Zero, receipt soundness, or zero knowledge.

namespace SymthaeaFormal.Zk

/-- Mathematical model of the complete Rust `u64` value domain. -/
abbrev U64 : Type := Fin (2 ^ 64)

/-- Private witness plus public threshold/nonce inputs. -/
structure BalanceInput where
  balance : U64
  requiredMinimum : U64
  nonce : U64
  deriving DecidableEq

/-- Public result committed by the balance statement. -/
structure BalanceOutput where
  sufficient : Bool
  requiredMinimum : U64
  nonce : U64
  deriving DecidableEq

/-- Pure mathematical statement kernel corresponding to the intended balance relation. -/
def balanceKernel (i : BalanceInput) : BalanceOutput :=
  {
    sufficient := decide (i.requiredMinimum ≤ i.balance)
    requiredMinimum := i.requiredMinimum
    nonce := i.nonce
  }

/-- The application relation: the public output is exactly the kernel result. -/
def BalanceRelation (i : BalanceInput) (o : BalanceOutput) : Prop :=
  o = balanceKernel i

/-- The canonical kernel output satisfies the relation by construction. -/
theorem relation_holds (i : BalanceInput) :
    BalanceRelation i (balanceKernel i) := by
  rfl

/-- The public sufficiency bit denotes exactly `requiredMinimum ≤ balance`. -/
theorem sufficient_eq_true_iff (i : BalanceInput) :
    (balanceKernel i).sufficient = true ↔ i.requiredMinimum ≤ i.balance := by
  simp [balanceKernel]

/-- A witness meeting the threshold produces `sufficient = true`. -/
theorem sufficient_true_of_ge (i : BalanceInput)
    (h : i.requiredMinimum ≤ i.balance) :
    (balanceKernel i).sufficient = true := by
  simp [balanceKernel, h]

/-- A witness below the threshold produces `sufficient = false`. -/
theorem sufficient_false_of_not_ge (i : BalanceInput)
    (h : ¬ i.requiredMinimum ≤ i.balance) :
    (balanceKernel i).sufficient = false := by
  simp [balanceKernel, h]

/-- The public threshold is preserved exactly. -/
theorem threshold_preserved (i : BalanceInput) :
    (balanceKernel i).requiredMinimum = i.requiredMinimum := by
  rfl

/-- The replay/context nonce is preserved exactly. -/
theorem nonce_preserved (i : BalanceInput) :
    (balanceKernel i).nonce = i.nonce := by
  rfl

/-- A fixed input has a unique public result under the relation. -/
theorem relation_deterministic (i : BalanceInput) (o₁ o₂ : BalanceOutput)
    (h₁ : BalanceRelation i o₁) (h₂ : BalanceRelation i o₂) : o₁ = o₂ := by
  calc
    o₁ = balanceKernel i := h₁
    _ = o₂ := h₂.symm

#print axioms relation_holds
#print axioms sufficient_eq_true_iff
#print axioms sufficient_true_of_ge
#print axioms sufficient_false_of_not_ge
#print axioms threshold_preserved
#print axioms nonce_preserved
#print axioms relation_deterministic

end SymthaeaFormal.Zk
