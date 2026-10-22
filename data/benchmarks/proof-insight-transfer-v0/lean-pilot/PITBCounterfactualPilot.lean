/-!
# PITB Lean counterfactual pilot seed

These examples calibrate the formal evidence path; they are not model-evaluation
results. A passing build establishes that Lean checks these declarations in this
pinned environment. It does not establish general transfer utility.

The three relevant distinctions are:
* alpha-renamed binders can yield definitionally equal proposition terms;
* different proposition forms can be propositionally equivalent with an explicit
  checked equivalence proof;
* removing a needed hypothesis can make the resulting universal statement false.

No `sorry`, custom axioms, `native_decide`, Mathlib dependency, or external tactic
is used in this file.
-/

namespace PITBCounterfactualPilot

/-- Binder-name reformulation: the propositions are definitionally equal. -/
theorem alpha_renaming_same_type :
    (∀ n : Nat, n = n) = (∀ x : Nat, x = x) := rfl

/-- Two different proposition forms, connected by an explicit proof of ↔. -/
theorem identity_statement_iff_true_function :
    (∀ n : Nat, n = n) ↔ (∀ n : Nat, True) :=
  Iff.intro
    (fun _ n => True.intro)
    (fun _ n => rfl)

/-- Exact base target, retaining the premise required by the strategy. -/
def SuccessorMonotonicityWithPremise : Prop :=
    ∀ a b : Nat, a ≤ b → Nat.succ a ≤ Nat.succ b

/-- Exact counterfactual target after the premise has been removed. -/
def PremiseRemovedVariant : Prop :=
    ∀ a b : Nat, Nat.succ a ≤ Nat.succ b

/-- The strategy works when its required order hypothesis is available. -/
theorem successor_monotonicity_with_premise :
    SuccessorMonotonicityWithPremise := by
  intro a b h
  exact Nat.succ_le_succ h

/-- Removing the hypothesis changes the contract and produces a false target. -/
theorem premise_removed_variant_is_false :
    ¬ PremiseRemovedVariant := by
  intro universal
  change (∀ a b : Nat, Nat.succ a ≤ Nat.succ b) at universal
  have h21 : Nat.succ 1 ≤ Nat.succ 0 := universal 1 0
  have h10 : 1 ≤ 0 := Nat.succ_le_succ_iff.mp h21
  have hEq : (1 : Nat) = 0 := Nat.le_antisymm h10 (Nat.zero_le 1)
  exact Nat.succ_ne_zero 0 hEq

end PITBCounterfactualPilot
