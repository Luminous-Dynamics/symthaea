/-!
# PITB Lean counterfactual pilot seed

These examples calibrate the formal evidence path; they are not model-evaluation
results. A passing build establishes that Lean checks these declarations in this
pinned environment. It does not establish general transfer utility.

The four relevant distinctions are:
* alpha-renamed binders can yield definitionally equal proposition terms;
* different proposition forms can be propositionally equivalent with an explicit
  checked equivalence proof;
* removing a needed hypothesis can make the resulting universal statement false; and
* a legal local-context intervention can preserve the goal while requiring the solver to choose an alternate route.

No `sorry`, custom axioms, `native_decide`, Mathlib dependency, or external tactic
is used in this file.
-/

namespace PITBCounterfactualPilot

open Lean Elab Tactic Meta in
elab "pitb_snapshot " label:str : tactic => withMainContext do
  let goal ← getMainGoal
  let goalType ← goal.getType
  let context ← Lean.MonadLCtx.getLCtx
  let goalText ← ppExpr goalType
  let mut structuredDeclarations : Array Json := #[]
  Lean.logInfo m!"PITB_STATE_BEGIN:{label.getString}"
  Lean.logInfo m!"PITB_STATE_GOAL:{goalText}"
  for declaration in context do
    let declarationType ← ppExpr declaration.type
    Lean.logInfo m!"PITB_STATE_DECL:{declaration.userName}:{declarationType}"
    structuredDeclarations := structuredDeclarations.push <| Json.mkObj [
      ("index", toJson declaration.index),
      ("user_name", toJson declaration.userName.toString),
      ("type", toJson declarationType.pretty)
    ]
  let snapshot := Json.mkObj [
    ("schema_version", toJson "pitb.lean-proof-state-snapshot.v0"),
    ("label", toJson label.getString),
    ("goal", toJson goalText.pretty),
    ("declarations", Json.arr structuredDeclarations)
  ]
  Lean.logInfo m!"PITB_STATE_JSON:{snapshot.compress}"
  Lean.logInfo m!"PITB_STATE_END:{label.getString}"

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

/--
A fixed-target proof-state intervention demonstration.

The theorem statement and goal type stay fixed across the two structured `pitb_snapshot` traces. The first state contains `h : a ≤ b`; after `clear h`, the exact
same goal remains but that named local declaration is absent. The remaining
equality hypothesis supplies an alternate route to close the same target.
This is a calibration example, not a model decision or a replayed benchmark row.
-/
theorem same_goal_after_hypothesis_clear
    (a b : Nat) (h : a ≤ b) (hEq : a = b) :
    Nat.succ a ≤ Nat.succ b := by
  pitb_snapshot "base"
  clear h
  pitb_snapshot "variant"
  subst b
  exact Nat.le_refl _

/-- Removing the hypothesis changes the contract and produces a false target. -/
theorem premise_removed_variant_is_false :
    ¬ PremiseRemovedVariant := by
  intro universal
  change (∀ a b : Nat, Nat.succ a ≤ Nat.succ b) at universal
  have h21 : Nat.succ 1 ≤ Nat.succ 0 := universal 1 0
  have h10 : 1 ≤ 0 := Nat.succ_le_succ_iff.mp h21
  have hEq : (1 : Nat) = 0 := Nat.le_antisymm h10 (Nat.zero_le 1)
  exact Nat.succ_ne_zero 0 hEq


/--
Removing an irrelevant local declaration must preserve the same target and
leave the required order hypothesis available. This is the positive control
for context invariance: unlike clearing a required premise, it must not force
the original move to be abandoned.
-/
theorem same_goal_after_irrelevant_hypothesis_clear
    (a b : Nat) (h : a ≤ b) (hIrrelevant : 17 = 17) :
    Nat.succ a ≤ Nat.succ b := by
  pitb_snapshot "irrelevant-base"
  clear hIrrelevant
  pitb_snapshot "irrelevant-variant"
  exact Nat.succ_le_succ h

end PITBCounterfactualPilot
