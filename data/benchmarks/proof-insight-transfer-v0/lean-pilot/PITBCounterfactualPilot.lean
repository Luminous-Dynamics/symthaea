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

open Lean Elab Tactic Meta
partial def collectExprFVars : Expr → FVarIdSet
  | .fvar id => ({id} : FVarIdSet)
  | .app fn arg => (collectExprFVars fn).union (collectExprFVars arg)
  | .lam _ type body _ => (collectExprFVars type).union (collectExprFVars body)
  | .forallE _ type body _ => (collectExprFVars type).union (collectExprFVars body)
  | .letE _ type value body _ =>
      (collectExprFVars type).union ((collectExprFVars value).union (collectExprFVars body))
  | .mdata _ expr => collectExprFVars expr
  | .proj _ _ expr => collectExprFVars expr
  | _ => {}

def mergeExprConstNames (left right : Array Name) : Array Name :=
  right.foldl (init := left) fun accumulated name =>
    if accumulated.contains name then accumulated else accumulated.push name

partial def collectExprConstNames : Expr → Array Name
  | .const name _ => #[name]
  | .app fn arg => mergeExprConstNames (collectExprConstNames fn) (collectExprConstNames arg)
  | .lam _ type body _ => mergeExprConstNames (collectExprConstNames type) (collectExprConstNames body)
  | .forallE _ type body _ => mergeExprConstNames (collectExprConstNames type) (collectExprConstNames body)
  | .letE _ type value body _ =>
      mergeExprConstNames (collectExprConstNames type)
        (mergeExprConstNames (collectExprConstNames value) (collectExprConstNames body))
  | .mdata _ expr => collectExprConstNames expr
  | .proj _ _ expr => collectExprConstNames expr
  | _ => #[]

/--
Typed local-support closure for an already elaborated proof term.
Follow declaration types and ordinary visible let-values only. Nondependent
have-bound values are opaque for context reconstruction and are recorded by
the separate ContextEvidence exporter as assigned-value provenance.
-/
partial def closeLocalSupport (lctx : LocalContext) (frontier visited : FVarIdSet) : FVarIdSet := Id.run do
  if frontier.isEmpty then
    return visited
  let mut next : FVarIdSet := {}
  let mut visited := visited
  for id in frontier do
    if !visited.contains id then
      visited := visited.insert id
      if let some declaration := lctx.find? id then
        next := next.union (collectExprFVars declaration.type)
        if let some value := declaration.value? false then
          next := next.union (collectExprFVars value)
  closeLocalSupport lctx next visited

def supportDeclarationJson (lctx : LocalContext) (id : FVarId) : MetaM Json := do
  let some declaration := lctx.find? id
    | throwError "PITB support receipt references a free variable outside the captured local context: {id}"
  let typeText ← ppExpr declaration.type
  return Json.mkObj [
    ("fvar_id", toJson id.name.toString),
    ("declaration_index", toJson declaration.index),
    ("user_name", toJson declaration.userName.toString),
    ("type", toJson typeText.pretty),
    ("is_let_declaration", toJson (declaration.isLet true))
  ]

syntax "pitb_prove_and_record_support " str " => " tacticSeq : tactic

elab_rules : tactic
  | `(tactic| pitb_prove_and_record_support $label:str => $sequence:tacticSeq) => withMainContext do
      let goal ← getMainGoal
      let declaration ← goal.getDecl
      evalTactic sequence
      let some proof ← getExprMVarAssignment? goal
        | throwError "PITB support receipt could not retrieve the assigned proof term"
      let proof ← instantiateMVars proof
      if proof.hasExprMVar then
        throwError "PITB support receipt refuses a proof term with unresolved metavariables"
      let direct := collectExprFVars proof
      let transitive := closeLocalSupport declaration.lctx direct {}
      let directConstants := collectExprConstNames proof
      let mut directRows : Array Json := #[]
      let mut transitiveRows : Array Json := #[]
      for id in direct do
        directRows := directRows.push (← supportDeclarationJson declaration.lctx id)
      for id in transitive do
        transitiveRows := transitiveRows.push (← supportDeclarationJson declaration.lctx id)
      let proofText ← ppExpr proof
      let receipt := Json.mkObj [
        ("schema_version", toJson "pitb.proof-term-support.v0"),
        ("label", toJson label.getString),
        ("proof_term", toJson proofText.pretty),
        ("direct_global_constants", Json.arr (directConstants.map fun name => toJson name.toString)),
        ("direct_free_variable_support", Json.arr directRows),
        ("transitive_local_dependency_closure", Json.arr transitiveRows)
      ]
      Lean.logInfo m!"PITB_PROOF_SUPPORT_JSON:{receipt.compress}"

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
  pitb_prove_and_record_support "alternate-route-after-required-hypothesis-withdrawal" =>
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
  pitb_prove_and_record_support "original-route-after-irrelevant-context-removal" =>
    exact Nat.succ_le_succ h

end PITBCounterfactualPilot
