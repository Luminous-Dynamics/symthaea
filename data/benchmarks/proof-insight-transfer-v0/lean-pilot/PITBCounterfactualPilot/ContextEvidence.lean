import PITBCounterfactualPilot

open Lean Elab Tactic Meta

namespace PITBCounterfactualPilot.ContextEvidence

open PITBCounterfactualPilot

/-- Include projection structure names as global constant references. -/
partial def collectConstantsComplete : Expr → Array Name
  | .const name _ => #[name]
  | .app fn arg => mergeExprConstNames (collectConstantsComplete fn) (collectConstantsComplete arg)
  | .lam _ type body _ => mergeExprConstNames (collectConstantsComplete type) (collectConstantsComplete body)
  | .forallE _ type body _ => mergeExprConstNames (collectConstantsComplete type) (collectConstantsComplete body)
  | .letE _ type value body _ =>
      mergeExprConstNames (collectConstantsComplete type)
        (mergeExprConstNames (collectConstantsComplete value) (collectConstantsComplete body))
  | .mdata _ expr => collectConstantsComplete expr
  | .proj typeName _ expr => mergeExprConstNames #[typeName] (collectConstantsComplete expr)
  | _ => #[]

def fvarNamesJson (ids : FVarIdSet) : Json := Id.run do
  let mut names : Array Json := #[]
  for id in ids do
    names := names.push (toJson id.name.toString)
  return Json.arr names

def constantNamesJson (names : Array Name) : Json :=
  Json.arr (names.map fun name => toJson name.toString)

def optionalTextJson (value : Option String) : Json :=
  match value with
  | some text => toJson text
  | none => Json.null

/--
Typing dependencies follow declaration types and visible transparent let values.
Nondependent have-bound values are deliberately excluded from this closure.
-/
partial def closeTypedDependencies
    (lctx : LocalContext) (frontier visited : FVarIdSet) : MetaM FVarIdSet := do
  if frontier.isEmpty then
    return visited
  let mut next : FVarIdSet := {}
  let mut visited := visited
  for id in frontier do
    if !visited.contains id then
      let some declaration := lctx.find? id
        | throwError "PITB typed dependency references a free variable outside the captured context: {id}"
      if declaration.hasExprMVar then
        throwError "PITB typed dependency refuses unresolved metavariables in {declaration.userName}"
      visited := visited.insert id
      next := next.union (collectExprFVars declaration.type)
      if let some value := declaration.value? false then
        next := next.union (collectExprFVars value)
  closeTypedDependencies lctx next visited

/--
Assigned-value provenance follows every assigned value, including nondependent
have-bound values. This relation is provenance only, not a context-typing gate.
-/
partial def closeAssignedValueProvenance
    (lctx : LocalContext) (frontier visited : FVarIdSet) : MetaM FVarIdSet := do
  if frontier.isEmpty then
    return visited
  let mut next : FVarIdSet := {}
  let mut visited := visited
  for id in frontier do
    if !visited.contains id then
      let some declaration := lctx.find? id
        | throwError "PITB value provenance references a free variable outside the captured context: {id}"
      if declaration.hasExprMVar then
        throwError "PITB value provenance refuses unresolved metavariables in {declaration.userName}"
      visited := visited.insert id
      next := next.union (collectExprFVars declaration.type)
      if let some value := declaration.value? true then
        next := next.union (collectExprFVars value)
  closeAssignedValueProvenance lctx next visited

def localDeclarationJson (declaration : LocalDecl) : MetaM Json := do
  if declaration.hasExprMVar then
    throwError "PITB context export refuses unresolved metavariables in {declaration.userName}"
  let typeText ← ppExpr declaration.type
  let transparentValue := declaration.value? false
  let allValue := declaration.value? true
  let transparentValueText : Option String ←
    match transparentValue with
    | some value => do
        let rendered ← ppExpr value
        pure (some rendered.pretty)
    | none => pure none
  let nondepValueText : Option String ←
    if declaration.isNondep then
      match allValue with
      | some value => do
          let rendered ← ppExpr value
          pure (some rendered.pretty)
      | none => throwError "PITB nondependent declaration has no inspectable assigned value: {declaration.userName}"
    else
      pure none
  let transparentValueFVars := match transparentValue with
    | some value => collectExprFVars value
    | none => {}
  let nondepValueFVars := match nondepValueText, allValue with
    | some _, some value => collectExprFVars value
    | _, _ => {}
  let typeConstants := collectConstantsComplete declaration.type
  let transparentConstants := match transparentValue with
    | some value => collectConstantsComplete value
    | none => #[]
  let nondepConstants := match nondepValueText, allValue with
    | some _, some value => collectConstantsComplete value
    | _, _ => #[]
  return Json.mkObj [
    ("fvar_id", toJson declaration.fvarId.name.toString),
    ("index", toJson declaration.index),
    ("user_name", toJson declaration.userName.toString),
    ("type", toJson typeText.pretty),
    ("type_fvar_ids", fvarNamesJson (collectExprFVars declaration.type)),
    ("type_global_constants", constantNamesJson typeConstants),
    ("is_let_declaration", toJson (declaration.isLet true)),
    ("is_nondep", toJson declaration.isNondep),
    ("is_implementation_detail", toJson declaration.isImplementationDetail),
    ("transparent_value", optionalTextJson transparentValueText),
    ("transparent_value_fvar_ids", fvarNamesJson transparentValueFVars),
    ("transparent_value_global_constants", constantNamesJson transparentConstants),
    ("nondep_value_provenance", optionalTextJson nondepValueText),
    ("nondep_value_fvar_ids", fvarNamesJson nondepValueFVars),
    ("nondep_value_global_constants", constantNamesJson nondepConstants)
  ]

def localDeclarationRows (lctx : LocalContext) (ids : FVarIdSet) : MetaM (Array Json) := do
  let mut rows : Array Json := #[]
  for id in ids do
    let some declaration := lctx.find? id
      | throwError "PITB support closure references a free variable outside the captured context: {id}"
    rows := rows.push (← localDeclarationJson declaration)
  return rows

def constantsInLocalSet (lctx : LocalContext) (ids : FVarIdSet)
    (includeNondepValues : Bool) : MetaM (Array Name) := do
  let mut constants : Array Name := #[]
  for id in ids do
    let some declaration := lctx.find? id
      | throwError "PITB constant collection references a free variable outside the captured context: {id}"
    constants := mergeExprConstNames constants (collectConstantsComplete declaration.type)
    if let some value := declaration.value? includeNondepValues then
      constants := mergeExprConstNames constants (collectConstantsComplete value)
  return constants

def opaqueValueProvenanceRows (lctx : LocalContext) (ids : FVarIdSet) : MetaM (Array Json) := do
  let mut rows : Array Json := #[]
  for id in ids do
    let some declaration := lctx.find? id
      | throwError "PITB opaque provenance references a free variable outside the captured context: {id}"
    if declaration.isNondep then
      let some value := declaration.value? true
        | throwError "PITB nondependent declaration has no assigned value: {declaration.userName}"
      rows := rows.push <| Json.mkObj [
        ("declaration_fvar_id", toJson declaration.fvarId.name.toString),
        ("declaration_user_name", toJson declaration.userName.toString),
        ("assigned_value_fvar_ids", fvarNamesJson (collectExprFVars value)),
        ("assigned_value_global_constants", constantNamesJson (collectConstantsComplete value))
      ]
  return rows

syntax "pitb_prove_and_export_context " str " => " tacticSeq : tactic

elab_rules : tactic
  | `(tactic| pitb_prove_and_export_context $label:str => $sequence:tacticSeq) => withMainContext do
      let goal ← getMainGoal
      let goalDeclaration ← goal.getDecl
      let lctx := goalDeclaration.lctx
      let target ← instantiateMVars (← goal.getType)
      if target.hasExprMVar then
        throwError "PITB context exporter refuses a target with unresolved metavariables"
      let targetText ← ppExpr target
      let mut contextRows : Array Json := #[]
      let mut contextOrder : Array Json := #[]
      for declaration in lctx do
        contextRows := contextRows.push (← localDeclarationJson declaration)
        contextOrder := contextOrder.push (toJson declaration.fvarId.name.toString)
      let targetFVars := collectExprFVars target
      let targetTypedClosure ← closeTypedDependencies lctx targetFVars {}
      let targetTypedRows ← localDeclarationRows lctx targetTypedClosure
      evalTactic sequence
      let some assignedProof ← getExprMVarAssignment? goal
        | throwError "PITB context exporter could not retrieve the assigned proof term"
      let proof ← instantiateMVars assignedProof
      if proof.hasExprMVar then
        throwError "PITB context exporter refuses a proof term with unresolved metavariables"
      let proofText ← ppExpr proof
      let directFVars := collectExprFVars proof
      let typedClosure ← closeTypedDependencies lctx directFVars {}
      let assignedClosure ← closeAssignedValueProvenance lctx directFVars {}
      let directRows ← localDeclarationRows lctx directFVars
      let typedRows ← localDeclarationRows lctx typedClosure
      let assignedRows ← localDeclarationRows lctx assignedClosure
      let typedConstants ← constantsInLocalSet lctx typedClosure false
      let assignedConstants ← constantsInLocalSet lctx assignedClosure true
      let opaqueRows ← opaqueValueProvenanceRows lctx typedClosure
      let evidence := Json.mkObj [
        ("schema_version", toJson "pitb.local-context-evidence.v1"),
        ("label", toJson label.getString),
        ("target", Json.mkObj [
          ("type", toJson targetText.pretty),
          ("free_variable_ids", fvarNamesJson targetFVars),
          ("typed_dependency_closure_ids", fvarNamesJson targetTypedClosure),
          ("typed_dependency_closure", Json.arr targetTypedRows),
          ("global_constants", constantNamesJson (collectConstantsComplete target))
        ]),
        ("local_context", Json.mkObj [
          ("declaration_order", Json.arr contextOrder),
          ("declarations", Json.arr contextRows)
        ]),
        ("proof_support", Json.mkObj [
          ("proof_term", toJson proofText.pretty),
          ("direct_free_variable_ids", fvarNamesJson directFVars),
          ("direct_declarations", Json.arr directRows),
          ("typed_dependency_closure_ids", fvarNamesJson typedClosure),
          ("typed_dependency_closure", Json.arr typedRows),
          ("typed_dependency_global_constants", constantNamesJson typedConstants),
          ("assigned_value_provenance_closure_ids", fvarNamesJson assignedClosure),
          ("assigned_value_provenance_closure", Json.arr assignedRows),
          ("assigned_value_provenance_global_constants", constantNamesJson assignedConstants),
          ("opaque_value_provenance", Json.arr opaqueRows),
          ("proof_term_global_constants", constantNamesJson (collectConstantsComplete proof))
        ]),
        ("interpretation_boundary", toJson
          "Typed context dependencies follow declaration types and transparent let values. Nondependent have-bound values are reported separately as assigned-value provenance and must not be treated as typing prerequisites. Neither closure establishes logical necessity.")
      ]
      Lean.logInfo m!"PITB_CONTEXT_EVIDENCE_JSON:{evidence.compress}"

/--
Calibration: ordinary let values are transparent typing dependencies;
have-bound nondependent values are assigned-value provenance only.
-/
theorem transparent_let_vs_have_provenance (h : 0 = 0) : 0 = 0 := by
  let transparentValue : 0 = 0 := h
  have opaqueValue : 0 = 0 := transparentValue
  pitb_prove_and_export_context "transparent-let-have-separation" =>
    exact opaqueValue

end PITBCounterfactualPilot.ContextEvidence
