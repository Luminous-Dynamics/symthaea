import Lean

open Lean Elab Tactic Meta

elab "pitb_capture_local_instances " sequence:tacticSeq : tactic => withMainContext do
  let goal ← getMainGoal
  let declaration ← goal.getDecl
  let mut rows : Array Json := #[]
  let mut foundDecidableInstance := false
  for localInstance in declaration.localInstances do
    let .fvar fvarId := localInstance.fvar
      | throwError "PITB compatibility shim: local instance is not a free-variable expression"
    let some localDecl := declaration.lctx.find? fvarId
      | throwError "PITB compatibility shim: local instance is absent from the local context"
    let binderIsInstance := match localDecl.binderInfo with
      | BinderInfo.instImplicit => true
      | _ => false
    if localInstance.className.toString == "PITBLocalInstanceProbe" && binderIsInstance then
      foundDecidableInstance := true
    let typeText ← ppExpr localDecl.type
    rows := rows.push <| Json.mkObj [
      ("class_name", toJson localInstance.className.toString),
      ("fvar_id", toJson fvarId.name.toString),
      ("binder_is_instance_implicit", toJson binderIsInstance),
      ("type", toJson typeText.pretty)
    ]
  if rows.size != 1 then
    throwError "PITB compatibility calibration expected one local instance, found {rows.size}"
  if !foundDecidableInstance then
    throwError "PITB compatibility calibration did not preserve PITBLocalInstanceProbe binder metadata"
  let payload := Json.mkObj [
    ("schema_version", toJson "pitb.local-instance-api-compat.v0"),
    ("local_instance_count", toJson rows.size),
    ("local_instances", Json.arr rows)
  ]
  logInfo m!"PITB_LOCAL_INSTANCE_COMPAT_PASS:{payload.compress}"
  evalTactic sequence

class PITBLocalInstanceProbe : Prop where
  witness : True

theorem pitb_local_instance_compatibility_calibration
    [localProbe : PITBLocalInstanceProbe] :
    PITBLocalInstanceProbe := by
  pitb_capture_local_instances =>
    exact (inferInstance : PITBLocalInstanceProbe)
