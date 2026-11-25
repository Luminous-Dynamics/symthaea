import Lean
import A
import ProofActionCertificateAndComm

open Lean Elab Command

/-- Fresh dependency walk: deliberately does not consult exportedAxiomsExt or
    Lean.CollectAxioms' cross-declaration result cache. The visited set is only
    a cycle guard for this root; no per-constant partial results are memoized. -/
partial def freshAxiomDependencies
    (env : Environment)
    (pending : List Name)
    (visited : NameSet := {})
    (axioms : NameSet := {}) : NameSet :=
  match pending with
  | [] => axioms
  | name :: rest =>
    if visited.contains name then
      freshAxiomDependencies env rest visited axioms
    else
      let visited := visited.insert name
      match env.checked.get.find? name with
      | some (.axiomInfo info) =>
          let next := info.type.getUsedConstants.toList ++ rest
          freshAxiomDependencies env next visited (axioms.insert name)
      | some (.defnInfo info) =>
          let next := info.type.getUsedConstants.toList ++ info.value.getUsedConstants.toList ++ rest
          freshAxiomDependencies env next visited axioms
      | some (.thmInfo info) =>
          let next := info.type.getUsedConstants.toList ++ info.value.getUsedConstants.toList ++ rest
          freshAxiomDependencies env next visited axioms
      | some (.opaqueInfo info) =>
          let next := info.type.getUsedConstants.toList ++ info.value.getUsedConstants.toList ++ rest
          freshAxiomDependencies env next visited axioms
      | some (.quotInfo _) =>
          freshAxiomDependencies env rest visited axioms
      | some (.ctorInfo info) =>
          let next := info.type.getUsedConstants.toList ++ rest
          freshAxiomDependencies env next visited axioms
      | some (.recInfo info) =>
          let next := info.type.getUsedConstants.toList ++ rest
          freshAxiomDependencies env next visited axioms
      | some (.inductInfo info) =>
          let next := info.type.getUsedConstants.toList ++ info.ctors.toList ++ rest
          freshAxiomDependencies env next visited axioms
      | none =>
          -- Preserve the unresolved name so the exact expected-set assertion fails closed.
          axioms.insert name

def freshAxiomDependenciesFor (env : Environment) (root : Name) : NameSet :=
  let privateEnv := env.setExporting false
  freshAxiomDependencies privateEnv [root]

def freshAxiomDependenciesText (dependencies : NameSet) : String :=
  let ordered := dependencies.toList.mergeSort Name.quickLt
  String.intercalate ", " (ordered.map toString)

syntax (name := auditFreshAxioms) "#audit_fresh_axioms " ident : command
syntax (name := auditFreshNoAxioms) "#audit_fresh_no_axioms " ident : command

elab_rules : command
  | `(#audit_fresh_axioms $declaration:ident) => do
      let root := declaration.getId
      let dependencies := freshAxiomDependenciesFor (← getEnv) root
      let rendered := freshAxiomDependenciesText dependencies
      logInfo m!"FRESH_AXIOM_AUDIT root={root} axioms=[{rendered}]"
      unless dependencies.toList == [`Classical.choice] do
        throwError m!"fresh audit did not produce the exact expected axiom set for {root}: {rendered}"
  | `(#audit_fresh_no_axioms $declaration:ident) => do
      let root := declaration.getId
      let dependencies := freshAxiomDependenciesFor (← getEnv) root
      let rendered := freshAxiomDependenciesText dependencies
      logInfo m!"FRESH_NO_AXIOMS_AUDIT root={root} axioms=[{rendered}]"
      unless dependencies.toList == [] do
        throwError m!"fresh audit found unexpected axiom dependencies for {root}: {rendered}"

theorem usesS9 (_ : S9) : True := trivial

-- Native output is diagnostic only; the fresh per-root walk above is the
-- acceptance oracle for this reproduction and the proof-certificate roots.
#print axioms S9
#print axioms S9.mk
#print axioms usesS9
#print axioms And.comm
#print axioms ProofPlanActionCertificatePilot.and_comm_split_iff
#print axioms ProofPlanActionCertificatePilot.and_comm_forward
#print axioms ProofPlanActionCertificatePilot.and_comm_reverse
#print axioms ProofPlanActionCertificatePilot.and_comm_plan_replay

#audit_fresh_axioms S9
#audit_fresh_axioms S9.mk
#audit_fresh_axioms usesS9
#audit_fresh_no_axioms And.comm
#audit_fresh_no_axioms ProofPlanActionCertificatePilot.and_comm_split_iff
#audit_fresh_no_axioms ProofPlanActionCertificatePilot.and_comm_forward
#audit_fresh_no_axioms ProofPlanActionCertificatePilot.and_comm_reverse
#audit_fresh_no_axioms ProofPlanActionCertificatePilot.and_comm_plan_replay
