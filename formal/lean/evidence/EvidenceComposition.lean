namespace Symthaea.Formal.Evidence

/-- Canonical formal-evidence planes. These constructors are intentionally
    distinct rather than ordered by a synthetic scalar strength score. -/
inductive EvidenceClass where
  | abstractFormalTheorem
  | extractedSourceRefinement
  | deductiveImplementationProof
  | boundedModelSafety
  | temporalModelEvidence
  | boundedTraceConformance
  | runtimeQualification
  deriving Repr, DecidableEq

/-- A small semantic receipt node. The type parameters keep subject, claim
    scope, and assumptions opaque to the composition algebra. -/
structure Receipt (Subject Scope Assumption : Type) where
  primary : EvidenceClass
  subject : Subject
  scope : Scope
  assumptions : List Assumption

/-- Compose one dependency into a root receipt.

    Composition accumulates assumptions but does not rewrite the root's
    primary evidence class, subject identity, or claim scope. -/
def composeOne
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption) :
    Receipt Subject Scope Assumption :=
  {
    primary := root.primary
    subject := root.subject
    scope := root.scope
    assumptions := root.assumptions ++ dependency.assumptions
  }

/-- Compose an arbitrary finite dependency list into a root receipt. -/
def composeMany
    {Subject Scope Assumption : Type}
    (root : Receipt Subject Scope Assumption) :
    List (Receipt Subject Scope Assumption) → Receipt Subject Scope Assumption
  | [] => root
  | dependency :: rest => composeMany (composeOne root dependency) rest

theorem compose_one_preserves_primary_class
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption) :
    (composeOne root dependency).primary = root.primary := by
  rfl

theorem compose_one_preserves_subject
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption) :
    (composeOne root dependency).subject = root.subject := by
  rfl

theorem compose_one_preserves_scope
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption) :
    (composeOne root dependency).scope = root.scope := by
  rfl

theorem compose_one_preserves_root_assumptions
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption)
    {assumption : Assumption}
    (h : assumption ∈ root.assumptions) :
    assumption ∈ (composeOne root dependency).assumptions := by
  simp [composeOne, h]

theorem compose_one_preserves_dependency_assumptions
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption)
    {assumption : Assumption}
    (h : assumption ∈ dependency.assumptions) :
    assumption ∈ (composeOne root dependency).assumptions := by
  simp [composeOne, h]

theorem compose_many_preserves_primary_class
    {Subject Scope Assumption : Type}
    (root : Receipt Subject Scope Assumption)
    (dependencies : List (Receipt Subject Scope Assumption)) :
    (composeMany root dependencies).primary = root.primary := by
  induction dependencies generalizing root with
  | nil => rfl
  | cons dependency rest ih =>
      simpa [composeMany, composeOne] using
        (ih (root := composeOne root dependency))

theorem compose_many_preserves_subject
    {Subject Scope Assumption : Type}
    (root : Receipt Subject Scope Assumption)
    (dependencies : List (Receipt Subject Scope Assumption)) :
    (composeMany root dependencies).subject = root.subject := by
  induction dependencies generalizing root with
  | nil => rfl
  | cons dependency rest ih =>
      simpa [composeMany, composeOne] using
        (ih (root := composeOne root dependency))

theorem compose_many_preserves_scope
    {Subject Scope Assumption : Type}
    (root : Receipt Subject Scope Assumption)
    (dependencies : List (Receipt Subject Scope Assumption)) :
    (composeMany root dependencies).scope = root.scope := by
  induction dependencies generalizing root with
  | nil => rfl
  | cons dependency rest ih =>
      simpa [composeMany, composeOne] using
        (ih (root := composeOne root dependency))

theorem compose_many_preserves_root_assumptions
    {Subject Scope Assumption : Type}
    (root : Receipt Subject Scope Assumption)
    (dependencies : List (Receipt Subject Scope Assumption))
    {assumption : Assumption}
    (h : assumption ∈ root.assumptions) :
    assumption ∈ (composeMany root dependencies).assumptions := by
  induction dependencies generalizing root with
  | nil => simpa [composeMany] using h
  | cons dependency rest ih =>
      apply ih (root := composeOne root dependency)
      exact compose_one_preserves_root_assumptions root dependency h

theorem compose_many_preserves_head_dependency_assumptions
    {Subject Scope Assumption : Type}
    (root dependency : Receipt Subject Scope Assumption)
    (rest : List (Receipt Subject Scope Assumption))
    {assumption : Assumption}
    (h : assumption ∈ dependency.assumptions) :
    assumption ∈ (composeMany root (dependency :: rest)).assumptions := by
  apply compose_many_preserves_root_assumptions
    (root := composeOne root dependency)
    (dependencies := rest)
  exact compose_one_preserves_dependency_assumptions root dependency h

#print axioms compose_one_preserves_primary_class
#print axioms compose_one_preserves_subject
#print axioms compose_one_preserves_scope
#print axioms compose_one_preserves_root_assumptions
#print axioms compose_one_preserves_dependency_assumptions
#print axioms compose_many_preserves_primary_class
#print axioms compose_many_preserves_subject
#print axioms compose_many_preserves_scope
#print axioms compose_many_preserves_root_assumptions
#print axioms compose_many_preserves_head_dependency_assumptions

end Symthaea.Formal.Evidence
