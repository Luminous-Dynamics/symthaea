namespace Symthaea.Formal.Evidence

/-- Exact qualification outcome recorded by a formal-evidence reference. -/
inductive QualificationResult where
  | pass
  | fail
  | blocked
  | environmentFailure
  deriving Repr, DecidableEq

/-- A receipt together with the exact generation/currentness identity observed
    when the reference was formed and the qualification result it carries. -/
structure EvidenceRef (Subject Scope Assumption : Type) where
  receipt : Receipt Subject Scope Assumption
  observedGeneration : Nat
  liveGeneration : Nat
  result : QualificationResult

/-- Local currentness is exact generation equality. This is intentionally not
    wall-clock freshness or distributed finality. -/
def IsCurrent
    {Subject Scope Assumption : Type}
    (ref : EvidenceRef Subject Scope Assumption) : Prop :=
  ref.observedGeneration = ref.liveGeneration

/-- Only an exact-current reference with an exact `Pass` result is admissible. -/
def Admissible
    {Subject Scope Assumption : Type}
    (ref : EvidenceRef Subject Scope Assumption) : Prop :=
  IsCurrent ref ∧ ref.result = QualificationResult.pass

/-- Every dependency in the finite list must be independently admissible. -/
def DependenciesAdmissible
    {Subject Scope Assumption : Type}
    (dependencies : List (EvidenceRef Subject Scope Assumption)) : Prop :=
  ∀ dependency, dependency ∈ dependencies → Admissible dependency

/-- Composite admission requires both the root and every dependency to be
    independently current and positively qualified. -/
def CompositeAdmissible
    {Subject Scope Assumption : Type}
    (root : EvidenceRef Subject Scope Assumption)
    (dependencies : List (EvidenceRef Subject Scope Assumption)) : Prop :=
  Admissible root ∧ DependenciesAdmissible dependencies

theorem stale_reference_not_admissible
    {Subject Scope Assumption : Type}
    (ref : EvidenceRef Subject Scope Assumption)
    (hStale : ref.observedGeneration ≠ ref.liveGeneration) :
    ¬ Admissible ref := by
  intro h
  exact hStale h.1

theorem fail_result_not_admissible
    {Subject Scope Assumption : Type}
    (ref : EvidenceRef Subject Scope Assumption)
    (hResult : ref.result = QualificationResult.fail) :
    ¬ Admissible ref := by
  intro h
  rw [hResult] at h
  exact QualificationResult.noConfusion h.2

theorem blocked_result_not_admissible
    {Subject Scope Assumption : Type}
    (ref : EvidenceRef Subject Scope Assumption)
    (hResult : ref.result = QualificationResult.blocked) :
    ¬ Admissible ref := by
  intro h
  rw [hResult] at h
  exact QualificationResult.noConfusion h.2

theorem environment_failure_not_admissible
    {Subject Scope Assumption : Type}
    (ref : EvidenceRef Subject Scope Assumption)
    (hResult : ref.result = QualificationResult.environmentFailure) :
    ¬ Admissible ref := by
  intro h
  rw [hResult] at h
  exact QualificationResult.noConfusion h.2

theorem advancing_live_generation_invalidates_prior_reference
    {Subject Scope Assumption : Type}
    (receipt : Receipt Subject Scope Assumption)
    (result : QualificationResult)
    (oldGeneration newGeneration : Nat)
    (hChanged : oldGeneration ≠ newGeneration) :
    ¬ Admissible {
      receipt := receipt
      observedGeneration := oldGeneration
      liveGeneration := newGeneration
      result := result
    } := by
  apply stale_reference_not_admissible
  exact hChanged

theorem composite_admission_implies_root_current_and_pass
    {Subject Scope Assumption : Type}
    (root : EvidenceRef Subject Scope Assumption)
    (dependencies : List (EvidenceRef Subject Scope Assumption))
    (h : CompositeAdmissible root dependencies) :
    IsCurrent root ∧ root.result = QualificationResult.pass := by
  exact h.1

theorem composite_admission_implies_dependency_current_and_pass
    {Subject Scope Assumption : Type}
    (root dependency : EvidenceRef Subject Scope Assumption)
    (dependencies : List (EvidenceRef Subject Scope Assumption))
    (hComposite : CompositeAdmissible root dependencies)
    (hMember : dependency ∈ dependencies) :
    IsCurrent dependency ∧ dependency.result = QualificationResult.pass := by
  exact hComposite.2 dependency hMember

theorem stale_dependency_blocks_composite
    {Subject Scope Assumption : Type}
    (root dependency : EvidenceRef Subject Scope Assumption)
    (dependencies : List (EvidenceRef Subject Scope Assumption))
    (hMember : dependency ∈ dependencies)
    (hStale : dependency.observedGeneration ≠ dependency.liveGeneration) :
    ¬ CompositeAdmissible root dependencies := by
  intro hComposite
  have hDependency := hComposite.2 dependency hMember
  exact hStale hDependency.1

theorem nonpass_dependency_blocks_composite
    {Subject Scope Assumption : Type}
    (root dependency : EvidenceRef Subject Scope Assumption)
    (dependencies : List (EvidenceRef Subject Scope Assumption))
    (hMember : dependency ∈ dependencies)
    (hNonPass : dependency.result ≠ QualificationResult.pass) :
    ¬ CompositeAdmissible root dependencies := by
  intro hComposite
  have hDependency := hComposite.2 dependency hMember
  exact hNonPass hDependency.2

theorem adding_dependency_requires_that_dependency_admissible
    {Subject Scope Assumption : Type}
    (root dependency : EvidenceRef Subject Scope Assumption)
    (rest : List (EvidenceRef Subject Scope Assumption))
    (h : CompositeAdmissible root (dependency :: rest)) :
    Admissible dependency := by
  exact h.2 dependency (by simp)

theorem adding_admissible_dependency_preserves_composite
    {Subject Scope Assumption : Type}
    (root dependency : EvidenceRef Subject Scope Assumption)
    (rest : List (EvidenceRef Subject Scope Assumption))
    (hComposite : CompositeAdmissible root rest)
    (hDependency : Admissible dependency) :
    CompositeAdmissible root (dependency :: rest) := by
  constructor
  · exact hComposite.1
  · intro candidate hMember
    simp only [List.mem_cons] at hMember
    rcases hMember with rfl | hRest
    · exact hDependency
    · exact hComposite.2 candidate hRest

#print axioms stale_reference_not_admissible
#print axioms fail_result_not_admissible
#print axioms blocked_result_not_admissible
#print axioms environment_failure_not_admissible
#print axioms advancing_live_generation_invalidates_prior_reference
#print axioms composite_admission_implies_root_current_and_pass
#print axioms composite_admission_implies_dependency_current_and_pass
#print axioms stale_dependency_blocks_composite
#print axioms nonpass_dependency_blocks_composite
#print axioms adding_dependency_requires_that_dependency_admissible
#print axioms adding_admissible_dependency_preserves_composite

end Symthaea.Formal.Evidence
