namespace Symthaea.Formal.Assurance

abbrev SubjectIdentity := Nat
abbrev ClaimIdentity := Nat
abbrev EvidenceIdentity := Nat
abbrev GenerationIdentity := Nat
abbrev Assumption := Nat
abbrev ProvenanceRoot := Nat
abbrev IndependenceBasis := Nat

inductive EvidenceClass where
  | AbstractProof
  | BoundedModel
  | RuntimeObservation
  | SourceAudit
  | RefinementProof
  | TemporalProof
  deriving DecidableEq, Repr

inductive PropertyKind where
  | Safety
  | Liveness
  | Other
  deriving DecidableEq, Repr

inductive EvidenceValidity where
  | Current
  | Superseded
  | Invalidated
  deriving DecidableEq, Repr

inductive CanonicalResult where
  | SemanticSuccess
  | SemanticCounterexample
  | ProofOrQualificationFailure
  | UnsupportedBoundary
  | InsufficientBound
  | ResourceExhaustion
  | MissingPrerequisite
  | StaleSubjectOrDependency
  | AmbiguousOrUnknownOutcome
  | UnclassifiedToolCrash
  | EnvironmentUnavailable
  | ToolInstallationFailure
  | RunnerInfrastructureFailure
  deriving DecidableEq, Repr

inductive OutcomeDisposition where
  | Pass
  | Fail
  | Blocked
  | EnvironmentFailure
  deriving DecidableEq, Repr

/-- Canonical result-to-disposition mapping shared with SYM-FV-INFRA-003B. -/
def disposition : CanonicalResult → OutcomeDisposition
  | .SemanticSuccess => .Pass
  | .SemanticCounterexample => .Fail
  | .ProofOrQualificationFailure => .Fail
  | .UnsupportedBoundary => .Blocked
  | .InsufficientBound => .Blocked
  | .ResourceExhaustion => .Blocked
  | .MissingPrerequisite => .Blocked
  | .StaleSubjectOrDependency => .Blocked
  | .AmbiguousOrUnknownOutcome => .Blocked
  | .UnclassifiedToolCrash => .Blocked
  | .EnvironmentUnavailable => .EnvironmentFailure
  | .ToolInstallationFailure => .EnvironmentFailure
  | .RunnerInfrastructureFailure => .EnvironmentFailure

structure EvidenceReceipt where
  evidenceId : EvidenceIdentity
  generation : GenerationIdentity
  subject : SubjectIdentity
  claim : ClaimIdentity
  evidenceClass : EvidenceClass
  propertyKind : PropertyKind
  result : CanonicalResult
  validity : EvidenceValidity
  assumptions : List Assumption
  provenanceRoot : ProvenanceRoot
  dependencies : List EvidenceIdentity
  deriving Repr

structure Obligation where
  generation : GenerationIdentity
  subject : SubjectIdentity
  claim : ClaimIdentity
  requiredClass : EvidenceClass
  propertyKind : PropertyKind
  assumptions : List Assumption
  mandatory : Bool
  requiresReachabilityWitness : Bool
  deriving Repr

/--
A typed witness permitting evidence to move from one exact subject/claim layer
into another. Assumptions may either remain explicit on the target or be
explicitly discharged by this witness.
-/
structure RefinementWitness where
  generation : GenerationIdentity
  sourceSubject : SubjectIdentity
  targetSubject : SubjectIdentity
  sourceClaim : ClaimIdentity
  targetClaim : ClaimIdentity
  dischargedAssumptions : List Assumption
  current : Bool
  deriving Repr

/-- Directional claim entailment. `strongClaim` may support `weakClaim`, not vice versa. -/
structure ClaimEntailmentWitness where
  generation : GenerationIdentity
  strongClaim : ClaimIdentity
  weakClaim : ClaimIdentity
  current : Bool
  deriving Repr

/--
Independence is stronger than distinct receipt IDs. The witness binds two exact
evidence identities, their provenance roots, and an explicit reviewed basis.
-/
structure IndependenceWitness where
  generation : GenerationIdentity
  leftEvidence : EvidenceIdentity
  rightEvidence : EvidenceIdentity
  leftProvenance : ProvenanceRoot
  rightProvenance : ProvenanceRoot
  basis : IndependenceBasis
  basisDeclared : Bool
  current : Bool
  deriving Repr

structure ReachabilityWitness where
  generation : GenerationIdentity
  subject : SubjectIdentity
  claim : ClaimIdentity
  current : Bool
  deriving Repr

/-- Every assumption used by the source remains explicit or is explicitly discharged. -/
def AssumptionsAllowed
    (source target discharged : List Assumption) : Prop :=
  ∀ a, a ∈ source → a ∈ target ∨ a ∈ discharged

/--
Subject transfer is exact by default. Crossing subject boundaries requires a
current typed refinement witness for this exact evidence generation whose claim
endpoints also match.
-/
def SubjectTransfer
    (refinements : List RefinementWitness)
    (e : EvidenceReceipt)
    (o : Obligation) : Prop :=
  (e.subject = o.subject ∧
    AssumptionsAllowed e.assumptions o.assumptions []) ∨
  ∃ w,
    w ∈ refinements ∧
    w.current = true ∧
    w.generation = o.generation ∧
    w.sourceSubject = e.subject ∧
    w.targetSubject = o.subject ∧
    w.sourceClaim = e.claim ∧
    w.targetClaim = o.claim ∧
    AssumptionsAllowed e.assumptions o.assumptions w.dischargedAssumptions

/-- Claim transfer is exact or explicitly narrowing through a directional witness. -/
def ClaimTransfer
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation) : Prop :=
  e.claim = o.claim ∨
  ∃ w,
    w ∈ entailments ∧
    w.current = true ∧
    w.generation = o.generation ∧
    w.strongClaim = e.claim ∧
    w.weakClaim = o.claim

/--
A receipt may positively support an obligation only if every structural binding
matches. Evidence-class and property-kind compatibility are deliberately exact
in this first kernel, so composition cannot manufacture a stronger class or
turn safety evidence into liveness evidence.
-/
def Supports
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation) : Prop :=
  e.validity = .Current ∧
  disposition e.result = .Pass ∧
  e.generation = o.generation ∧
  e.evidenceClass = o.requiredClass ∧
  e.propertyKind = o.propertyKind ∧
  SubjectTransfer refinements e o ∧
  ClaimTransfer entailments e o

/-- Positive reachability is first-class when the obligation requires it. -/
def ReachabilitySatisfied
    (witnesses : List ReachabilityWitness)
    (o : Obligation) : Prop :=
  o.requiresReachabilityWitness = false ∨
  ∃ w,
    w ∈ witnesses ∧
    w.current = true ∧
    w.generation = o.generation ∧
    w.subject = o.subject ∧
    w.claim = o.claim

/-- Semantically or procedurally conflicting current dispositions remain visible. -/
def Conflicting (left right : EvidenceReceipt) : Prop :=
  left.subject = right.subject ∧
  left.claim = right.claim ∧
  left.generation = right.generation ∧
  left.validity = .Current ∧
  right.validity = .Current ∧
  disposition left.result = .Pass ∧
  disposition right.result = .Fail

def ConflictFree (receipts : List EvidenceReceipt) : Prop :=
  ∀ left, left ∈ receipts →
    ∀ right, right ∈ receipts →
      ¬ Conflicting left right

/--
Dependency evidence is recursively closed. Every declared prerequisite must be
present as a current semantic Pass, strictly lower in the well-founded rank,
and itself dependency-closed.
-/
inductive EvidenceClosed
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt) :
    EvidenceReceipt → Prop where
  | intro (e : EvidenceReceipt)
      (current : e.validity = .Current)
      (passed : disposition e.result = .Pass)
      (dependencies :
        ∀ dependency,
          dependency ∈ e.dependencies →
          ∃ prerequisite,
            prerequisite ∈ receipts ∧
            prerequisite.evidenceId = dependency ∧
            rank prerequisite.evidenceId < rank e.evidenceId ∧
            EvidenceClosed rank receipts prerequisite) :
      EvidenceClosed rank receipts e

/--
Finite first-kernel closure: every mandatory obligation has current admissible
positive support, a recursively closed dependency tree, and any required
positive reachability witness, while the current evidence set is conflict-free.
-/
def Closed
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (obligations : List Obligation)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (reachability : List ReachabilityWitness) : Prop :=
  ConflictFree receipts ∧
  ∀ o, o ∈ obligations → o.mandatory = true →
    ∃ e,
      e ∈ receipts ∧
      Supports refinements entailments e o ∧
      EvidenceClosed rank receipts e ∧
      ReachabilitySatisfied reachability o

/--
An explicit independence claim requires distinct evidence identities, distinct
provenance roots, and a current declared independence basis for the same
closure generation.
-/
def IndependentSupport
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (independence : List IndependenceWitness)
    (left right : EvidenceReceipt)
    (leftObligation rightObligation : Obligation) : Prop :=
  Supports refinements entailments left leftObligation ∧
  Supports refinements entailments right rightObligation ∧
  left.evidenceId ≠ right.evidenceId ∧
  left.provenanceRoot ≠ right.provenanceRoot ∧
  ∃ w,
    w ∈ independence ∧
    w.current = true ∧
    w.generation = left.generation ∧
    w.generation = right.generation ∧
    w.basisDeclared = true ∧
    w.leftEvidence = left.evidenceId ∧
    w.rightEvidence = right.evidenceId ∧
    w.leftProvenance = left.provenanceRoot ∧
    w.rightProvenance = right.provenanceRoot

/--
A rank-decreasing dependency path gives an explicit finite anti-cycle witness
for the same order used by `EvidenceClosed`.
-/
inductive DependencyPath
    (rank : EvidenceIdentity → Nat) :
    EvidenceIdentity → EvidenceIdentity → Prop where
  | edge {dependent prerequisite} :
      rank prerequisite < rank dependent →
      DependencyPath rank dependent prerequisite
  | trans {a b c} :
      DependencyPath rank a b →
      DependencyPath rank b c →
      DependencyPath rank a c

theorem blocked_cannot_support
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hblocked : disposition e.result = .Blocked) :
    ¬ Supports refinements entailments e o := by
  intro hs
  have hpass : disposition e.result = .Pass := hs.2.1
  rw [hblocked] at hpass
  cases hpass

theorem environment_failure_cannot_support
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (henv : disposition e.result = .EnvironmentFailure) :
    ¬ Supports refinements entailments e o := by
  intro hs
  have hpass : disposition e.result = .Pass := hs.2.1
  rw [henv] at hpass
  cases hpass

theorem superseded_cannot_support
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hsuperseded : e.validity = .Superseded) :
    ¬ Supports refinements entailments e o := by
  intro hs
  have hcurrent : e.validity = .Current := hs.1
  rw [hsuperseded] at hcurrent
  cases hcurrent

theorem invalidated_cannot_support
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hinvalidated : e.validity = .Invalidated) :
    ¬ Supports refinements entailments e o := by
  intro hs
  have hcurrent : e.validity = .Current := hs.1
  rw [hinvalidated] at hcurrent
  cases hcurrent

theorem stale_generation_cannot_support
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hstale : e.generation ≠ o.generation) :
    ¬ Supports refinements entailments e o := by
  intro hs
  exact hstale hs.2.2.1

theorem evidence_class_non_amplification
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hmismatch : e.evidenceClass ≠ o.requiredClass) :
    ¬ Supports refinements entailments e o := by
  intro hs
  exact hmismatch hs.2.2.2.1

theorem safety_cannot_close_liveness
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hsafety : e.propertyKind = .Safety)
    (hliveness : o.propertyKind = .Liveness) :
    ¬ Supports refinements entailments e o := by
  intro hs
  have hkind : e.propertyKind = o.propertyKind := hs.2.2.2.2.1
  rw [hsafety, hliveness] at hkind
  cases hkind

theorem mismatched_subject_requires_refinement
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hmismatch : e.subject ≠ o.subject)
    (hs : Supports refinements entailments e o) :
    ∃ w,
      w ∈ refinements ∧
      w.current = true ∧
      w.generation = o.generation ∧
      w.sourceSubject = e.subject ∧
      w.targetSubject = o.subject ∧
      w.sourceClaim = e.claim ∧
      w.targetClaim = o.claim ∧
      AssumptionsAllowed e.assumptions o.assumptions w.dischargedAssumptions := by
  have hsubject : SubjectTransfer refinements e o := hs.2.2.2.2.2.1
  cases hsubject with
  | inl hexact => exact False.elim (hmismatch hexact.1)
  | inr hrefinement => exact hrefinement

theorem mismatched_claim_requires_entailment
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hmismatch : e.claim ≠ o.claim)
    (hs : Supports refinements entailments e o) :
    ∃ w,
      w ∈ entailments ∧
      w.current = true ∧
      w.generation = o.generation ∧
      w.strongClaim = e.claim ∧
      w.weakClaim = o.claim := by
  have hclaim : ClaimTransfer entailments e o := hs.2.2.2.2.2.2
  cases hclaim with
  | inl hexact => exact False.elim (hmismatch hexact)
  | inr hentails => exact hentails

theorem exact_subject_support_preserves_assumptions
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hs : Supports [] entailments e o) :
    AssumptionsAllowed e.assumptions o.assumptions [] := by
  have hsubject : SubjectTransfer [] e o := hs.2.2.2.2.2.1
  cases hsubject with
  | inl hexact => exact hexact.2
  | inr hrefinement =>
      cases hrefinement with
      | intro w hw =>
          have hmember : w ∈ ([] : List RefinementWitness) := hw.1
          cases hmember

theorem same_provenance_cannot_count_as_independent
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (independence : List IndependenceWitness)
    (left right : EvidenceReceipt)
    (leftObligation rightObligation : Obligation)
    (hsame : left.provenanceRoot = right.provenanceRoot) :
    ¬ IndependentSupport refinements entailments independence
        left right leftObligation rightObligation := by
  intro hs
  exact hs.2.2.2.1 hsame

theorem required_reachability_empty_rejected
    (o : Obligation)
    (hrequired : o.requiresReachabilityWitness = true) :
    ¬ ReachabilitySatisfied [] o := by
  intro hs
  cases hs with
  | inl hdisabled =>
      rw [hrequired] at hdisabled
      cases hdisabled
  | inr hwitness =>
      cases hwitness with
      | intro w hw =>
          have hmember : w ∈ ([] : List ReachabilityWitness) := hw.1
          cases hmember

theorem evidence_closed_is_current_pass
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (e : EvidenceReceipt)
    (hclosed : EvidenceClosed rank receipts e) :
    e.validity = .Current ∧ disposition e.result = .Pass := by
  cases hclosed with
  | intro _ current passed _ => exact ⟨current, passed⟩

theorem evidence_closed_rejects_self_dependency
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (e : EvidenceReceipt)
    (hclosed : EvidenceClosed rank receipts e)
    (hself : e.evidenceId ∈ e.dependencies) : False := by
  cases hclosed with
  | intro _ _ _ dependencies =>
      have hexists := dependencies e.evidenceId hself
      cases hexists with
      | intro prerequisite hrest =>
          have hid : prerequisite.evidenceId = e.evidenceId := hrest.2.1
          have hlt : rank prerequisite.evidenceId < rank e.evidenceId := hrest.2.2.1
          rw [hid] at hlt
          exact (Nat.lt_irrefl (rank e.evidenceId)) hlt

theorem closed_requires_every_mandatory_obligation
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (obligations : List Obligation)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (reachability : List ReachabilityWitness)
    (hclosed : Closed rank receipts obligations refinements entailments reachability)
    (o : Obligation)
    (hmember : o ∈ obligations)
    (hmandatory : o.mandatory = true) :
    ∃ e,
      e ∈ receipts ∧
      Supports refinements entailments e o ∧
      EvidenceClosed rank receipts e ∧
      ReachabilitySatisfied reachability o := by
  exact hclosed.2 o hmember hmandatory

theorem contradictory_current_evidence_prevents_closure
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (obligations : List Obligation)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (reachability : List ReachabilityWitness)
    (left right : EvidenceReceipt)
    (hleft : left ∈ receipts)
    (hright : right ∈ receipts)
    (hconflict : Conflicting left right) :
    ¬ Closed rank receipts obligations refinements entailments reachability := by
  intro hclosed
  exact (hclosed.1 left hleft right hright) hconflict

theorem dependency_path_decreases
    (rank : EvidenceIdentity → Nat)
    {from to : EvidenceIdentity}
    (hpath : DependencyPath rank from to) :
    rank to < rank from := by
  induction hpath with
  | edge hlt => exact hlt
  | trans hab hbc ihab ihbc => exact Nat.lt_trans ihbc ihab

theorem dependency_cycles_rejected
    (rank : EvidenceIdentity → Nat)
    (evidence : EvidenceIdentity) :
    ¬ DependencyPath rank evidence evidence := by
  intro hcycle
  have hlt : rank evidence < rank evidence := dependency_path_decreases rank hcycle
  exact (Nat.lt_irrefl (rank evidence)) hlt

theorem support_non_amplification_summary
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (e : EvidenceReceipt)
    (o : Obligation)
    (hs : Supports refinements entailments e o) :
    e.validity = .Current ∧
    disposition e.result = .Pass ∧
    e.generation = o.generation ∧
    e.evidenceClass = o.requiredClass ∧
    e.propertyKind = o.propertyKind ∧
    SubjectTransfer refinements e o ∧
    ClaimTransfer entailments e o := by
  exact hs

#print axioms blocked_cannot_support
#print axioms environment_failure_cannot_support
#print axioms superseded_cannot_support
#print axioms invalidated_cannot_support
#print axioms stale_generation_cannot_support
#print axioms evidence_class_non_amplification
#print axioms safety_cannot_close_liveness
#print axioms mismatched_subject_requires_refinement
#print axioms mismatched_claim_requires_entailment
#print axioms exact_subject_support_preserves_assumptions
#print axioms same_provenance_cannot_count_as_independent
#print axioms required_reachability_empty_rejected
#print axioms evidence_closed_is_current_pass
#print axioms evidence_closed_rejects_self_dependency
#print axioms closed_requires_every_mandatory_obligation
#print axioms contradictory_current_evidence_prevents_closure
#print axioms dependency_path_decreases
#print axioms dependency_cycles_rejected
#print axioms support_non_amplification_summary

end Symthaea.Formal.Assurance
