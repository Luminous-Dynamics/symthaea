namespace Symthaea.Formal.Authority

universe u v w

/--
A finite execution consisting only of transitions classified as non-promoting.

This is intentionally independent of any concrete capability, policy, restore,
or runtime representation. Concrete systems must separately refine their state
and transitions into this relation.
-/
inductive NonPromotingTrace {State : Type u} {Event : Type v}
    (step : State → Event → State → Prop)
    (nonPromoting : Event → Prop) : State → State → Prop where
  | nil (s : State) : NonPromotingTrace step nonPromoting s s
  | cons {s s' t : State} {e : Event}
      (hstep : step s e s')
      (hnp : nonPromoting e)
      (tail : NonPromotingTrace step nonPromoting s' t) :
      NonPromotingTrace step nonPromoting s t

/--
The concrete refinement obligation for one ordinary/non-promoting step.
-/
def OneStepNonAmplification {State : Type u} {Event : Type v} {Authority : Type w}
    [Preorder Authority]
    (authority : State → Authority)
    (step : State → Event → State → Prop)
    (nonPromoting : Event → Prop) : Prop :=
  ∀ ⦃s e s'⦄, step s e s' → nonPromoting e → authority s' ≤ authority s

/--
Every finite non-promoting execution preserves authority non-amplification.
-/
theorem trace_nonamplification {State : Type u} {Event : Type v} {Authority : Type w}
    [Preorder Authority]
    {authority : State → Authority}
    {step : State → Event → State → Prop}
    {nonPromoting : Event → Prop}
    (one : OneStepNonAmplification authority step nonPromoting)
    {s t : State}
    (trace : NonPromotingTrace step nonPromoting s t) :
    authority t ≤ authority s := by
  induction trace with
  | nil s => exact le_refl _
  | cons hstep hnp tail ih =>
      exact le_trans ih (one hstep hnp)

/--
If every event is explicitly classified as either promoting or non-promoting,
a valid step that violates non-amplification must be in the promoting class.

This theorem does not establish that the promoting event carries a valid
promotion capability; that is a concrete refinement obligation.
-/
theorem nonamplification_violation_implies_promoting
    {State : Type u} {Event : Type v} {Authority : Type w}
    [Preorder Authority]
    {authority : State → Authority}
    {step : State → Event → State → Prop}
    {nonPromoting promoting : Event → Prop}
    (partition : ∀ e, promoting e ∨ nonPromoting e)
    (one : OneStepNonAmplification authority step nonPromoting)
    {s s' : State} {e : Event}
    (hstep : step s e s')
    (hviol : ¬ authority s' ≤ authority s) :
    promoting e := by
  cases partition e with
  | inl hp => exact hp
  | inr hnp =>
      exact False.elim (hviol (one hstep hnp))

/-- Evidence-only events inherit non-amplification once classified non-promoting. -/
theorem evidence_only_nonamplifying
    {State : Type u} {Event : Type v} {Authority : Type w}
    [Preorder Authority]
    {authority : State → Authority}
    {step : State → Event → State → Prop}
    {nonPromoting evidenceOnly : Event → Prop}
    (evidence_is_nonpromoting : ∀ ⦃e⦄, evidenceOnly e → nonPromoting e)
    (one : OneStepNonAmplification authority step nonPromoting)
    {s s' : State} {e : Event}
    (hstep : step s e s')
    (hevidence : evidenceOnly e) :
    authority s' ≤ authority s :=
  one hstep (evidence_is_nonpromoting hevidence)

/--
Security-relevant semantic coordinates bound by an abstract authorization.

Concrete systems may bind additional fields. Refinement into this kernel is not
allowed to erase a concrete security-relevant coordinate.
-/
structure Binding (Id : Type u) where
  subject : Id
  action : Id
  resource : Id
  policy : Id
  evidence : Id
  authority : Id
  context : Id
  authorityEpoch : Nat
  policyEpoch : Nat
  evidenceEpoch : Nat

/-- Abstract admission requires exact semantic binding equality. -/
def bindingAccepts {Id : Type u} (authorized candidate : Binding Id) : Prop :=
  candidate = authorized

/-- An exactly identical binding is admitted. -/
theorem binding_exact_accepts {Id : Type u} (b : Binding Id) : bindingAccepts b b := by
  rfl

/-- Any changed subject invalidates the abstract binding. -/
theorem subject_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.subject ≠ authorized.subject) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed action invalidates the abstract binding. -/
theorem action_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.action ≠ authorized.action) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed resource invalidates the abstract binding. -/
theorem resource_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.resource ≠ authorized.resource) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed policy identity invalidates the abstract binding. -/
theorem policy_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.policy ≠ authorized.policy) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed evidence identity invalidates the abstract binding. -/
theorem evidence_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.evidence ≠ authorized.evidence) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed authority identity invalidates the abstract binding. -/
theorem authority_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.authority ≠ authorized.authority) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed context identity invalidates the abstract binding. -/
theorem context_mismatch_reject {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.context ≠ authorized.context) : ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed authority epoch invalidates the abstract binding. -/
theorem authority_epoch_binding_mismatch_reject
    {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.authorityEpoch ≠ authorized.authorityEpoch) :
    ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed policy epoch invalidates the abstract binding. -/
theorem policy_epoch_binding_mismatch_reject
    {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.policyEpoch ≠ authorized.policyEpoch) :
    ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Any changed evidence epoch invalidates the abstract binding. -/
theorem evidence_epoch_binding_mismatch_reject
    {Id : Type u} {authorized candidate : Binding Id}
    (h : candidate.evidenceEpoch ≠ authorized.evidenceEpoch) :
    ¬ bindingAccepts authorized candidate := by
  intro hacc
  cases hacc
  exact h rfl

/-- Live epoch/currentness state observed at the abstract admission boundary. -/
structure LiveEpochs where
  authorityEpoch : Nat
  policyEpoch : Nat
  evidenceEpoch : Nat

/-- Prepared authorization is current only when every required epoch still matches. -/
def isCurrent {Id : Type u} (prepared : Binding Id) (live : LiveEpochs) : Prop :=
  prepared.authorityEpoch = live.authorityEpoch ∧
  prepared.policyEpoch = live.policyEpoch ∧
  prepared.evidenceEpoch = live.evidenceEpoch

/-- Authority-epoch drift rejects currentness. -/
theorem authority_epoch_currentness_mismatch_reject
    {Id : Type u} {prepared : Binding Id} {live : LiveEpochs}
    (h : prepared.authorityEpoch ≠ live.authorityEpoch) : ¬ isCurrent prepared live := by
  intro hcurrent
  exact h hcurrent.1

/-- Policy-epoch drift rejects currentness. -/
theorem policy_epoch_currentness_mismatch_reject
    {Id : Type u} {prepared : Binding Id} {live : LiveEpochs}
    (h : prepared.policyEpoch ≠ live.policyEpoch) : ¬ isCurrent prepared live := by
  intro hcurrent
  exact h hcurrent.2.1

/-- Evidence-epoch drift rejects currentness. -/
theorem evidence_epoch_currentness_mismatch_reject
    {Id : Type u} {prepared : Binding Id} {live : LiveEpochs}
    (h : prepared.evidenceEpoch ≠ live.evidenceEpoch) : ¬ isCurrent prepared live := by
  intro hcurrent
  exact h hcurrent.2.2

#print axioms trace_nonamplification
#print axioms nonamplification_violation_implies_promoting
#print axioms evidence_only_nonamplifying
#print axioms binding_exact_accepts
#print axioms subject_mismatch_reject
#print axioms action_mismatch_reject
#print axioms resource_mismatch_reject
#print axioms policy_mismatch_reject
#print axioms evidence_mismatch_reject
#print axioms authority_mismatch_reject
#print axioms context_mismatch_reject
#print axioms authority_epoch_binding_mismatch_reject
#print axioms policy_epoch_binding_mismatch_reject
#print axioms evidence_epoch_binding_mismatch_reject
#print axioms authority_epoch_currentness_mismatch_reject
#print axioms policy_epoch_currentness_mismatch_reject
#print axioms evidence_epoch_currentness_mismatch_reject

end Symthaea.Formal.Authority
