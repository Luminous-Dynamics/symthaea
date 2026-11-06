import assurance.Closure

namespace Symthaea.Formal.REA

open Symthaea.Formal.Assurance

/-- Closed REA claim vocabulary. V0 has a rule only for the observation meta-claim. -/
inductive ClaimPredicate where
  | ProviderReportedObservation
  | MemberContentIntegrity
  | ArchiveContentIdentity
  | WholeArchiveCompleteness
  | ProducerAuthenticity
  | SignatureCryptographicValidity
  | SignerIdentity
  | SignerTrust
  | PackageAuthorization
  | PlatformAcceptance
  | ProviderCorroboration
  | CausalFinding
  | SecurityControlPass
  deriving DecidableEq, Repr

/-- Confidence is descriptive metadata, not a strength ordering. -/
inductive ObservationConfidence where
  | Observed
  | Derived
  | Inferred
  deriving DecidableEq, Repr

/-- Authority is descriptive provenance, not a claim-promotion lattice. -/
inductive ObservationAuthority where
  | ShippedArtifact
  | ControlledReplay
  | HistoricalReference
  | ExternalService
  | AnalystInference
  deriving DecidableEq, Repr

/--
The complete REA request identity needed at the formal boundary.

The digest commits to the full canonical request, including the asserted value,
ordered Evidence IDs, assumptions, falsification condition and represented
scope. Separate schema and rule generations prevent a new interpretation from
silently reusing an old result.
-/
structure ClaimBinding where
  claimRequestDigest : Nat
  requestSchemaGeneration : Nat
  resultSchemaGeneration : Nat
  evaluatorRuleGeneration : GenerationIdentity
  ruleId : Nat
  operation : Nat
  environment : Nat
  predicate : ClaimPredicate
  deriving DecidableEq, Repr

/--
A provider-observation receipt wraps the existing generic closure receipt.
The distinct ProviderObservation class is not RuntimeObservation, SourceAudit
or AbstractProof. Confidence and authority remain orthogonal descriptors.
-/
structure ProviderObservationReceipt where
  base : EvidenceReceipt
  binding : ClaimBinding
  confidence : ObservationConfidence
  authority : ObservationAuthority
  deriving Repr

/-- The only predicate kind supported by the REA evaluator v0 rule. -/
structure ProviderObservationObligation where
  base : Obligation
  binding : ClaimBinding
  deriving Repr

/--
A positive REA v0 result supports only the exact provider-report meta-claim.
The generic closure kernel still enforces currentness, result polarity,
generation, subject, claim, property kind, assumptions and dependency closure.
This adapter adds the exact canonical request/schema/rule/operation/environment
binding and the distinct evidence class.
-/
def ProviderObservationSupports
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation) : Prop :=
  receipt.binding = obligation.binding ∧
  receipt.binding.predicate = .ProviderReportedObservation ∧
  obligation.binding.predicate = .ProviderReportedObservation ∧
  receipt.binding.evaluatorRuleGeneration = receipt.base.generation ∧
  obligation.binding.evaluatorRuleGeneration = obligation.base.generation ∧
  receipt.base.evidenceClass = .ProviderObservation ∧
  obligation.base.requiredClass = .ProviderObservation ∧
  Supports refinements entailments receipt.base obligation.base

/-- Exact request, schema, rule, operation and environment bindings are retained. -/
theorem provider_observation_support_binds_full_request
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hs : ProviderObservationSupports refinements entailments receipt obligation) :
    receipt.binding = obligation.binding := by
  exact hs.1

/-- The formal evidence class remains distinct from all existing stronger classes. -/
theorem provider_observation_is_a_distinct_evidence_class :
    EvidenceClass.ProviderObservation ≠ EvidenceClass.RuntimeObservation ∧
    EvidenceClass.ProviderObservation ≠ EvidenceClass.SourceAudit ∧
    EvidenceClass.ProviderObservation ≠ EvidenceClass.AbstractProof := by
  exact ⟨by decide, by decide, by decide⟩

/-- Any successful v0 adapter result has the provider-observation meta-claim kind. -/
theorem provider_observation_support_has_exact_predicate
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hs : ProviderObservationSupports refinements entailments receipt obligation) :
    obligation.binding.predicate = .ProviderReportedObservation := by
  exact hs.2.2.1

/--
A provider observation cannot close an obligation about an underlying artifact
property. This generic theorem covers integrity, authenticity, signature
validity, signer trust, authorization, completeness, platform acceptance and
security-control predicates: none equals ProviderReportedObservation.
-/
theorem provider_observation_cannot_promote_to_other_predicate
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hOther : obligation.binding.predicate ≠ .ProviderReportedObservation)
    (hs : ProviderObservationSupports refinements entailments receipt obligation) :
    False := by
  exact hOther (provider_observation_support_has_exact_predicate
    refinements entailments receipt obligation hs)

/-- In particular, an observed provider report is not member-content integrity. -/
theorem provider_observation_cannot_close_member_integrity
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hIntegrity : obligation.binding.predicate = .MemberContentIntegrity) :
    ¬ ProviderObservationSupports refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    refinements entailments receipt obligation
    (by rw [hIntegrity]; decide)
    hs

/-- Provider observation cannot establish producer authenticity. -/
theorem provider_observation_cannot_close_producer_authenticity
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hAuthenticity : obligation.binding.predicate = .ProducerAuthenticity) :
    ¬ ProviderObservationSupports refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    refinements entailments receipt obligation
    (by rw [hAuthenticity]; decide)
    hs

/-- Provider observation cannot establish signer trust. -/
theorem provider_observation_cannot_close_signer_trust
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hTrust : obligation.binding.predicate = .SignerTrust) :
    ¬ ProviderObservationSupports refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    refinements entailments receipt obligation
    (by rw [hTrust]; decide)
    hs

/-- Provider observation cannot establish package authorization. -/
theorem provider_observation_cannot_close_package_authorization
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hAuthorization : obligation.binding.predicate = .PackageAuthorization) :
    ¬ ProviderObservationSupports refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    refinements entailments receipt obligation
    (by rw [hAuthorization]; decide)
    hs

/-- Provider observation cannot establish a security-control pass. -/
theorem provider_observation_cannot_close_security_control
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hControl : obligation.binding.predicate = .SecurityControlPass) :
    ¬ ProviderObservationSupports refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    refinements entailments receipt obligation
    (by rw [hControl]; decide)
    hs

/--
Changing confidence/authority descriptors does not alter the exact receipt
identity, evidence class, claim scope or admissibility relation.
-/
def withObservationDescriptors
    (receipt : ProviderObservationReceipt)
    (confidence : ObservationConfidence)
    (authority : ObservationAuthority) : ProviderObservationReceipt :=
  { receipt with confidence := confidence, authority := authority }

theorem confidence_authority_are_non_amplifying
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (confidence : ObservationConfidence)
    (authority : ObservationAuthority) :
    ProviderObservationSupports refinements entailments
      (withObservationDescriptors receipt confidence authority) obligation
      ↔ ProviderObservationSupports refinements entailments receipt obligation := by
  rfl

/--
A semantic contradiction from a provider is bound to the same exact provider
observation request. Its predicate remains the observation meta-claim; there
is no conversion from this relation to an artifact-property counterexample.
-/
def ProviderObservationRefutesMetaClaim
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation) : Prop :=
  receipt.binding = obligation.binding ∧
  receipt.binding.predicate = .ProviderReportedObservation ∧
  receipt.base.evidenceClass = .ProviderObservation ∧
  receipt.base.validity = .Current ∧
  receipt.base.generation = obligation.base.generation ∧
  receipt.base.subject = obligation.base.subject ∧
  receipt.base.claim = obligation.base.claim ∧
  semanticPolarity receipt.base.result = .SemanticCounterevidence

theorem provider_observation_refutation_remains_meta_claim
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (h : ProviderObservationRefutesMetaClaim receipt obligation) :
    receipt.binding.predicate = .ProviderReportedObservation := by
  exact h.2.1

/--
Two differently named/represented provider reports sharing one provenance root
cannot satisfy the generic kernel's independent-support requirement. The
existing reviewed independence witness and provenance rule remain authoritative.
-/
theorem provider_aliases_do_not_create_independence
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (independence : List IndependenceWitness)
    (left right : ProviderObservationReceipt)
    (leftObligation rightObligation : ProviderObservationObligation)
    (hSameRoot : left.base.provenanceRoot = right.base.provenanceRoot) :
    ¬ IndependentSupport refinements entailments independence
      left.base right.base leftObligation.base rightObligation.base := by
  exact same_provenance_cannot_count_as_independent
    refinements entailments independence left.base right.base
    leftObligation.base rightObligation.base hSameRoot

#print axioms provider_observation_support_binds_full_request
#print axioms provider_observation_is_a_distinct_evidence_class
#print axioms provider_observation_support_has_exact_predicate
#print axioms provider_observation_cannot_promote_to_other_predicate
#print axioms provider_observation_cannot_close_member_integrity
#print axioms provider_observation_cannot_close_producer_authenticity
#print axioms provider_observation_cannot_close_signer_trust
#print axioms provider_observation_cannot_close_package_authorization
#print axioms provider_observation_cannot_close_security_control
#print axioms confidence_authority_are_non_amplifying
#print axioms provider_observation_refutation_remains_meta_claim
#print axioms provider_aliases_do_not_create_independence

end Symthaea.Formal.REA
