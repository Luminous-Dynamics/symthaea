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
  claimRequestDigest : String
  derivationDigest : String
  requestSchemaGeneration : String
  resultSchemaGeneration : String
  evaluatorRuleGeneration : GenerationIdentity
  evaluatorVersion : String
  ruleId : String
  claimId : String
  subjectSha256 : String
  operation : String
  platform : Option String
  platformVersion : Option String
  environment : Option String
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
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
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
  Supports refinements entailments receipt.base obligation.base ∧
  EvidenceClosed rank receipts receipt.base ∧
  ReachabilitySatisfied reachability obligation.base

/-- Positive formal support requires every declared evidence dependency to close. -/
theorem provider_observation_support_requires_dependency_closure
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hs : ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation) :
    EvidenceClosed rank receipts receipt.base := by
  rcases hs with ⟨_, _, _, _, _, _, _, _, hClosed, _⟩
  exact hClosed

/-- Positive formal support requires an explicit current reachability witness when the obligation requires one. -/
theorem provider_observation_support_requires_reachability
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hs : ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation) :
    ReachabilitySatisfied reachability obligation.base := by
  rcases hs with ⟨_, _, _, _, _, _, _, _, _, hReachability⟩
  exact hReachability

/-- Exact request, schema, rule, operation and environment bindings are retained. -/
theorem provider_observation_support_binds_full_request
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hs : ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation) :
    receipt.binding = obligation.binding := by
  exact hs.1

/-- A changed producer derivation commitment cannot reuse a prior formal result. -/
theorem changed_derivation_digest_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hDigest : receipt.binding.derivationDigest ≠ obligation.binding.derivationDigest) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hDigest (congrArg ClaimBinding.derivationDigest hs.1)

/-- A changed human-facing claim ID cannot reuse a prior formal result. -/
theorem changed_claim_id_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hClaim : receipt.binding.claimId ≠ obligation.binding.claimId) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hClaim (congrArg ClaimBinding.claimId hs.1)

/-- A changed exact subject digest cannot reuse a prior formal result. -/
theorem changed_subject_digest_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hSubject : receipt.binding.subjectSha256 ≠ obligation.binding.subjectSha256) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hSubject (congrArg ClaimBinding.subjectSha256 hs.1)

/-- A changed canonical request digest cannot reuse a prior result. -/
theorem changed_claim_digest_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hDigest : receipt.binding.claimRequestDigest ≠ obligation.binding.claimRequestDigest) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hDigest (congrArg ClaimBinding.claimRequestDigest hs.1)

/-- A changed request schema generation cannot reuse a prior result. -/
theorem changed_request_schema_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hSchema : receipt.binding.requestSchemaGeneration ≠ obligation.binding.requestSchemaGeneration) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hSchema (congrArg ClaimBinding.requestSchemaGeneration hs.1)

/-- A changed result schema generation cannot reuse a prior result. -/
theorem changed_result_schema_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hSchema : receipt.binding.resultSchemaGeneration ≠ obligation.binding.resultSchemaGeneration) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hSchema (congrArg ClaimBinding.resultSchemaGeneration hs.1)

/-- A changed evaluator generation cannot reuse a prior result. -/
theorem changed_rule_generation_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hRule : receipt.binding.evaluatorRuleGeneration ≠ obligation.binding.evaluatorRuleGeneration) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hRule (congrArg ClaimBinding.evaluatorRuleGeneration hs.1)

/-- A changed target platform cannot reuse a provider observation. -/
theorem changed_platform_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hPlatform : receipt.binding.platform ≠ obligation.binding.platform) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hPlatform (congrArg ClaimBinding.platform hs.1)

/-- A changed platform version cannot reuse a provider observation. -/
theorem changed_platform_version_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hVersion : receipt.binding.platformVersion ≠ obligation.binding.platformVersion) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hVersion (congrArg ClaimBinding.platformVersion hs.1)

/-- A changed represented environment cannot reuse a provider observation. -/
theorem changed_environment_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hEnvironment : receipt.binding.environment ≠ obligation.binding.environment) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hEnvironment (congrArg ClaimBinding.environment hs.1)

/-- The formal evidence class remains distinct from all existing stronger classes. -/
theorem provider_observation_is_a_distinct_evidence_class :
    EvidenceClass.ProviderObservation ≠ EvidenceClass.RuntimeObservation ∧
    EvidenceClass.ProviderObservation ≠ EvidenceClass.SourceAudit ∧
    EvidenceClass.ProviderObservation ≠ EvidenceClass.AbstractProof := by
  exact ⟨by decide, by decide, by decide⟩

/-- Any successful v0 adapter result has the provider-observation meta-claim kind. -/
theorem provider_observation_support_has_exact_predicate
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hs : ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation) :
    obligation.binding.predicate = .ProviderReportedObservation := by
  exact hs.2.2.1

/--
A provider observation cannot close an obligation about an underlying artifact
property. This generic theorem covers integrity, authenticity, signature
validity, signer trust, authorization, completeness, platform acceptance and
security-control predicates: none equals ProviderReportedObservation.
-/
theorem provider_observation_cannot_promote_to_other_predicate
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hOther : obligation.binding.predicate ≠ .ProviderReportedObservation)
    (hs : ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation) :
    False := by
  exact hOther (provider_observation_support_has_exact_predicate
    rank receipts reachability refinements entailments receipt obligation hs)

/-- In particular, an observed provider report is not member-content integrity. -/
theorem provider_observation_cannot_close_member_integrity
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hIntegrity : obligation.binding.predicate = .MemberContentIntegrity) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hIntegrity]; decide)
    hs

/-- Provider observation cannot establish producer authenticity. -/
theorem provider_observation_cannot_close_producer_authenticity
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hAuthenticity : obligation.binding.predicate = .ProducerAuthenticity) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hAuthenticity]; decide)
    hs

/-- Provider observation cannot establish signer trust. -/
theorem provider_observation_cannot_close_signer_trust
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hTrust : obligation.binding.predicate = .SignerTrust) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hTrust]; decide)
    hs

/-- Provider observation cannot establish package authorization. -/
theorem provider_observation_cannot_close_package_authorization
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hAuthorization : obligation.binding.predicate = .PackageAuthorization) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hAuthorization]; decide)
    hs

/-- Provider observation cannot establish a security-control pass. -/
theorem provider_observation_cannot_close_security_control
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hControl : obligation.binding.predicate = .SecurityControlPass) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hControl]; decide)
    hs

/-- A changed rule identifier cannot reuse a prior result. -/
theorem changed_rule_id_cannot_support
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hRule : receipt.binding.ruleId ≠ obligation.binding.ruleId) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact hRule (congrArg ClaimBinding.ruleId hs.1)

/-- Provider observation cannot establish cryptographic signature validity. -/
theorem provider_observation_cannot_close_signature_validity
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hSignature : obligation.binding.predicate = .SignatureCryptographicValidity) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hSignature]; decide)
    hs

/-- Provider observation cannot establish signer identity. -/
theorem provider_observation_cannot_close_signer_identity
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hIdentity : obligation.binding.predicate = .SignerIdentity) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hIdentity]; decide)
    hs

/-- Provider observation cannot establish whole-archive completeness. -/
theorem provider_observation_cannot_close_archive_completeness
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hCompleteness : obligation.binding.predicate = .WholeArchiveCompleteness) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hCompleteness]; decide)
    hs

/-- Provider observation cannot establish platform acceptance. -/
theorem provider_observation_cannot_close_platform_acceptance
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (hAcceptance : obligation.binding.predicate = .PlatformAcceptance) :
    ¬ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
  intro hs
  exact provider_observation_cannot_promote_to_other_predicate
    rank receipts reachability refinements entailments receipt obligation
    (by rw [hAcceptance]; decide)
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
    (rank : EvidenceIdentity → Nat)
    (receipts : List EvidenceReceipt)
    (reachability : List ReachabilityWitness)
    (refinements : List RefinementWitness)
    (entailments : List ClaimEntailmentWitness)
    (receipt : ProviderObservationReceipt)
    (obligation : ProviderObservationObligation)
    (confidence : ObservationConfidence)
    (authority : ObservationAuthority) :
    ProviderObservationSupports rank receipts reachability refinements entailments
      (withObservationDescriptors receipt confidence authority) obligation
      ↔ ProviderObservationSupports rank receipts reachability refinements entailments receipt obligation := by
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
#print axioms provider_observation_support_requires_dependency_closure
#print axioms provider_observation_support_requires_reachability
#print axioms changed_derivation_digest_cannot_support
#print axioms changed_claim_id_cannot_support
#print axioms changed_subject_digest_cannot_support
#print axioms changed_claim_digest_cannot_support
#print axioms changed_request_schema_cannot_support
#print axioms changed_result_schema_cannot_support
#print axioms changed_rule_generation_cannot_support
#print axioms changed_rule_id_cannot_support
#print axioms changed_platform_cannot_support
#print axioms changed_platform_version_cannot_support
#print axioms changed_environment_cannot_support
#print axioms provider_observation_is_a_distinct_evidence_class
#print axioms provider_observation_support_has_exact_predicate
#print axioms provider_observation_cannot_promote_to_other_predicate
#print axioms provider_observation_cannot_close_member_integrity
#print axioms provider_observation_cannot_close_producer_authenticity
#print axioms provider_observation_cannot_close_signature_validity
#print axioms provider_observation_cannot_close_signer_identity
#print axioms provider_observation_cannot_close_signer_trust
#print axioms provider_observation_cannot_close_package_authorization
#print axioms provider_observation_cannot_close_archive_completeness
#print axioms provider_observation_cannot_close_platform_acceptance
#print axioms provider_observation_cannot_close_security_control
#print axioms confidence_authority_are_non_amplifying
#print axioms provider_observation_refutation_remains_meta_claim
#print axioms provider_aliases_do_not_create_independence

end Symthaea.Formal.REA
