use super::*;
use symthaea_assurance_campaign::{
    CampaignEvidenceLedgerV1, EvidenceRequirementV1, OrderingReceiptV1, PreregistrationReceiptV1,
    RegistrationStatementV1, ReproductionRequirementV1, SupportCriterionV1,
    evidence_commitment_statement_digest, resolve_terminal_registration_in_view,
};
use symthaea_assurance_core::{EvidenceKind, EvidenceProvenance};
use symthaea_assurance_semantics::{DefinitionSchemaV1, SemanticCommitmentV1};
use symthaea_assurance_subject::{
    AiSurfaceKind, MaterialCommitment, SurfaceBinding, SurfaceLocator, SurfaceProfile, SurfaceState,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn semantic(name: &str, byte: char) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
        id(name),
        DefinitionSchemaV1::new(
            id("symthaea.assurance.test-semantic-schema.v1"),
            digest('e'),
        ),
        digest(byte),
    )
}

fn subject() -> AiSubjectManifest {
    let profile = SurfaceProfile::new(id("resolver-profile"), vec![AiSurfaceKind::Model]).unwrap();
    AiSubjectManifest::new(
        id("resolver-agent"),
        profile,
        vec![
            SurfaceBinding::applicable(
                AiSurfaceKind::Model,
                SurfaceLocator::new(Some(id("provider-a")), id("model-a"), Some(id("v1"))),
                SurfaceState::Known(MaterialCommitment::artifact_bytes(digest('a'))),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn claim(subject: &AiSubjectManifest) -> Claim {
    Claim::new(
        id("authority-boundary-claim"),
        subject.core_subject_id().unwrap(),
        "high-impact action requires registered approval authority",
        id("agent-authority"),
    )
    .unwrap()
}

fn plan(subject: &AiSubjectManifest) -> CampaignPlanV1 {
    let claim = claim(subject);
    CampaignPlanV1::new(
        id("resolver-plan"),
        id("resolver-campaign"),
        &claim,
        subject,
        SupportTier::CausallySupported,
        ReproductionRequirementV1::NotRequired,
        vec![
            EvidenceRequirementV1::builtin(EvidenceKind::Observation).unwrap(),
            EvidenceRequirementV1::builtin(EvidenceKind::ControlledIntervention).unwrap(),
        ],
        vec![
            SupportCriterionV1::new(
                SupportTier::Observed,
                semantic("observable-authority-decision", '1'),
            ),
            SupportCriterionV1::new(
                SupportTier::CausallySupported,
                semantic("authority-boundary-causal-effect", '2'),
            ),
        ],
        vec![
            semantic("baseline", '3'),
            semantic("denied-path", '4'),
            semantic("valid-authority", '5'),
        ],
        vec![semantic("allowed-path-broken", '6')],
        vec![semantic("unauthorized-execution-observed", '7')],
        vec![semantic("instrumentation-incomplete", '8')],
        vec![],
    )
    .unwrap()
}

fn ordering(sequence: u64, statement: DigestSha256, receipt_byte: char) -> OrderingReceiptV1 {
    OrderingReceiptV1::new(
        id("transparency-log-a"),
        semantic("monotonic-ordering-profile-v1", 'f'),
        1,
        sequence,
        statement,
        digest(receipt_byte),
    )
    .unwrap()
}

struct Fixture {
    subject: AiSubjectManifest,
    claim: Claim,
    plan: CampaignPlanV1,
    current: TerminalRegistrationInViewV1,
    ledger: CampaignEvidenceLedgerV1,
    next_sequence: u64,
    next_artifact: usize,
    entries: Vec<ResolverEvidenceV1>,
}

impl Fixture {
    fn new() -> Self {
        let subject = subject();
        let claim = claim(&subject);
        let plan = plan(&subject);
        let statement = RegistrationStatementV1::new(&plan, id("registrar-a"), None).unwrap();
        let registration = PreregistrationReceiptV1::new(
            statement.clone(),
            ordering(10, statement.digest(), 'a'),
            None,
        )
        .unwrap();
        let current = resolve_terminal_registration_in_view(&[registration], &[]).unwrap();
        let ledger = CampaignEvidenceLedgerV1::new(&current);
        Self {
            subject,
            claim,
            plan,
            current,
            ledger,
            next_sequence: 11,
            next_artifact: 0,
            entries: Vec::new(),
        }
    }

    fn admit(
        &mut self,
        role: PredicateRoleV1,
        semantic: SemanticCommitmentV1,
        disposition: PredicateDispositionV1,
        kind: EvidenceKind,
    ) {
        const DIGEST_BYTES: [char; 16] = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
        ];
        let artifact_byte = DIGEST_BYTES[self.next_artifact % DIGEST_BYTES.len()];
        self.next_artifact += 1;
        let evidence = EvidenceArtifact::new(
            id(&format!("evidence-{}", self.next_artifact)),
            self.subject.core_subject_id().unwrap(),
            self.claim.digest(),
            kind,
            digest(artifact_byte),
            EvidenceProvenance::new(id("producer"), id("executor"), Some(id("verifier")), None),
        );
        let commitment_statement = evidence_commitment_statement_digest(&self.current, &evidence);
        let commitment_ordering = ordering(self.next_sequence, commitment_statement, 'b');
        self.next_sequence += 1;
        let admission_statement = self
            .ledger
            .admission_statement_digest(&self.current, &evidence, &commitment_ordering)
            .unwrap();
        let admission_ordering = ordering(self.next_sequence, admission_statement, 'c');
        self.next_sequence += 1;

        let previous_root = self.ledger.evidence_root().clone();
        let ordinal = self.ledger.admitted_count() + 1;
        let admission = self
            .ledger
            .admit_preregistered(
                &self.plan,
                &self.current,
                &evidence,
                &commitment_ordering,
                &admission_ordering,
            )
            .unwrap();
        let binding = AdmissionBindingV1::new(
            self.current.digest(),
            self.plan.digest(),
            self.plan.campaign_nonce().clone(),
            ordinal,
            evidence.digest(),
            commitment_ordering.digest(),
            admission_ordering.digest(),
            previous_root,
            admission.evidence_root().clone(),
        );
        let observation = PredicateObservationV1::new(
            self.plan.digest(),
            evidence.digest(),
            role,
            semantic,
            disposition,
        );
        self.entries.push(ResolverEvidenceV1::new(
            evidence,
            &admission,
            binding,
            observation,
        ));
    }

    fn admit_cleared_guards(&mut self) {
        self.admit(
            PredicateRoleV1::Control,
            semantic("baseline", '3'),
            PredicateDispositionV1::Satisfied,
            EvidenceKind::Observation,
        );
        self.admit(
            PredicateRoleV1::Control,
            semantic("denied-path", '4'),
            PredicateDispositionV1::Satisfied,
            EvidenceKind::Observation,
        );
        self.admit(
            PredicateRoleV1::Control,
            semantic("valid-authority", '5'),
            PredicateDispositionV1::Satisfied,
            EvidenceKind::Observation,
        );
        self.admit(
            PredicateRoleV1::FailureCondition,
            semantic("allowed-path-broken", '6'),
            PredicateDispositionV1::NotSatisfied,
            EvidenceKind::Observation,
        );
        self.admit(
            PredicateRoleV1::ContradictionCondition,
            semantic("unauthorized-execution-observed", '7'),
            PredicateDispositionV1::NotSatisfied,
            EvidenceKind::Observation,
        );
        self.admit(
            PredicateRoleV1::InconclusiveCondition,
            semantic("instrumentation-incomplete", '8'),
            PredicateDispositionV1::NotSatisfied,
            EvidenceKind::Observation,
        );
    }

    fn admit_observed_support(&mut self) {
        self.admit(
            PredicateRoleV1::SupportCriterion,
            semantic("observable-authority-decision", '1'),
            PredicateDispositionV1::Satisfied,
            EvidenceKind::Observation,
        );
    }

    fn resolve(&self) -> ResolutionV1 {
        resolve_v1(
            &self.plan,
            &self.current,
            &self.subject,
            &self.claim,
            self.ledger.evidence_root(),
            &self.entries,
        )
    }
}

#[test]
fn observed_support_does_not_promote_to_campaign_ceiling() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Supported(SupportTier::Observed)
    );
}

#[test]
fn causal_support_requires_declared_criterion_and_core_evidence_class() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.admit(
        PredicateRoleV1::SupportCriterion,
        semantic("authority-boundary-causal-effect", '2'),
        PredicateDispositionV1::Satisfied,
        EvidenceKind::ControlledIntervention,
    );
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Supported(SupportTier::CausallySupported)
    );
}

#[test]
fn causal_semantic_without_controlled_intervention_falls_back() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.admit(
        PredicateRoleV1::SupportCriterion,
        semantic("authority-boundary-causal-effect", '2'),
        PredicateDispositionV1::Satisfied,
        EvidenceKind::Observation,
    );
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Supported(SupportTier::Observed)
    );
}

#[test]
fn higher_tier_miss_falls_back_to_lower_supported_tier() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.admit(
        PredicateRoleV1::SupportCriterion,
        semantic("authority-boundary-causal-effect", '2'),
        PredicateDispositionV1::NotSatisfied,
        EvidenceKind::ControlledIntervention,
    );
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Supported(SupportTier::Observed)
    );
}

#[test]
fn contradiction_is_refuted_not_low_score() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    let contradiction = fixture
        .entries
        .iter_mut()
        .find(|item| {
            item.observation.role == PredicateRoleV1::ContradictionCondition
                && item.observation.semantic.semantic_id().as_str()
                    == "unauthorized-execution-observed"
        })
        .unwrap();
    contradiction.observation.disposition = PredicateDispositionV1::Satisfied;
    assert_eq!(fixture.resolve().outcome(), ResolutionOutcomeV1::Refuted);
}

#[test]
fn opposite_observations_remain_conflicted() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.admit(
        PredicateRoleV1::ContradictionCondition,
        semantic("unauthorized-execution-observed", '7'),
        PredicateDispositionV1::Satisfied,
        EvidenceKind::Observation,
    );
    assert_eq!(fixture.resolve().outcome(), ResolutionOutcomeV1::Conflicted);
}

#[test]
fn failed_control_is_insufficient() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    let denied_path = fixture
        .entries
        .iter_mut()
        .find(|item| {
            item.observation.role == PredicateRoleV1::Control
                && item.observation.semantic.semantic_id().as_str() == "denied-path"
        })
        .unwrap();
    denied_path.observation.disposition = PredicateDispositionV1::NotSatisfied;
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Insufficient
    );
}

#[test]
fn missing_guard_is_insufficient() {
    let mut fixture = Fixture::new();
    fixture.admit_observed_support();
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Insufficient
    );
}

#[test]
fn inconclusive_guard_is_insufficient() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.admit(
        PredicateRoleV1::InconclusiveCondition,
        semantic("instrumentation-incomplete", '8'),
        PredicateDispositionV1::Inconclusive,
        EvidenceKind::Observation,
    );
    assert_eq!(
        fixture.resolve().outcome(),
        ResolutionOutcomeV1::Insufficient
    );
}

#[test]
fn changed_semantic_schema_is_malformed() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    let last = fixture.entries.last_mut().unwrap();
    last.observation.semantic = SemanticCommitmentV1::new(
        id("observable-authority-decision"),
        DefinitionSchemaV1::new(id("different-schema"), digest('e')),
        digest('1'),
    );
    assert_eq!(fixture.resolve().outcome(), ResolutionOutcomeV1::Malformed);
}

#[test]
fn mutated_admission_binding_is_malformed() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.entries[0].admission_binding.ordinal += 1;
    assert_eq!(fixture.resolve().outcome(), ResolutionOutcomeV1::Malformed);
}

#[test]
fn omitted_admission_breaks_root_chain() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.entries.remove(2);
    assert_eq!(fixture.resolve().outcome(), ResolutionOutcomeV1::Malformed);
}

#[test]
fn resolution_is_input_order_independent() {
    let mut fixture = Fixture::new();
    fixture.admit_cleared_guards();
    fixture.admit_observed_support();
    fixture.admit(
        PredicateRoleV1::SupportCriterion,
        semantic("authority-boundary-causal-effect", '2'),
        PredicateDispositionV1::Satisfied,
        EvidenceKind::ControlledIntervention,
    );
    let first = fixture.resolve();
    fixture.entries.reverse();
    let second = fixture.resolve();
    assert_eq!(first.outcome(), second.outcome());
    assert_eq!(first.digest(), second.digest());
}
