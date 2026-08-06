use symthaea_legal_reasoning::Atom;
use symthaea_legal_reasoning::{
    AdjudicationOutcome, AdjudicationRequest, AuthorityId, AuthorityLedger, AuthorityRecord,
    AuthorityRecordId, AuthorityWeight, BurdenPlan, BurdenStage, BurdenStageId, CanonicalEvidence,
    ClaimDefinition, ClaimId, ClaimOutcome, ClaimSubmission, CurrencyId, DecisionId, DocumentId,
    FactAssertion, FactBase, FormalRule, GrantedRemedy, InferenceProfile, IssueId,
    IssueRequirement, JurisdictionId, LegalDate, Literal, MonetaryRemedyPlan, Money, PartyId,
    ProvisionId, RemedyComponent, RemedyComponentId, RemedyComponentKind, RemedyDefinition,
    RemedyEligibility, RemedyId, RemedyKind, RemedyOutcome, RuleId, RuleKind, RulePack, RulePackId,
    SourceRef, adjudicate, infer, validate_adjudication_record, validate_authority_ledger,
    validate_claim_definition, validate_remedy_definition,
};

fn positive(value: &str) -> Literal {
    Literal::Positive(Atom::new(value).unwrap())
}

fn date() -> LegalDate {
    LegalDate::new(2026, 7, 21).unwrap()
}

fn source(document: &str, provision: &str) -> SourceRef {
    SourceRef::new(
        DocumentId::new(document).unwrap(),
        ProvisionId::new(provision).unwrap(),
    )
}

fn pack() -> RulePack {
    RulePack::new(
        RulePackId::new("contract-v06").unwrap(),
        [
            FormalRule::new(
                RuleId::new("agreement-holding").unwrap(),
                RuleKind::Strict,
                [positive("signed")],
                positive("agreement"),
            )
            .unwrap()
            .with_source(source("contract-act", "agreement")),
            FormalRule::new(
                RuleId::new("breach-holding").unwrap(),
                RuleKind::Strict,
                [positive("nonperformance")],
                positive("breach"),
            )
            .unwrap()
            .with_source(source("contract-act", "breach")),
        ],
        [],
    )
    .unwrap()
}

fn ledger() -> AuthorityLedger {
    AuthorityLedger::new(
        [AuthorityRecord::new(
            AuthorityRecordId::new("binding-case").unwrap(),
            DocumentId::new("binding-case").unwrap(),
            AuthorityId::new("supreme-court").unwrap(),
            JurisdictionId::new("ZA").unwrap(),
            date(),
            AuthorityWeight::Binding,
            [
                RuleId::new("agreement-holding").unwrap(),
                RuleId::new("breach-holding").unwrap(),
            ],
            source("binding-case", "holding"),
        )
        .unwrap()],
        [],
    )
    .unwrap()
}

fn element(issue: &str) -> IssueRequirement {
    IssueRequirement::new(
        IssueId::new(issue).unwrap(),
        positive(issue),
        BurdenPlan::new([BurdenStage::production(
            BurdenStageId::new(format!("{issue}-production")).unwrap(),
            PartyId::new("claimant").unwrap(),
            [],
        )])
        .unwrap(),
    )
    .with_source(source("contract-act", issue))
}

fn claim() -> ClaimDefinition {
    ClaimDefinition::new(
        ClaimId::new("contract-claim").unwrap(),
        PartyId::new("claimant").unwrap(),
        PartyId::new("respondent").unwrap(),
        [element("agreement"), element("breach")],
        [],
        [RemedyId::new("contract-damages").unwrap()],
    )
    .unwrap()
    .with_source(source("contract-act", "cause-of-action"))
}

fn remedy(reverse: bool) -> RemedyDefinition {
    let base = RemedyComponent::new(
        RemedyComponentId::new("expectation").unwrap(),
        RemedyComponentKind::Compensatory,
        Money::new(CurrencyId::new("ZAR").unwrap(), 50_000),
    )
    .with_source(source("damages-schedule", "expectation"));
    let mitigation = RemedyComponent::new(
        RemedyComponentId::new("mitigation").unwrap(),
        RemedyComponentKind::Mitigation,
        Money::new(CurrencyId::new("ZAR").unwrap(), 5_000),
    )
    .with_source(source("damages-schedule", "mitigation"));
    let components = if reverse {
        vec![mitigation, base]
    } else {
        vec![base, mitigation]
    };
    RemedyDefinition::new(
        RemedyId::new("contract-damages").unwrap(),
        [ClaimId::new("contract-claim").unwrap()],
        RemedyEligibility::AllEstablishedClaims,
        RemedyKind::Monetary(
            MonetaryRemedyPlan::new(
                CurrencyId::new("ZAR").unwrap(),
                components,
                Some(Money::new(CurrencyId::new("ZAR").unwrap(), 100_000)),
                None,
            )
            .unwrap(),
        ),
    )
    .unwrap()
    .with_source(source("contract-act", "damages"))
}

fn record(reverse: bool) -> symthaea_legal_reasoning::AdjudicationRecord {
    let pack = pack();
    let facts = if reverse {
        FactBase::new([
            FactAssertion::observed(positive("nonperformance")),
            FactAssertion::observed(positive("signed")),
        ])
    } else {
        FactBase::new([
            FactAssertion::observed(positive("signed")),
            FactAssertion::observed(positive("nonperformance")),
        ])
    };
    let result = infer(&pack, &facts, &InferenceProfile::grounded_blocking_v1()).unwrap();
    let ledger = ledger();
    let authorities = ledger
        .select([AuthorityRecordId::new("binding-case").unwrap()], date())
        .unwrap();
    let request = AdjudicationRequest::new(
        DecisionId::new("decision-1").unwrap(),
        date(),
        [ClaimSubmission::new(claim(), [])],
        [remedy(reverse)],
        authorities,
    )
    .unwrap();
    adjudicate(&request, &result).unwrap()
}

#[test]
fn claim_authority_remedy_and_adjudication_form_one_replayable_chain() {
    let record = record(false);
    assert_eq!(record.outcome, AdjudicationOutcome::Resolved);
    assert_eq!(record.claims[0].outcome, ClaimOutcome::Established);
    assert_eq!(record.remedies[0].outcome, RemedyOutcome::Awarded);
    match record.remedies[0].grant.as_ref().unwrap() {
        GrantedRemedy::Monetary(calculation) => {
            assert_eq!(calculation.total.minor_units, 45_000);
            assert_eq!(calculation.total.currency.as_str(), "ZAR");
        }
        other => panic!("unexpected remedy: {other:?}"),
    }
    assert!(!record.canonical_bytes().is_empty());
}

#[test]
fn v06_artifacts_validate_without_changing_outcomes() {
    let pack = pack();
    let ledger = ledger();
    let claim = claim();
    let remedy = remedy(false);
    let record = record(false);

    assert!(!validate_claim_definition(&claim).has_errors());
    assert!(!validate_remedy_definition(&remedy, &[claim]).has_errors());
    assert!(!validate_authority_ledger(&ledger, &pack).has_errors());
    assert!(!validate_adjudication_record(&record, &ledger).has_errors());
    assert_eq!(record.outcome, AdjudicationOutcome::Resolved);
}

#[test]
fn canonical_adjudication_is_input_order_invariant() {
    assert_eq!(
        record(false).canonical_bytes(),
        record(true).canonical_bytes()
    );
}
