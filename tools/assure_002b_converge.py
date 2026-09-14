#!/usr/bin/env python3
from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    text, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text


# ASSURE-002F: expose already-committed generic evidence context without changing wire bytes.
core = Path("crates/core/symthaea-assurance-core/src/lib.rs")
text = core.read_text()
text = replace_once(
    text,
    """    pub fn evidence_id(&self) -> &StableId {
        &self.evidence_id
    }

    pub fn kind(&self) -> &EvidenceKind {
""",
    """    pub fn evidence_id(&self) -> &StableId {
        &self.evidence_id
    }

    pub fn subject_id(&self) -> &DigestSha256 {
        &self.subject_id
    }

    pub fn claim_digest(&self) -> &DigestSha256 {
        &self.claim_digest
    }

    pub fn kind(&self) -> &EvidenceKind {
""",
    "core evidence accessors",
)
core.write_text(text)

core_tests = Path("crates/core/symthaea-assurance-core/tests/assure000.rs")
text = core_tests.read_text()
if "fn evidence_context_accessors_preserve_qualified_wire_identity()" in text:
    raise SystemExit("core accessor regression already exists")
text += r'''

#[test]
fn evidence_context_accessors_preserve_qualified_wire_identity() {
    let subject = subject();
    let claim = claim(&subject);
    let observation = artifact(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let expected_subject = subject.subject_id();
    let expected_claim = claim.digest();
    let evidence_digest = observation.digest();

    assert_eq!(
        expected_subject.as_str(),
        "f1d6799d2aa3d718639a298f06fd7318efefd1b798d2e241f4dd154351f35786"
    );
    assert_eq!(
        expected_claim.as_str(),
        "3530e01aebb221303404012e1b490d556066fa0b029f0e723f2dea5e8b5ab8c8"
    );
    assert_eq!(observation.subject_id(), &expected_subject);
    assert_eq!(observation.claim_digest(), &expected_claim);
    assert_eq!(
        evidence_digest.as_str(),
        "802cebd6e4c713391f8aa73c262fc8f48cf0abe2f7fcf216f2c8d2d44c6bcc7f"
    );
}
'''
core_tests.write_text(text)

# Campaign crate now consumes shared semantic commitment authority.
campaign_toml = Path("crates/core/symthaea-assurance-campaign/Cargo.toml")
text = campaign_toml.read_text()
text = replace_once(
    text,
    'symthaea-assurance-core = { path = "../symthaea-assurance-core" }\n',
    'symthaea-assurance-core = { path = "../symthaea-assurance-core" }\n'
    'symthaea-assurance-semantics = { path = "../symthaea-assurance-semantics" }\n',
    "campaign semantic dependency",
)
campaign_toml.write_text(text)

campaign = Path("crates/core/symthaea-assurance-campaign/src/lib.rs")
text = campaign.read_text()

text = replace_once(
    text,
    "//! registration receipt exists\n//!     != registration is current\n",
    "//! unique terminal receipt in supplied view\n//!     != authoritative external currentness\n",
    "module currentness theorem",
)
text = replace_once(
    text,
    "use symthaea_assurance_subject::{AiSubjectManifest, SubjectError};\n",
    "use symthaea_assurance_semantics::{\n"
    "    SemanticCommitmentError, SemanticCommitmentV1, canonical_semantic_set,\n"
    "};\n"
    "use symthaea_assurance_subject::{AiSubjectManifest, SubjectError};\n",
    "shared semantic import",
)

error_replacements = {
    '#[error("the registration lineage has no current registration")]\n    NoCurrentRegistration,':
        '#[error("the supplied registration view has no terminal unwithdrawn registration")]\n    NoTerminalRegistrationInView,',
    '#[error("campaign plan does not match the current registration")]\n    CurrentPlanMismatch,':
        '#[error("campaign plan does not match the terminal registration in the supplied view")]\n    TerminalPlanMismatch,',
    '#[error("evidence timing statement does not bind the exact evidence/current registration")]\n    EvidenceProductionStatementMismatch,':
        '#[error("evidence commitment statement does not bind the exact evidence/terminal registration in the supplied view")]\n    EvidenceCommitmentStatementMismatch,',
    '#[error("evidence is not preregistered for the current plan: {0:?}")]\n    EvidenceNotPreregistered(EvidenceTimingClass),':
        '#[error("evidence commitment is not after the terminal registration in the supplied view: {0:?}")]\n    EvidenceCommitmentNotAfterRegistration(EvidenceCommitmentTimingClass),',
    '#[error("evidence admission ordering is not strictly later than production")]\n    AdmissionNotAfterProduction,':
        '#[error("evidence admission ordering is not strictly later than evidence commitment ordering")]\n    AdmissionNotAfterCommitment,',
}
for old, new in error_replacements.items():
    text = replace_once(text, old, new, old.splitlines()[-1])

text = replace_once(
    text,
    '    #[error("evidence kind is not registered by the current campaign plan")]\n'
    '    UnregisteredEvidenceKind,\n',
    '    #[error("evidence subject does not match the exact campaign subject")]\n'
    '    EvidenceSubjectMismatch,\n'
    '    #[error("evidence claim does not match the exact campaign claim")]\n'
    '    EvidenceClaimMismatch,\n'
    '    #[error("evidence kind is not registered by the terminal campaign plan in the supplied view")]\n'
    '    UnregisteredEvidenceKind,\n',
    "evidence context errors",
)
text = replace_once(
    text,
    '    #[error("duplicate evidence digest in the admitted campaign ledger")]\n'
    '    DuplicateEvidenceDigest,\n',
    '    #[error("duplicate evidence digest in the admitted campaign ledger")]\n'
    '    DuplicateEvidenceDigest,\n'
    '    #[error("evidence ordinal overflow")]\n'
    '    EvidenceOrdinalOverflow,\n',
    "ordinal overflow error",
)

# Delete the campaign-local semantic-definition type/implementation.
text = sub_once(
    text,
    r'\n#\[derive\(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash\)\]\n'
    r'pub struct SemanticCommitmentV1 \{.*?\n\}\n\n'
    r'impl SemanticCommitmentV1 \{.*?\n\}\n',
    "\n",
    "local semantic type deletion",
    re.S,
)

text = replace_once(
    text,
    '            Self::Custom(semantic) => EvidenceKind::Custom(semantic.semantic_id.clone()),',
    '            Self::Custom(semantic) => EvidenceKind::Custom(semantic.semantic_id().clone()),',
    "custom evidence core bridge",
)
text = sub_once(
    text,
    r'\n    fn definition_digest\(&self\) -> Option<&DigestSha256> \{.*?\n    \}\n',
    "\n",
    "local definition digest accessor",
    re.S,
)

# Shared semantic set canonicalizer is authoritative.
text = replace_once(
    text,
    '            controls: canonical_semantics("controls", controls)?,',
    '            controls: canonical_semantic_set(controls)\n'
    '                .map_err(|error| semantic_set_error("controls", error))?,',
    "controls shared canonicalizer",
)
text = replace_once(
    text,
    '            failure_conditions: canonical_semantics("failure-conditions", failure_conditions)?,',
    '            failure_conditions: canonical_semantic_set(failure_conditions)\n'
    '                .map_err(|error| semantic_set_error("failure-conditions", error))?,',
    "failure shared canonicalizer",
)
text = replace_once(
    text,
    """            contradiction_conditions: canonical_semantics(
                "contradiction-conditions",
                contradiction_conditions,
            )?,""",
    """            contradiction_conditions: canonical_semantic_set(contradiction_conditions)
                .map_err(|error| semantic_set_error("contradiction-conditions", error))?,""",
    "contradiction shared canonicalizer",
)
text = replace_once(
    text,
    """            inconclusive_conditions: canonical_semantics(
                "inconclusive-conditions",
                inconclusive_conditions,
            )?,""",
    """            inconclusive_conditions: canonical_semantic_set(inconclusive_conditions)
                .map_err(|error| semantic_set_error("inconclusive-conditions", error))?,""",
    "inconclusive shared canonicalizer",
)
text = replace_once(
    text,
    """            invalidation_conditions: canonical_semantics(
                "invalidation-conditions",
                invalidation_conditions,
            )?,""",
    """            invalidation_conditions: canonical_semantic_set(invalidation_conditions)
                .map_err(|error| semantic_set_error("invalidation-conditions", error))?,""",
    "invalidation shared canonicalizer",
)
text = text.replace("condition.semantic_id.clone()", "condition.semantic_id().clone()")

text = replace_once(
    text,
    """    pub fn subject_manifest_id(&self) -> &DigestSha256 {
        &self.subject_manifest_id
    }
""",
    """    pub fn subject_manifest_id(&self) -> &DigestSha256 {
        &self.subject_manifest_id
    }

    pub fn subject_core_id(&self) -> &DigestSha256 {
        &self.subject_core_id
    }
""",
    "campaign core subject accessor",
)

# Nested semantics are bound once by the shared commitment digest.
text = replace_once(
    text,
    """        field(
            &mut out,
            "validation-profile",
            self.validation_profile.semantic_id.as_str(),
        );
        field(
            &mut out,
            "validation-profile-definition",
            self.validation_profile.definition_digest.as_str(),
        );""",
    """        field(
            &mut out,
            "validation-profile-id",
            self.validation_profile.semantic_id().as_str(),
        );
        field(
            &mut out,
            "validation-profile-commitment",
            self.validation_profile.digest().as_str(),
        );""",
    "validation profile shared digest",
)

# Public API vocabulary narrows currentness and temporal provenance claims.
for old, new in (
    ("CurrentRegistrationV1", "TerminalRegistrationInViewV1"),
    ("resolve_current_registration", "resolve_terminal_registration_in_view"),
    ("NoCurrentRegistration", "NoTerminalRegistrationInView"),
    ("CurrentPlanMismatch", "TerminalPlanMismatch"),
    ("EvidenceTimingClass", "EvidenceCommitmentTimingClass"),
    ("evidence_production_statement_digest", "evidence_commitment_statement_digest"),
    ("classify_evidence_timing", "classify_evidence_commitment_timing"),
    ("AfterCurrentRegistration", "CommittedAfterTerminalRegistrationInView"),
    ("ProducedBeforeOrAtRegistration", "CommittedBeforeOrAtTerminalRegistrationInView"),
    ("EvidenceProductionStatementMismatch", "EvidenceCommitmentStatementMismatch"),
    ("EvidenceNotPreregistered", "EvidenceCommitmentNotAfterRegistration"),
    ("AdmissionNotAfterProduction", "AdmissionNotAfterCommitment"),
    ("evidence-production-statement-v1", "evidence-commitment-statement-v1"),
    ("production_ordering_digest", "commitment_ordering_digest"),
    ("production_ordering", "commitment_ordering"),
    ('"production-ordering"', '"commitment-ordering"'),
):
    text = text.replace(old, new)
text = re.sub(r"\bproduction\b", "commitment_ordering", text)

# Context binding precedes kind, duplicate, timing, admission, and root mutation checks.
context_anchor = """        if plan.digest() != self.plan_digest
            || plan.campaign_nonce() != &self.campaign_nonce
            || current.receipt.plan_digest() != &plan.digest()
        {
            return Err(CampaignError::TerminalPlanMismatch);
        }
"""
context_replacement = context_anchor + """        if evidence.subject_id() != plan.subject_core_id() {
            return Err(CampaignError::EvidenceSubjectMismatch);
        }
        if evidence.claim_digest() != plan.claim_digest() {
            return Err(CampaignError::EvidenceClaimMismatch);
        }
"""
text = replace_once(text, context_anchor, context_replacement, "evidence context admission guard")

# Checked ordinal is shared by statement identity and transition identity.
text = replace_once(
    text,
    """    ) -> Result<DigestSha256, CampaignError> {
        self.require_current(current)?;
        let mut out = String::from("symthaea-assurance-evidence-admission-statement-v1\\n");""",
    """    ) -> Result<DigestSha256, CampaignError> {
        self.require_current(current)?;
        let ordinal = checked_next_evidence_ordinal(self.admitted_count)?;
        let mut out = String::from("symthaea-assurance-evidence-admission-statement-v1\\n");""",
    "checked admission statement ordinal",
)
text = replace_once(
    text,
    '        field(&mut out, "ordinal", &(self.admitted_count + 1).to_string());',
    '        field(&mut out, "ordinal", &canonical_u64(ordinal));',
    "admission statement ordinal wire",
)
text = replace_once(
    text,
    "        let ordinal = self.admitted_count + 1;",
    "        let ordinal = checked_next_evidence_ordinal(self.admitted_count)?;",
    "checked admission transition ordinal",
)

# Explicit minimal unsigned decimal ASCII representation for public u64 wire fields.
text = text.replace(
    'field(&mut out, "epoch", &self.epoch.to_string());',
    'field(&mut out, "epoch", &canonical_u64(self.epoch));',
)
text = text.replace(
    'field(&mut out, "sequence", &self.sequence.to_string());',
    'field(&mut out, "sequence", &canonical_u64(self.sequence));',
)
text = text.replace(
    'field(&mut out, "ordinal", &self.ordinal.to_string());',
    'field(&mut out, "ordinal", &canonical_u64(self.ordinal));',
)
text = text.replace(
    'field(&mut out, "ordinal", &ordinal.to_string());',
    'field(&mut out, "ordinal", &canonical_u64(ordinal));',
)

# Replace implicit Rust enum ordering and delete the local semantic-set canonicalizer.
text = sub_once(
    text,
    r"fn canonical_support_criteria\(.*?\nfn empty_evidence_root\(",
    r'''fn canonical_support_criteria(
    mut criteria: Vec<SupportCriterionV1>,
    maximum_support: SupportTier,
) -> Result<Vec<SupportCriterionV1>, CampaignError> {
    for criterion in &criteria {
        if support_tier_rank(criterion.tier) > support_tier_rank(maximum_support) {
            return Err(CampaignError::SupportCriterionAboveCeiling);
        }
    }
    criteria.sort_by(|left, right| {
        support_tier_rank(left.tier)
            .cmp(&support_tier_rank(right.tier))
            .then_with(|| left.criterion.semantic_id().cmp(right.criterion.semantic_id()))
    });
    let mut ids = BTreeSet::new();
    for criterion in &criteria {
        if !ids.insert(criterion.criterion.semantic_id().clone()) {
            return Err(CampaignError::DuplicateSemanticId {
                set: "support-criteria",
                id: criterion.criterion.semantic_id().as_str().to_owned(),
            });
        }
    }
    Ok(criteria)
}

fn semantic_set_error(set: &'static str, error: SemanticCommitmentError) -> CampaignError {
    match error {
        SemanticCommitmentError::DuplicateSemanticId(id) => CampaignError::DuplicateSemanticId { set, id },
    }
}

fn checked_next_evidence_ordinal(current: u64) -> Result<u64, CampaignError> {
    current.checked_add(1).ok_or(CampaignError::EvidenceOrdinalOverflow)
}

fn canonical_u64(value: u64) -> String {
    value.to_string()
}

fn empty_evidence_root(''',
    "support/semantic helper convergence",
    re.S,
)

# Campaign wire binds slot identity plus shared commitment digest, never a local definition tuple.
text = sub_once(
    text,
    r"fn append_evidence_requirements\(.*?\nfn support_tier_name\(",
    r'''fn append_evidence_requirements(out: &mut String, requirements: &[EvidenceRequirementV1]) {
    field(out, "evidence-kind-count", &requirements.len().to_string());
    for requirement in requirements {
        field(out, "evidence-kind", &requirement.canonical_name());
        match requirement {
            EvidenceRequirementV1::Builtin(_) => field(out, "evidence-kind-semantic-commitment", ""),
            EvidenceRequirementV1::Custom(semantic) => field(
                out,
                "evidence-kind-semantic-commitment",
                semantic.digest().as_str(),
            ),
        }
    }
}

fn append_support_criteria(out: &mut String, criteria: &[SupportCriterionV1]) {
    field(out, "support-criterion-count", &criteria.len().to_string());
    for criterion in criteria {
        field(out, "support-criterion-tier", support_tier_name(criterion.tier));
        field(out, "support-criterion-id", criterion.criterion.semantic_id().as_str());
        field(
            out,
            "support-criterion-commitment",
            criterion.criterion.digest().as_str(),
        );
    }
}

fn append_semantics(out: &mut String, label: &str, values: &[SemanticCommitmentV1]) {
    field(out, &format!("{label}-count"), &values.len().to_string());
    for value in values {
        field(out, &format!("{label}-id"), value.semantic_id().as_str());
        field(out, &format!("{label}-commitment"), value.digest().as_str());
    }
}

fn support_tier_rank(tier: SupportTier) -> u8 {
    match tier {
        SupportTier::Structural => 0,
        SupportTier::Observed => 1,
        SupportTier::CausallySupported => 2,
        SupportTier::FunctionallySupported => 3,
    }
}

fn support_tier_name(''',
    "campaign semantic wire convergence",
    re.S,
)

if "fn evidence_ordinal_overflow_fails_closed()" in text:
    raise SystemExit("internal convergence tests already exist")
text += r'''

#[cfg(test)]
mod convergence_tests {
    use super::*;

    #[test]
    fn evidence_ordinal_overflow_fails_closed() {
        assert_eq!(
            checked_next_evidence_ordinal(u64::MAX),
            Err(CampaignError::EvidenceOrdinalOverflow)
        );
    }

    #[test]
    fn canonical_u64_is_minimal_unsigned_decimal_ascii() {
        assert_eq!(canonical_u64(0), "0");
        assert_eq!(canonical_u64(7), "7");
        assert_eq!(canonical_u64(u64::MAX), "18446744073709551615");
    }
}
'''
campaign.write_text(text)

# Campaign integration tests migrate to shared semantics and bounded theorem vocabulary.
tests = Path("crates/core/symthaea-assurance-campaign/tests/assure002.rs")
text = tests.read_text()
text = replace_once(
    text,
    "    RegistrationWithdrawalV1, ReproductionRequirementV1, SemanticCommitmentV1, SupportCriterionV1,\n",
    "    RegistrationWithdrawalV1, ReproductionRequirementV1, SupportCriterionV1,\n",
    "campaign test local semantic import",
)
text = replace_once(
    text,
    "use symthaea_assurance_subject::{\n",
    "use symthaea_assurance_semantics::{DefinitionSchemaV1, SemanticCommitmentV1};\n"
    "use symthaea_assurance_subject::{\n",
    "campaign test shared semantic import",
)
text = replace_once(
    text,
    """fn semantic(name: &str, byte: char) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(id(name), digest(byte))
}
""",
    """fn semantic_with_schema(
    name: &str,
    schema_id: &str,
    specification_byte: char,
    definition_byte: char,
) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
        id(name),
        DefinitionSchemaV1::new(id(schema_id), digest(specification_byte)),
        digest(definition_byte),
    )
}

fn semantic(name: &str, byte: char) -> SemanticCommitmentV1 {
    semantic_with_schema(
        name,
        "symthaea.assurance.test-semantic-schema.v1",
        'e',
        byte,
    )
}
""",
    "campaign test semantic helper",
)
text = replace_once(
    text,
    """fn plan_with_control_definition(
    subject: &AiSubjectManifest,
    campaign_nonce: &str,
    control_definition: char,
) -> CampaignPlanV1 {""",
    """fn plan_with_control_semantic(
    subject: &AiSubjectManifest,
    campaign_nonce: &str,
    control: SemanticCommitmentV1,
) -> CampaignPlanV1 {""",
    "control semantic helper signature",
)
text = replace_once(
    text,
    '            semantic("denied-path", control_definition),',
    "            control,",
    "control semantic insertion",
)
text = replace_once(
    text,
    "fn plan(subject: &AiSubjectManifest, campaign_nonce: &str) -> CampaignPlanV1 {\n",
    """fn plan_with_control_definition(
    subject: &AiSubjectManifest,
    campaign_nonce: &str,
    control_definition: char,
) -> CampaignPlanV1 {
    plan_with_control_semantic(
        subject,
        campaign_nonce,
        semantic("denied-path", control_definition),
    )
}

fn plan(subject: &AiSubjectManifest, campaign_nonce: &str) -> CampaignPlanV1 {
""",
    "control definition wrapper",
)

for old, new in (
    ("EvidenceTimingClass", "EvidenceCommitmentTimingClass"),
    ("classify_evidence_timing", "classify_evidence_commitment_timing"),
    ("evidence_production_statement_digest", "evidence_commitment_statement_digest"),
    ("resolve_current_registration", "resolve_terminal_registration_in_view"),
    ("AfterCurrentRegistration", "CommittedAfterTerminalRegistrationInView"),
    ("ProducedBeforeOrAtRegistration", "CommittedBeforeOrAtTerminalRegistrationInView"),
    ("AdmissionNotAfterProduction", "AdmissionNotAfterCommitment"),
    ("NoCurrentRegistration", "NoTerminalRegistrationInView"),
    ("CurrentPlanMismatch", "TerminalPlanMismatch"),
):
    text = text.replace(old, new)
# Rename standalone variable tokens only; never rewrite ReproductionRequirementV1.
text = re.sub(r"\bproduction\b", "commitment", text)

if "fn semantic_schema_id_drift_changes_richer_plan_identity()" in text:
    raise SystemExit("campaign convergence regression block already exists")
text += r'''

#[test]
fn semantic_schema_id_drift_changes_richer_plan_identity() {
    let subject = subject('a');
    let original = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-a", 'e', '4'),
    );
    let changed = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-b", 'e', '4'),
    );
    assert_ne!(original.digest(), changed.digest());
    assert_eq!(original.core_plan().digest(), changed.core_plan().digest());
}

#[test]
fn semantic_schema_specification_drift_changes_richer_plan_identity() {
    let subject = subject('a');
    let original = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-a", 'e', '4'),
    );
    let changed = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-a", 'd', '4'),
    );
    assert_ne!(original.digest(), changed.digest());
    assert_eq!(original.core_plan().digest(), changed.core_plan().digest());
}

#[test]
fn duplicate_semantic_ids_fail_closed_across_schema_drift() {
    let subject = subject('a');
    let claim = claim(&subject);
    let error = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![req(EvidenceKind::Observation)],
        vec![],
        vec![
            semantic_with_schema("same-control", "schema-a", 'e', '1'),
            semantic_with_schema("same-control", "schema-b", 'e', '1'),
        ],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CampaignError::DuplicateSemanticId {
            set: "controls",
            ..
        }
    ));
}

#[test]
fn validation_profile_schema_id_drift_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let changed = OrderingReceiptV1::new(
        id("transparency-log-a"),
        semantic_with_schema(
            "monotonic-ordering-profile-v1",
            "different-schema",
            'e',
            'f',
        ),
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &evidence),
        digest('b'),
    )
    .unwrap();
    assert_eq!(
        classify_evidence_commitment_timing(&terminal, &evidence, &changed).unwrap(),
        EvidenceCommitmentTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn validation_profile_schema_specification_drift_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let changed = OrderingReceiptV1::new(
        id("transparency-log-a"),
        semantic_with_schema(
            "monotonic-ordering-profile-v1",
            "symthaea.assurance.test-semantic-schema.v1",
            '0',
            'f',
        ),
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &evidence),
        digest('b'),
    )
    .unwrap();
    assert_eq!(
        classify_evidence_commitment_timing(&terminal, &evidence, &changed).unwrap(),
        EvidenceCommitmentTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn foreign_subject_evidence_fails_before_ledger_mutation() {
    let subject = subject('a');
    let foreign_subject = subject('b');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&terminal);
    let initial_root = ledger.evidence_root().clone();
    let foreign = evidence(
        &foreign_subject,
        EvidenceKind::Observation,
        "context-id",
        'c',
    );
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &foreign),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&terminal, &foreign, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &terminal, &foreign, &commitment, &admission)
            .unwrap_err(),
        CampaignError::EvidenceSubjectMismatch
    );
    assert_eq!(ledger.admitted_count(), 0);
    assert_eq!(ledger.evidence_root(), &initial_root);

    let valid = evidence(&subject, EvidenceKind::Observation, "context-id", 'd');
    let valid_commitment = ordering(
        "transparency-log-a",
        1,
        8,
        evidence_commitment_statement_digest(&terminal, &valid),
        '3',
    );
    let valid_admission_statement = ledger
        .admission_statement_digest(&terminal, &valid, &valid_commitment)
        .unwrap();
    let valid_admission = ordering(
        "transparency-log-a",
        1,
        9,
        valid_admission_statement,
        '4',
    );
    ledger
        .admit_preregistered(
            &plan,
            &terminal,
            &valid,
            &valid_commitment,
            &valid_admission,
        )
        .unwrap();
    assert_eq!(ledger.admitted_count(), 1);
}

#[test]
fn foreign_claim_evidence_fails_before_ledger_mutation() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&terminal);
    let initial_root = ledger.evidence_root().clone();
    let foreign_claim = Claim::new(
        id("foreign-claim"),
        subject.core_subject_id().unwrap(),
        "foreign proposition",
        id("agent-authority"),
    )
    .unwrap();
    let foreign = EvidenceArtifact::new(
        id("foreign-claim-evidence"),
        subject.core_subject_id().unwrap(),
        foreign_claim.digest(),
        EvidenceKind::Observation,
        digest('c'),
        EvidenceProvenance::new(
            id("producer"),
            id("executor"),
            Some(id("verifier")),
            None,
        ),
    );
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &foreign),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&terminal, &foreign, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &terminal, &foreign, &commitment, &admission)
            .unwrap_err(),
        CampaignError::EvidenceClaimMismatch
    );
    assert_eq!(ledger.admitted_count(), 0);
    assert_eq!(ledger.evidence_root(), &initial_root);
}
'''
tests.write_text(text)

# Root lock projection: no external dependency changes permitted.
lock = Path("Cargo.lock")
text = lock.read_text()
semantic_lock = Path("/tmp/semantics-parent.Cargo.lock").read_text()


def stanza(source: str, package: str) -> str:
    pattern = re.compile(
        r'\[\[package\]\]\nname = "' + re.escape(package) + r'"\n.*?(?=\n\[\[package\]\]|\Z)',
        re.S,
    )
    match = pattern.search(source)
    if not match:
        raise SystemExit(f"missing lock stanza for {package}")
    return match.group(0)


campaign_stanza = stanza(text, "symthaea-assurance-campaign")
if ' "symthaea-assurance-semantics",' in campaign_stanza:
    raise SystemExit("campaign lock already contains semantics dependency")
new_campaign_stanza = campaign_stanza.replace(
    ' "symthaea-assurance-core",\n',
    ' "symthaea-assurance-core",\n "symthaea-assurance-semantics",\n',
    1,
)
if new_campaign_stanza == campaign_stanza:
    raise SystemExit("campaign lock dependency insertion anchor missing")
text = text.replace(campaign_stanza, new_campaign_stanza, 1)

if 'name = "symthaea-assurance-semantics"' in text:
    raise SystemExit("semantics lock stanza unexpectedly already present")
core_stanza = stanza(text, "symthaea-assurance-core")
semantics_stanza = stanza(semantic_lock, "symthaea-assurance-semantics")
text = text.replace(core_stanza, core_stanza + "\n" + semantics_stanza, 1)
lock.write_text(text)
