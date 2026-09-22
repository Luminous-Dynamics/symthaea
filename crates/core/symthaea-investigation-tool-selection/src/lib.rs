// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, zero-I/O methodology selection for Symthaea OSINT investigations.
//!
//! Tool profiles are projected observation contracts. Selecting one never creates connector,
//! credential, network, browser, OPSEC, lease, persistence, or action authority.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use symthaea_investigation::{ProfileRef, SearchPlanAuthorityScopeV1, ToolProfileRef};

pub const TOOL_SELECTION_PROFILE_V1: &str = "symthaea:osint-tool-selection:pareto:v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolCapabilityV1 {
    WebSearch,
    ScientificLiteratureSearch,
    MediaProvenanceInspection,
    BrowserRenderedCapture,
    WebArchiveLookup,
    CodeForgeSearch,
    CtiFeedLookup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolViewClassV1 {
    AnonymousPublicView,
    AuthenticatedAccountView,
    DeclaredProviderCorpus,
    LocalArtifactView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolTransformV1 {
    ProviderIndexing,
    SnippetGeneration,
    PersonalizedRanking,
    StructuredMetadataProjection,
    StructuredApiProjection,
    ArchiveIndexProjection,
    DomConstruction,
    JavascriptExecution,
    RenderedPixels,
    TextExtraction,
    MetadataParsing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DisclosureSurfaceV1 {
    QueryTerms,
    TargetIdentifiers,
    UploadedMedia,
    DnsTransportMetadata,
    HeadersBody,
    IpNetworkOrigin,
    ProviderAnalytics,
    StoredSearchHistory,
    AccountProviderIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SideEffectClassV1 {
    ObservationOnly,
    ReadWithRemoteDisclosure,
    StateCreating,
}

impl SideEffectClassV1 {
    fn burden_rank(self) -> usize {
        match self {
            Self::ObservationOnly => 0,
            Self::ReadWithRemoteDisclosure => 1,
            Self::StateCreating => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCurrentnessV1 {
    CurrentQualified,
    HistoricalKnownProfile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverageClassV1 {
    TopKOnly { result_cap: u32 },
    PersonalizedView { result_cap: u32 },
    ExactFiniteCorpus { corpus_commitment: ProfileRef },
    BestEffortSearch,
    TimeBoundWindow { window_ref: ProfileRef },
    DeclaredProviderCorpus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolProfileProjectionV1 {
    pub profile_ref: ToolProfileRef,
    pub capabilities: Vec<ToolCapabilityV1>,
    pub coverage: CoverageClassV1,
    pub view: ToolViewClassV1,
    /// Preserved for audit but deliberately ignored by methodology preference.
    pub provider_ranking_ref: ProfileRef,
    pub transforms: Vec<ToolTransformV1>,
    pub disclosure_surfaces: Vec<DisclosureSurfaceV1>,
    pub side_effect: SideEffectClassV1,
    pub currentness: ToolCurrentnessV1,
    pub limitations: Vec<ProfileRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageRequirementV1 {
    BestEffortOrBetter,
    ExactFiniteCorpus,
    BestEffortSearch,
    WorldComplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurrentnessRequirementV1 {
    Any,
    CurrentQualified,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSelectionTaskV1 {
    pub required_capabilities: Vec<ToolCapabilityV1>,
    pub coverage_requirement: CoverageRequirementV1,
    pub required_corpus_commitment: Option<ProfileRef>,
    pub currentness_requirement: CurrentnessRequirementV1,
    pub allowed_views: Vec<ToolViewClassV1>,
    pub forbidden_views: Vec<ToolViewClassV1>,
    pub forbidden_disclosure_surfaces: Vec<DisclosureSurfaceV1>,
    pub observation_only_required: bool,
    pub required_transforms: Vec<ToolTransformV1>,
    pub external_authorization_review_required: bool,
}

#[derive(Clone, PartialEq, Eq)]
pub enum ToolSelectionError {
    EmptyRequiredCapabilities,
    DuplicateProfileRef(String),
}

impl fmt::Display for ToolSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRequiredCapabilities => {
                write!(f, "tool selection requires at least one capability")
            }
            Self::DuplicateProfileRef(_) => write!(f, "duplicate tool profile ref: <redacted>"),
        }
    }
}

impl fmt::Debug for ToolSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolSelectionError({self})")
    }
}

impl Error for ToolSelectionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IneligibilityReasonV1 {
    IneligibleCapability,
    CoverageInsufficient,
    CorpusCommitmentMismatch,
    CurrentnessMismatch,
    BlockedViewClass,
    ForbiddenDisclosureSurface,
    SideEffectNotObservationOnly,
    MissingRequiredTransform,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedToolProfileV1 {
    pub profile_ref: ToolProfileRef,
    pub reasons: Vec<IneligibilityReasonV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMethodDispositionV1 {
    EligibleMethodFit,
    NeedsExternalAuthorization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibleToolProfileV1 {
    pub profile_ref: ToolProfileRef,
    pub disposition: ToolMethodDispositionV1,
    pub carried_limitations: Vec<ProfileRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnmetToolRequirementV1 {
    NoEligibleToolProfile,
    NoToolProfileCanEstablishRequestedCoverage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSelectionSummaryV1 {
    pub eligible_profiles: Vec<EligibleToolProfileV1>,
    pub blocked_profiles: Vec<BlockedToolProfileV1>,
    pub partial_evidence_profiles: Vec<ToolProfileRef>,
    pub pareto_front_profiles: Vec<ToolProfileRef>,
    pub unmet_requirements: Vec<UnmetToolRequirementV1>,
}

impl ToolSelectionSummaryV1 {
    pub fn authority_scope(&self) -> SearchPlanAuthorityScopeV1 {
        SearchPlanAuthorityScopeV1::ProposalOnly
    }
}

pub fn select_tool_profiles(
    task: &ToolSelectionTaskV1,
    profiles: Vec<ToolProfileProjectionV1>,
) -> Result<ToolSelectionSummaryV1, ToolSelectionError> {
    let required_capabilities: BTreeSet<_> = task.required_capabilities.iter().copied().collect();
    if required_capabilities.is_empty() {
        return Err(ToolSelectionError::EmptyRequiredCapabilities);
    }

    let mut seen_profile_refs = BTreeSet::new();
    for profile in &profiles {
        if !seen_profile_refs.insert(profile.profile_ref.clone()) {
            return Err(ToolSelectionError::DuplicateProfileRef(
                profile.profile_ref.as_str().to_string(),
            ));
        }
    }

    let required_transforms: BTreeSet<_> = task.required_transforms.iter().copied().collect();
    let forbidden_views: BTreeSet<_> = task.forbidden_views.iter().copied().collect();
    let allowed_views: BTreeSet<_> = task.allowed_views.iter().copied().collect();
    let forbidden_surfaces: BTreeSet<_> =
        task.forbidden_disclosure_surfaces.iter().copied().collect();

    let mut eligible_raw = Vec::new();
    let mut blocked_profiles = Vec::new();
    let mut partial_evidence_profiles = BTreeSet::new();

    for profile in profiles {
        let capabilities: BTreeSet<_> = profile.capabilities.iter().copied().collect();
        let transforms: BTreeSet<_> = profile.transforms.iter().copied().collect();
        let disclosure_surfaces: BTreeSet<_> =
            profile.disclosure_surfaces.iter().copied().collect();
        let mut reasons = BTreeSet::new();

        if !required_capabilities.is_subset(&capabilities) {
            reasons.insert(IneligibilityReasonV1::IneligibleCapability);
        }

        let coverage_ok = coverage_satisfies(task, &profile.coverage);
        if !coverage_ok {
            let reason = if task.coverage_requirement == CoverageRequirementV1::ExactFiniteCorpus
                && matches!(profile.coverage, CoverageClassV1::ExactFiniteCorpus { .. })
            {
                IneligibilityReasonV1::CorpusCommitmentMismatch
            } else {
                IneligibilityReasonV1::CoverageInsufficient
            };
            reasons.insert(reason);
        }

        if task.currentness_requirement == CurrentnessRequirementV1::CurrentQualified
            && profile.currentness != ToolCurrentnessV1::CurrentQualified
        {
            reasons.insert(IneligibilityReasonV1::CurrentnessMismatch);
        }

        if forbidden_views.contains(&profile.view)
            || (!allowed_views.is_empty() && !allowed_views.contains(&profile.view))
        {
            reasons.insert(IneligibilityReasonV1::BlockedViewClass);
        }

        if disclosure_surfaces
            .iter()
            .any(|surface| forbidden_surfaces.contains(surface))
        {
            reasons.insert(IneligibilityReasonV1::ForbiddenDisclosureSurface);
        }

        if task.observation_only_required && profile.side_effect != SideEffectClassV1::ObservationOnly
        {
            reasons.insert(IneligibilityReasonV1::SideEffectNotObservationOnly);
        }

        if !required_transforms.is_subset(&transforms) {
            reasons.insert(IneligibilityReasonV1::MissingRequiredTransform);
        }

        let capability_ok = required_capabilities.is_subset(&capabilities);
        let noncoverage_reasons = reasons
            .iter()
            .filter(|reason| {
                !matches!(
                    reason,
                    IneligibilityReasonV1::CoverageInsufficient
                        | IneligibilityReasonV1::CorpusCommitmentMismatch
                )
            })
            .count();
        if capability_ok && !coverage_ok && noncoverage_reasons == 0 {
            partial_evidence_profiles.insert(profile.profile_ref.clone());
        }

        if reasons.is_empty() {
            eligible_raw.push(profile);
        } else {
            blocked_profiles.push(BlockedToolProfileV1 {
                profile_ref: profile.profile_ref,
                reasons: reasons.into_iter().collect(),
            });
        }
    }

    blocked_profiles.sort_by(|a, b| a.profile_ref.as_str().cmp(b.profile_ref.as_str()));
    let pareto_front_profiles = methodology_pareto_front(task, &eligible_raw);

    let mut eligible_profiles: Vec<_> = eligible_raw
        .into_iter()
        .map(|profile| {
            let disposition = if task.external_authorization_review_required
                && (!profile.disclosure_surfaces.is_empty()
                    || profile.side_effect != SideEffectClassV1::ObservationOnly)
            {
                ToolMethodDispositionV1::NeedsExternalAuthorization
            } else {
                ToolMethodDispositionV1::EligibleMethodFit
            };
            let mut limitations = profile.limitations;
            limitations.sort_by(|a, b| a.as_str().cmp(b.as_str()));
            EligibleToolProfileV1 {
                profile_ref: profile.profile_ref,
                disposition,
                carried_limitations: limitations,
            }
        })
        .collect();
    eligible_profiles.sort_by(|a, b| a.profile_ref.as_str().cmp(b.profile_ref.as_str()));

    let mut unmet_requirements = Vec::new();
    if eligible_profiles.is_empty() {
        unmet_requirements.push(UnmetToolRequirementV1::NoEligibleToolProfile);
        if task.coverage_requirement == CoverageRequirementV1::WorldComplete {
            unmet_requirements
                .push(UnmetToolRequirementV1::NoToolProfileCanEstablishRequestedCoverage);
        }
    }

    Ok(ToolSelectionSummaryV1 {
        eligible_profiles,
        blocked_profiles,
        partial_evidence_profiles: partial_evidence_profiles.into_iter().collect(),
        pareto_front_profiles,
        unmet_requirements,
    })
}

fn coverage_satisfies(task: &ToolSelectionTaskV1, coverage: &CoverageClassV1) -> bool {
    match task.coverage_requirement {
        CoverageRequirementV1::BestEffortOrBetter => true,
        CoverageRequirementV1::BestEffortSearch => {
            matches!(coverage, CoverageClassV1::BestEffortSearch)
        }
        CoverageRequirementV1::WorldComplete => false,
        CoverageRequirementV1::ExactFiniteCorpus => match coverage {
            CoverageClassV1::ExactFiniteCorpus { corpus_commitment } => {
                match task.required_corpus_commitment.as_ref() {
                    Some(required) => required == corpus_commitment,
                    None => true,
                }
            }
            _ => false,
        },
    }
}

fn methodology_pareto_front(
    task: &ToolSelectionTaskV1,
    profiles: &[ToolProfileProjectionV1],
) -> Vec<ToolProfileRef> {
    let required_transforms: BTreeSet<_> = task.required_transforms.iter().copied().collect();
    let mut front = Vec::new();

    for candidate in profiles {
        let dominated = profiles.iter().any(|other| {
            candidate.profile_ref != other.profile_ref
                && methodology_dominates(&required_transforms, other, candidate)
        });
        if !dominated {
            front.push(candidate.profile_ref.clone());
        }
    }

    front.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    front
}

fn methodology_dominates(
    required_transforms: &BTreeSet<ToolTransformV1>,
    a: &ToolProfileProjectionV1,
    b: &ToolProfileProjectionV1,
) -> bool {
    let a_disclosures = a.disclosure_surfaces.iter().copied().collect::<BTreeSet<_>>().len();
    let b_disclosures = b.disclosure_surfaces.iter().copied().collect::<BTreeSet<_>>().len();
    let a_limitations = a
        .limitations
        .iter()
        .map(ProfileRef::as_str)
        .collect::<BTreeSet<_>>()
        .len();
    let b_limitations = b
        .limitations
        .iter()
        .map(ProfileRef::as_str)
        .collect::<BTreeSet<_>>()
        .len();
    let a_surplus = a
        .transforms
        .iter()
        .filter(|t| !required_transforms.contains(t))
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let b_surplus = b
        .transforms
        .iter()
        .filter(|t| !required_transforms.contains(t))
        .copied()
        .collect::<BTreeSet<_>>()
        .len();

    let coordinates = [
        (a_disclosures, b_disclosures),
        (a_limitations, b_limitations),
        (a.side_effect.burden_rank(), b.side_effect.burden_rank()),
        (a_surplus, b_surplus),
    ];

    coordinates.iter().all(|(x, y)| x <= y) && coordinates.iter().any(|(x, y)| x < y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(value: &str) -> ProfileRef {
        ProfileRef::new(value).unwrap()
    }

    fn t(value: &str) -> ToolProfileRef {
        ToolProfileRef::new(value).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn profile(
        id: &str,
        capability: ToolCapabilityV1,
        coverage: CoverageClassV1,
        view: ToolViewClassV1,
        transforms: Vec<ToolTransformV1>,
        surfaces: Vec<DisclosureSurfaceV1>,
        side_effect: SideEffectClassV1,
        currentness: ToolCurrentnessV1,
        limitations: &[&str],
    ) -> ToolProfileProjectionV1 {
        ToolProfileProjectionV1 {
            profile_ref: t(id),
            capabilities: vec![capability],
            coverage,
            view,
            provider_ranking_ref: p("ranking:synthetic"),
            transforms,
            disclosure_surfaces: surfaces,
            side_effect,
            currentness,
            limitations: limitations.iter().map(|x| p(x)).collect(),
        }
    }

    fn corpus_profiles() -> Vec<ToolProfileProjectionV1> {
        vec![
            profile(
                "T_WEB_PUBLIC_TOPK",
                ToolCapabilityV1::WebSearch,
                CoverageClassV1::TopKOnly { result_cap: 10 },
                ToolViewClassV1::AnonymousPublicView,
                vec![ToolTransformV1::ProviderIndexing, ToolTransformV1::SnippetGeneration],
                vec![DisclosureSurfaceV1::QueryTerms, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::IpNetworkOrigin, DisclosureSurfaceV1::ProviderAnalytics],
                SideEffectClassV1::ReadWithRemoteDisclosure,
                ToolCurrentnessV1::CurrentQualified,
                &["limitation:ResultCap", "limitation:ProviderRankingOpaque", "limitation:CoverageUnknown"],
            ),
            profile(
                "T_WEB_AUTH_VIEW",
                ToolCapabilityV1::WebSearch,
                CoverageClassV1::PersonalizedView { result_cap: 10 },
                ToolViewClassV1::AuthenticatedAccountView,
                vec![ToolTransformV1::ProviderIndexing, ToolTransformV1::SnippetGeneration, ToolTransformV1::PersonalizedRanking],
                vec![DisclosureSurfaceV1::QueryTerms, DisclosureSurfaceV1::AccountProviderIdentity, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::IpNetworkOrigin, DisclosureSurfaceV1::StoredSearchHistory, DisclosureSurfaceV1::ProviderAnalytics],
                SideEffectClassV1::ReadWithRemoteDisclosure,
                ToolCurrentnessV1::CurrentQualified,
                &["limitation:AuthenticationRequired", "limitation:PersonalizationPossible", "limitation:ResultCap", "limitation:ProviderRankingOpaque"],
            ),
            profile(
                "T_SCI_FINITE",
                ToolCapabilityV1::ScientificLiteratureSearch,
                CoverageClassV1::ExactFiniteCorpus { corpus_commitment: p("sha256:synthetic-science-corpus-v1") },
                ToolViewClassV1::DeclaredProviderCorpus,
                vec![ToolTransformV1::StructuredMetadataProjection],
                vec![],
                SideEffectClassV1::ObservationOnly,
                ToolCurrentnessV1::CurrentQualified,
                &[],
            ),
            profile(
                "T_CODE_BEST_EFFORT",
                ToolCapabilityV1::CodeForgeSearch,
                CoverageClassV1::BestEffortSearch,
                ToolViewClassV1::AnonymousPublicView,
                vec![ToolTransformV1::ProviderIndexing, ToolTransformV1::SnippetGeneration],
                vec![DisclosureSurfaceV1::QueryTerms, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::IpNetworkOrigin],
                SideEffectClassV1::ReadWithRemoteDisclosure,
                ToolCurrentnessV1::CurrentQualified,
                &["limitation:CoverageUnknown"],
            ),
            profile(
                "T_CTI_WINDOW",
                ToolCapabilityV1::CtiFeedLookup,
                CoverageClassV1::TimeBoundWindow { window_ref: p("window:last-30-days") },
                ToolViewClassV1::DeclaredProviderCorpus,
                vec![ToolTransformV1::StructuredApiProjection],
                vec![DisclosureSurfaceV1::QueryTerms, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::IpNetworkOrigin],
                SideEffectClassV1::ReadWithRemoteDisclosure,
                ToolCurrentnessV1::CurrentQualified,
                &["limitation:TemporalCoverageLimit"],
            ),
            profile(
                "T_ARCHIVE_PARTIAL",
                ToolCapabilityV1::WebArchiveLookup,
                CoverageClassV1::DeclaredProviderCorpus,
                ToolViewClassV1::AnonymousPublicView,
                vec![ToolTransformV1::ArchiveIndexProjection],
                vec![DisclosureSurfaceV1::TargetIdentifiers, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::IpNetworkOrigin],
                SideEffectClassV1::ReadWithRemoteDisclosure,
                ToolCurrentnessV1::HistoricalKnownProfile,
                &["limitation:ArchiveIncomplete", "limitation:CoverageUnknown"],
            ),
            profile(
                "T_BROWSER_CAPTURE",
                ToolCapabilityV1::BrowserRenderedCapture,
                CoverageClassV1::BestEffortSearch,
                ToolViewClassV1::AnonymousPublicView,
                vec![ToolTransformV1::DomConstruction, ToolTransformV1::JavascriptExecution, ToolTransformV1::RenderedPixels, ToolTransformV1::TextExtraction],
                vec![DisclosureSurfaceV1::TargetIdentifiers, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::HeadersBody, DisclosureSurfaceV1::IpNetworkOrigin, DisclosureSurfaceV1::ProviderAnalytics],
                SideEffectClassV1::ReadWithRemoteDisclosure,
                ToolCurrentnessV1::CurrentQualified,
                &["limitation:CoverageUnknown", "limitation:MetadataStripped"],
            ),
            profile(
                "T_MEDIA_PROV",
                ToolCapabilityV1::MediaProvenanceInspection,
                CoverageClassV1::ExactFiniteCorpus { corpus_commitment: p("sha256:single-input-artifact") },
                ToolViewClassV1::LocalArtifactView,
                vec![ToolTransformV1::MetadataParsing],
                vec![],
                SideEffectClassV1::ObservationOnly,
                ToolCurrentnessV1::CurrentQualified,
                &[],
            ),
        ]
    }

    fn ids(summary: &ToolSelectionSummaryV1) -> Vec<&str> {
        summary.eligible_profiles.iter().map(|x| x.profile_ref.as_str()).collect()
    }

    fn blocked(summary: &ToolSelectionSummaryV1, id: &str, reason: IneligibilityReasonV1) -> bool {
        summary.blocked_profiles.iter().any(|entry| entry.profile_ref.as_str() == id && entry.reasons.contains(&reason))
    }

    fn public_web_task() -> ToolSelectionTaskV1 {
        ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::WebSearch],
            coverage_requirement: CoverageRequirementV1::BestEffortOrBetter,
            required_corpus_commitment: None,
            currentness_requirement: CurrentnessRequirementV1::CurrentQualified,
            allowed_views: vec![ToolViewClassV1::AnonymousPublicView],
            forbidden_views: vec![ToolViewClassV1::AuthenticatedAccountView],
            forbidden_disclosure_surfaces: vec![],
            observation_only_required: false,
            required_transforms: vec![],
            external_authorization_review_required: false,
        }
    }

    #[test]
    fn public_web_selects_anonymous_and_blocks_authenticated_view() {
        let summary = select_tool_profiles(&public_web_task(), corpus_profiles()).unwrap();
        assert_eq!(ids(&summary), vec!["T_WEB_PUBLIC_TOPK"]);
        assert!(blocked(&summary, "T_WEB_AUTH_VIEW", IneligibilityReasonV1::BlockedViewClass));
        assert_eq!(summary.pareto_front_profiles[0].as_str(), "T_WEB_PUBLIC_TOPK");
        assert_eq!(summary.authority_scope(), SearchPlanAuthorityScopeV1::ProposalOnly);
    }

    #[test]
    fn exact_finite_science_corpus_selects_only_exact_profile() {
        let task = ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::ScientificLiteratureSearch],
            coverage_requirement: CoverageRequirementV1::ExactFiniteCorpus,
            required_corpus_commitment: Some(p("sha256:synthetic-science-corpus-v1")),
            currentness_requirement: CurrentnessRequirementV1::CurrentQualified,
            allowed_views: vec![ToolViewClassV1::DeclaredProviderCorpus],
            forbidden_views: vec![], forbidden_disclosure_surfaces: vec![], observation_only_required: true,
            required_transforms: vec![], external_authorization_review_required: false,
        };
        let summary = select_tool_profiles(&task, corpus_profiles()).unwrap();
        assert_eq!(ids(&summary), vec!["T_SCI_FINITE"]);
    }

    #[test]
    fn local_media_provenance_selects_zero_disclosure_observation_profile() {
        let task = ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::MediaProvenanceInspection],
            coverage_requirement: CoverageRequirementV1::ExactFiniteCorpus,
            required_corpus_commitment: None,
            currentness_requirement: CurrentnessRequirementV1::CurrentQualified,
            allowed_views: vec![ToolViewClassV1::LocalArtifactView], forbidden_views: vec![],
            forbidden_disclosure_surfaces: vec![DisclosureSurfaceV1::QueryTerms, DisclosureSurfaceV1::TargetIdentifiers, DisclosureSurfaceV1::UploadedMedia, DisclosureSurfaceV1::DnsTransportMetadata, DisclosureSurfaceV1::HeadersBody, DisclosureSurfaceV1::IpNetworkOrigin, DisclosureSurfaceV1::ProviderAnalytics, DisclosureSurfaceV1::StoredSearchHistory],
            observation_only_required: true, required_transforms: vec![ToolTransformV1::MetadataParsing], external_authorization_review_required: false,
        };
        let summary = select_tool_profiles(&task, corpus_profiles()).unwrap();
        assert_eq!(ids(&summary), vec!["T_MEDIA_PROV"]);
    }

    #[test]
    fn rendered_page_method_fit_still_requires_external_authorization() {
        let task = ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::BrowserRenderedCapture], coverage_requirement: CoverageRequirementV1::BestEffortSearch,
            required_corpus_commitment: None, currentness_requirement: CurrentnessRequirementV1::CurrentQualified,
            allowed_views: vec![ToolViewClassV1::AnonymousPublicView], forbidden_views: vec![], forbidden_disclosure_surfaces: vec![], observation_only_required: false,
            required_transforms: vec![ToolTransformV1::JavascriptExecution, ToolTransformV1::DomConstruction, ToolTransformV1::TextExtraction], external_authorization_review_required: true,
        };
        let summary = select_tool_profiles(&task, corpus_profiles()).unwrap();
        assert_eq!(ids(&summary), vec!["T_BROWSER_CAPTURE"]);
        assert_eq!(summary.eligible_profiles[0].disposition, ToolMethodDispositionV1::NeedsExternalAuthorization);
    }

    #[test]
    fn world_absence_is_not_weakened_to_partial_archive_coverage() {
        let task = ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::WebArchiveLookup], coverage_requirement: CoverageRequirementV1::WorldComplete,
            required_corpus_commitment: None, currentness_requirement: CurrentnessRequirementV1::Any,
            allowed_views: vec![ToolViewClassV1::AnonymousPublicView, ToolViewClassV1::DeclaredProviderCorpus], forbidden_views: vec![], forbidden_disclosure_surfaces: vec![], observation_only_required: false,
            required_transforms: vec![], external_authorization_review_required: false,
        };
        let summary = select_tool_profiles(&task, corpus_profiles()).unwrap();
        assert!(summary.eligible_profiles.is_empty());
        assert_eq!(summary.partial_evidence_profiles.iter().map(ToolProfileRef::as_str).collect::<Vec<_>>(), vec!["T_ARCHIVE_PARTIAL"]);
        assert!(summary.unmet_requirements.contains(&UnmetToolRequirementV1::NoToolProfileCanEstablishRequestedCoverage));
    }

    #[test]
    fn provider_rank_and_candidate_order_do_not_change_selection() {
        let a = select_tool_profiles(&public_web_task(), corpus_profiles()).unwrap();
        let mut mutated = corpus_profiles();
        mutated.reverse();
        for profile in &mut mutated { profile.provider_ranking_ref = p("ranking:changed"); }
        let b = select_tool_profiles(&public_web_task(), mutated).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn missing_required_browser_transform_is_ineligible() {
        let task = ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::BrowserRenderedCapture], coverage_requirement: CoverageRequirementV1::BestEffortSearch,
            required_corpus_commitment: None, currentness_requirement: CurrentnessRequirementV1::CurrentQualified,
            allowed_views: vec![ToolViewClassV1::AnonymousPublicView], forbidden_views: vec![], forbidden_disclosure_surfaces: vec![], observation_only_required: false,
            required_transforms: vec![ToolTransformV1::JavascriptExecution], external_authorization_review_required: true,
        };
        let mut profiles = corpus_profiles();
        profiles.iter_mut().find(|p| p.profile_ref.as_str() == "T_BROWSER_CAPTURE").unwrap().transforms.retain(|t| *t != ToolTransformV1::JavascriptExecution);
        let summary = select_tool_profiles(&task, profiles).unwrap();
        assert!(summary.eligible_profiles.is_empty());
        assert!(blocked(&summary, "T_BROWSER_CAPTURE", IneligibilityReasonV1::MissingRequiredTransform));
    }

    #[test]
    fn duplicate_profile_identity_rejects_with_redacted_diagnostics() {
        let mut profiles = corpus_profiles();
        profiles.push(profiles[0].clone());
        let error = select_tool_profiles(&public_web_task(), profiles).unwrap_err();
        assert!(!format!("{error}").contains("T_WEB_PUBLIC_TOPK"));
        assert!(!format!("{error:?}").contains("T_WEB_PUBLIC_TOPK"));
    }
}
