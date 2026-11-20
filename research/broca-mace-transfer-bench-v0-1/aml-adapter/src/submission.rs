// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Static preflight for the official Agent Memory Leaderboard Add/Search
//! submission contract. This validates declarations only; it does not verify
//! runtime model use, public reachability, authentication, or a successful
//! platform smoke test.

use std::net::IpAddr;

use sha2::{Digest, Sha256};

use super::Track;

pub const AML_OPEN_METHOD_EMBEDDING_MODEL: &str = "text-embedding-v4";
pub const AML_OPEN_METHOD_LLM_MODEL: &str = "gpt-4o-mini";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionDivision {
    OpenSourceMethods,
    CommercialProducts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostingMode {
    ParticipantHostedApi,
    RepositoryOnly,
    PlatformManaged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthenticationMode {
    None,
    MemorySystemKeyConfigured,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelUsageProfileV1 {
    /// Every embedding model actually used by the submitted system.
    pub embedding_model_ids: Vec<String>,
    /// Every LLM-related component actually used by the submitted system.
    pub llm_model_ids: Vec<String>,
    /// AML leaves reranker choice unrestricted.
    pub reranker_model_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmlSubmissionProfileV1 {
    pub division: SubmissionDivision,
    pub track: Track,
    pub hosting_mode: HostingMode,
    pub api_contract_version: String,
    pub service_version: String,
    pub add_endpoint: String,
    pub search_endpoint: String,
    pub authentication: AuthenticationMode,
    pub source_revision_sha256: String,
    pub runtime_image_sha256: String,
    pub model_usage: ModelUsageProfileV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionProfileReject {
    EmptyContractVersion,
    EmptyServiceVersion,
    ParticipantHostedApiRequired,
    InvalidAddEndpoint,
    InvalidSearchEndpoint,
    EndpointContainsCredentialsOrQuery,
    EndpointUsesLocalHost,
    InvalidSourceRevisionDigest,
    InvalidRuntimeImageDigest,
    EmptyModelIdentifier,
    OpenMethodsEmbeddingModelRestricted,
    OpenMethodsLlmModelRestricted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredAmlSubmissionProfileV1 {
    profile: AmlSubmissionProfileV1,
    canonical_sha256: String,
}

impl DeclaredAmlSubmissionProfileV1 {
    pub fn profile(&self) -> &AmlSubmissionProfileV1 {
        &self.profile
    }

    pub fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn endpoint_host(value: &str) -> Option<&str> {
    let authority_and_path = value.strip_prefix("https://")?;
    let authority = authority_and_path.split('/').next()?;
    if authority.is_empty() {
        return None;
    }
    if authority.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return None;
    }

    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        return Some(&rest[..end]);
    }
    Some(authority.split(':').next()?)
}

fn is_obviously_local_host(host: &str) -> bool {
    let normalized = host.trim_end_matches('.').to_ascii_lowercase();
    if normalized == "localhost"
        || normalized.ends_with(".localhost")
        || normalized.ends_with(".local")
        || normalized == "::1"
        || normalized == "0.0.0.0"
    {
        return true;
    }
    if let Ok(address) = normalized.parse::<IpAddr>() {
        return match address {
            IpAddr::V4(ip) => {
                ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified()
            }
            IpAddr::V6(ip) => {
                let bytes = ip.octets();
                ip.is_loopback()
                    || ip.is_unspecified()
                    || (bytes[0] & 0xfe == 0xfc)
                    || (bytes[0] == 0xfe && bytes[1] & 0xc0 == 0x80)
            },
        };
    }
    false
}

fn validate_endpoint(value: &str) -> Result<(), SubmissionProfileReject> {
    if !value.starts_with("https://") {
        return Err(SubmissionProfileReject::InvalidAddEndpoint);
    }
    if value.contains('@') || value.contains('?') || value.contains('#') {
        return Err(SubmissionProfileReject::EndpointContainsCredentialsOrQuery);
    }
    let host = endpoint_host(value).ok_or(SubmissionProfileReject::InvalidAddEndpoint)?;
    if host.is_empty() || (!host.contains('.') && host.parse::<IpAddr>().is_err()) {
        return Err(SubmissionProfileReject::InvalidAddEndpoint);
    }
    if is_obviously_local_host(host) {
        return Err(SubmissionProfileReject::EndpointUsesLocalHost);
    }
    Ok(())
}

fn profile_digest(profile: &AmlSubmissionProfileV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:aml-submission-profile:v1\0");
    hash_field(
        &mut hasher,
        match profile.division {
            SubmissionDivision::OpenSourceMethods => "open_source_methods",
            SubmissionDivision::CommercialProducts => "commercial_products",
        },
    );
    hash_field(
        &mut hasher,
        match profile.track {
            Track::Textual => "textual",
            Track::Coding => "coding",
            Track::Multimodal => "multimodal",
        },
    );
    hash_field(
        &mut hasher,
        match profile.hosting_mode {
            HostingMode::ParticipantHostedApi => "participant_hosted_api",
            HostingMode::RepositoryOnly => "repository_only",
            HostingMode::PlatformManaged => "platform_managed",
        },
    );
    hash_field(&mut hasher, &profile.api_contract_version);
    hash_field(&mut hasher, &profile.service_version);
    hash_field(&mut hasher, &profile.add_endpoint);
    hash_field(&mut hasher, &profile.search_endpoint);
    hash_field(
        &mut hasher,
        match profile.authentication {
            AuthenticationMode::None => "none",
            AuthenticationMode::MemorySystemKeyConfigured => "memory_system_key_configured",
        },
    );
    hash_field(&mut hasher, &profile.source_revision_sha256);
    hash_field(&mut hasher, &profile.runtime_image_sha256);

    for (kind, models) in [
        ("embedding", &profile.model_usage.embedding_model_ids),
        ("llm", &profile.model_usage.llm_model_ids),
        ("reranker", &profile.model_usage.reranker_model_ids),
    ] {
        hash_field(&mut hasher, kind);
        hasher.update((models.len() as u64).to_be_bytes());
        for model in models {
            hash_field(&mut hasher, model);
        }
    }

    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Check a participant-declared submission profile against the public AML
/// rules currently published for Cycle 2. This does not prove that a running
/// endpoint actually uses the declared models or is reachable from AML.
pub fn validate_declared_submission_profile(
    profile: AmlSubmissionProfileV1,
) -> Result<DeclaredAmlSubmissionProfileV1, SubmissionProfileReject> {
    if profile.api_contract_version.trim().is_empty() {
        return Err(SubmissionProfileReject::EmptyContractVersion);
    }
    if profile.service_version.trim().is_empty() {
        return Err(SubmissionProfileReject::EmptyServiceVersion);
    }
    if profile.hosting_mode != HostingMode::ParticipantHostedApi {
        return Err(SubmissionProfileReject::ParticipantHostedApiRequired);
    }
    validate_endpoint(&profile.add_endpoint)
        .map_err(|error| match error {
            SubmissionProfileReject::InvalidAddEndpoint
            | SubmissionProfileReject::EndpointContainsCredentialsOrQuery
            | SubmissionProfileReject::EndpointUsesLocalHost => error,
            _ => SubmissionProfileReject::InvalidAddEndpoint,
        })?;
    validate_endpoint(&profile.search_endpoint)
        .map_err(|error| match error {
            SubmissionProfileReject::InvalidAddEndpoint => {
                SubmissionProfileReject::InvalidSearchEndpoint
            }
            SubmissionProfileReject::EndpointContainsCredentialsOrQuery
            | SubmissionProfileReject::EndpointUsesLocalHost => error,
            _ => SubmissionProfileReject::InvalidSearchEndpoint,
        })?;
    if !is_sha256_hex(&profile.source_revision_sha256) {
        return Err(SubmissionProfileReject::InvalidSourceRevisionDigest);
    }
    if !is_sha256_hex(&profile.runtime_image_sha256) {
        return Err(SubmissionProfileReject::InvalidRuntimeImageDigest);
    }

    let model_groups = [
        &profile.model_usage.embedding_model_ids,
        &profile.model_usage.llm_model_ids,
        &profile.model_usage.reranker_model_ids,
    ];
    if model_groups
        .iter()
        .flat_map(|models| models.iter())
        .any(|model| model.trim().is_empty())
    {
        return Err(SubmissionProfileReject::EmptyModelIdentifier);
    }

    if profile.division == SubmissionDivision::OpenSourceMethods {
        if profile
            .model_usage
            .embedding_model_ids
            .iter()
            .any(|model| model != AML_OPEN_METHOD_EMBEDDING_MODEL)
        {
            return Err(SubmissionProfileReject::OpenMethodsEmbeddingModelRestricted);
        }
        if profile
            .model_usage
            .llm_model_ids
            .iter()
            .any(|model| model != AML_OPEN_METHOD_LLM_MODEL)
        {
            return Err(SubmissionProfileReject::OpenMethodsLlmModelRestricted);
        }
    }

    let canonical_sha256 = profile_digest(&profile);
    Ok(DeclaredAmlSubmissionProfileV1 {
        profile,
        canonical_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(division: SubmissionDivision) -> AmlSubmissionProfileV1 {
        AmlSubmissionProfileV1 {
            division,
            track: Track::Textual,
            hosting_mode: HostingMode::ParticipantHostedApi,
            api_contract_version: "aml-add-search-v1".into(),
            service_version: "mace-aml-0.1.0".into(),
            add_endpoint: "https://memory.example.org/add".into(),
            search_endpoint: "https://memory.example.org/search".into(),
            authentication: AuthenticationMode::MemorySystemKeyConfigured,
            source_revision_sha256: "a".repeat(64),
            runtime_image_sha256: "b".repeat(64),
            model_usage: ModelUsageProfileV1 {
                embedding_model_ids: vec![AML_OPEN_METHOD_EMBEDDING_MODEL.into()],
                llm_model_ids: vec![AML_OPEN_METHOD_LLM_MODEL.into()],
                reranker_model_ids: vec!["cross-encoder/custom-v7".into()],
            },
        }
    }

    #[test]
    fn open_methods_profile_accepts_required_models_and_unrestricted_reranker() {
        let result =
            validate_declared_submission_profile(profile(SubmissionDivision::OpenSourceMethods));
        assert!(result.is_ok());
        let digest = result.unwrap().canonical_sha256().to_owned();
        assert!(is_sha256_hex(&digest));
    }

    #[test]
    fn open_methods_rejects_noncompliant_embedding_model() {
        let mut value = profile(SubmissionDivision::OpenSourceMethods);
        value.model_usage.embedding_model_ids = vec!["text-embedding-3-large".into()];
        assert_eq!(
            validate_declared_submission_profile(value),
            Err(SubmissionProfileReject::OpenMethodsEmbeddingModelRestricted)
        );
    }

    #[test]
    fn open_methods_rejects_noncompliant_llm_component_model() {
        let mut value = profile(SubmissionDivision::OpenSourceMethods);
        value.model_usage.llm_model_ids = vec!["gpt-6".into()];
        assert_eq!(
            validate_declared_submission_profile(value),
            Err(SubmissionProfileReject::OpenMethodsLlmModelRestricted)
        );
    }

    #[test]
    fn commercial_products_have_no_cycle_two_model_restrictions() {
        let mut value = profile(SubmissionDivision::CommercialProducts);
        value.model_usage.embedding_model_ids = vec!["vendor-embedding-9".into()];
        value.model_usage.llm_model_ids = vec!["vendor-llm-4".into()];
        let result = validate_declared_submission_profile(value);
        assert!(result.is_ok());
    }

    #[test]
    fn open_methods_can_omit_components_the_system_does_not_use() {
        let mut value = profile(SubmissionDivision::OpenSourceMethods);
        value.model_usage.embedding_model_ids.clear();
        value.model_usage.llm_model_ids.clear();
        assert!(validate_declared_submission_profile(value).is_ok());
    }

    #[test]
    fn profile_requires_participant_hosted_https_endpoints() {
        let mut repository_only = profile(SubmissionDivision::OpenSourceMethods);
        repository_only.hosting_mode = HostingMode::RepositoryOnly;
        assert_eq!(
            validate_declared_submission_profile(repository_only),
            Err(SubmissionProfileReject::ParticipantHostedApiRequired)
        );

        let mut insecure = profile(SubmissionDivision::OpenSourceMethods);
        insecure.add_endpoint = "http://memory.example.org/add".into();
        assert_eq!(
            validate_declared_submission_profile(insecure),
            Err(SubmissionProfileReject::InvalidAddEndpoint)
        );
    }

    #[test]
    fn endpoint_credentials_query_and_local_hosts_are_rejected() {
        for endpoint in [
            "https://user:secret@memory.example.org/add",
            "https://memory.example.org/add?key=secret",
            "https://localhost/add",
            "https://127.0.0.1/add",
            "https://service.local/add",
        ] {
            let mut value = profile(SubmissionDivision::OpenSourceMethods);
            value.add_endpoint = endpoint.into();
            assert!(validate_declared_submission_profile(value).is_err());
        }
    }

    #[test]
    fn search_endpoint_has_distinct_validation_and_secret_url_rejects_are_specific() {
        let mut bad_search = profile(SubmissionDivision::OpenSourceMethods);
        bad_search.search_endpoint = "http://memory.example.org/search".into();
        assert_eq!(
            validate_declared_submission_profile(bad_search),
            Err(SubmissionProfileReject::InvalidSearchEndpoint)
        );

        let mut credentialed = profile(SubmissionDivision::OpenSourceMethods);
        credentialed.add_endpoint = "https://user:secret@memory.example.org/add".into();
        assert_eq!(
            validate_declared_submission_profile(credentialed),
            Err(SubmissionProfileReject::EndpointContainsCredentialsOrQuery)
        );
    }

    #[test]
    fn profile_digest_changes_when_any_declared_model_changes() {
        let original =
            validate_declared_submission_profile(profile(SubmissionDivision::OpenSourceMethods))
                .unwrap();
        let mut changed = profile(SubmissionDivision::OpenSourceMethods);
        changed.model_usage.reranker_model_ids = vec!["another-reranker".into()];
        let altered = validate_declared_submission_profile(changed).unwrap();
        assert_ne!(original.canonical_sha256(), altered.canonical_sha256());
    }

    #[test]
    fn profile_rejects_missing_version_or_malformed_deployment_digests() {
        let mut missing_version = profile(SubmissionDivision::OpenSourceMethods);
        missing_version.service_version.clear();
        assert_eq!(
            validate_declared_submission_profile(missing_version),
            Err(SubmissionProfileReject::EmptyServiceVersion)
        );

        let mut malformed_digest = profile(SubmissionDivision::CommercialProducts);
        malformed_digest.runtime_image_sha256 = "nope".into();
        assert_eq!(
            validate_declared_submission_profile(malformed_digest),
            Err(SubmissionProfileReject::InvalidRuntimeImageDigest)
        );
    }
}
