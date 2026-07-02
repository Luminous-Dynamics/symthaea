// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use personal_leptos_types::{
    ActivityItemView, BiometricView, ConsentGrantView, CredentialType, DataSharingPreferenceView,
    MasterKeyView, PreferenceChangeLogView, ProfileView, StoredCredentialView,
};
use std::collections::HashMap;

pub fn mock_profile() -> ProfileView {
    let mut metadata = HashMap::new();
    metadata.insert("language".into(), "en-US".into());
    metadata.insert("timezone".into(), "UTC".into());
    metadata.insert("stewardship".into(), "guardian-grove".into());

    ProfileView {
        display_name: "Tara Vale".into(),
        avatar: None,
        bio: Some("Custodian of identity, credentials, and selective disclosure.".into()),
        metadata,
        updated_at: 1_776_500_000_000_000,
    }
}

pub fn mock_keys() -> Vec<MasterKeyView> {
    vec![
        MasterKeyView {
            label: "Primary signing key".into(),
            purpose: "signing".into(),
            public_key_hex: "a3f21b7cd9a8e34173c0d4885d0ab13f".into(),
            active: true,
            created_at: 1_775_800_000_000_000,
        },
        MasterKeyView {
            label: "Vault encryption key".into(),
            purpose: "encryption".into(),
            public_key_hex: "ff81c2a00b12dd7842d4811f9833cce2".into(),
            active: true,
            created_at: 1_775_900_000_000_000,
        },
        MasterKeyView {
            label: "Credential issuer key".into(),
            purpose: "credential_issuance".into(),
            public_key_hex: "77eb4315af101ceea238114e0fd329c9".into(),
            active: false,
            created_at: 1_774_900_000_000_000,
        },
    ]
}

pub fn mock_credentials() -> Vec<StoredCredentialView> {
    vec![
        StoredCredentialView {
            hash: "cred-health-1".into(),
            credential_type: CredentialType::Health,
            issuer: "did:mycelix:health".into(),
            issued_at: 1_776_200_000_000_000,
            expires_at: None,
            revoked: false,
        },
        StoredCredentialView {
            hash: "cred-identity-1".into(),
            credential_type: CredentialType::Identity,
            issuer: "did:mycelix:identity".into(),
            issued_at: 1_776_000_000_000_000,
            expires_at: None,
            revoked: false,
        },
        StoredCredentialView {
            hash: "cred-governance-1".into(),
            credential_type: CredentialType::Governance,
            issuer: "did:mycelix:governance".into(),
            issued_at: 1_775_600_000_000_000,
            expires_at: Some(1_779_600_000_000_000),
            revoked: false,
        },
        StoredCredentialView {
            hash: "cred-domain-water".into(),
            credential_type: CredentialType::Domain("Water Steward".into()),
            issuer: "did:mycelix:commons".into(),
            issued_at: 1_774_600_000_000_000,
            expires_at: None,
            revoked: false,
        },
    ]
}

pub fn mock_health_records() -> usize {
    7
}

pub fn mock_biometrics() -> Vec<BiometricView> {
    vec![
        BiometricView {
            hash: "bio-1".into(),
            metric_type: "heart_rate".into(),
            value: 63.0,
            unit: "bpm".into(),
            measured_at: 1_776_510_000_000_000,
        },
        BiometricView {
            hash: "bio-2".into(),
            metric_type: "sleep_hours".into(),
            value: 7.4,
            unit: "hours".into(),
            measured_at: 1_776_430_000_000_000,
        },
        BiometricView {
            hash: "bio-3".into(),
            metric_type: "steps".into(),
            value: 8421.0,
            unit: "count".into(),
            measured_at: 1_776_350_000_000_000,
        },
    ]
}

pub fn mock_consents() -> Vec<ConsentGrantView> {
    vec![
        ConsentGrantView {
            hash: "consent-health".into(),
            grantee: "did:mycelix:health:care-team".into(),
            record_types: vec!["medication".into(), "allergy".into()],
            expires_at: Some(1_779_000_000_000_000),
            active: true,
            created_at: 1_776_100_000_000_000,
        },
        ConsentGrantView {
            hash: "consent-identity".into(),
            grantee: "did:mycelix:identity:verification".into(),
            record_types: vec!["*".into()],
            expires_at: None,
            active: true,
            created_at: 1_775_900_000_000_000,
        },
        ConsentGrantView {
            hash: "consent-research".into(),
            grantee: "did:mycelix:research:commons".into(),
            record_types: vec!["lab_result".into()],
            expires_at: Some(1_777_000_000_000_000),
            active: false,
            created_at: 1_775_100_000_000_000,
        },
    ]
}

pub fn mock_activity() -> Vec<ActivityItemView> {
    vec![
        ActivityItemView {
            id: "event-identity-proof".into(),
            kind: "event".into(),
            domain: "identity".into(),
            title: "identity disclosure event".into(),
            detail: "Shared identity proof with Governance for voting eligibility.".into(),
            created_at: 1_776_510_000_000_000,
            success: None,
        },
        ActivityItemView {
            id: "event-health-consent".into(),
            kind: "event".into(),
            domain: "health".into(),
            title: "health consent event".into(),
            detail: "Granted medication visibility to Health care team.".into(),
            created_at: 1_776_500_000_000_000,
            success: None,
        },
        ActivityItemView {
            id: "query-credential".into(),
            kind: "query".into(),
            domain: "credential".into(),
            title: "present_credential query".into(),
            detail: "Received updated Health credential from trusted issuer.".into(),
            created_at: 1_776_490_000_000_000,
            success: Some(true),
        },
        ActivityItemView {
            id: "query-research-block".into(),
            kind: "query".into(),
            domain: "research".into(),
            title: "disclosure block".into(),
            detail: "Revoked expired research disclosure scope.".into(),
            created_at: 1_776_480_000_000_000,
            success: Some(false),
        },
    ]
}

pub fn mock_preferences() -> Vec<DataSharingPreferenceView> {
    vec![
        DataSharingPreferenceView {
            source_cluster: "personal".into(),
            target_cluster: "identity".into(),
            allowed: true,
            blocked_zomes: vec![],
            reason: "Allow credential verification and profile handoff.".into(),
            updated_at: 1_776_420_000_000_000,
        },
        DataSharingPreferenceView {
            source_cluster: "personal".into(),
            target_cluster: "health".into(),
            allowed: true,
            blocked_zomes: vec!["lab_result".into()],
            reason: "Allow care coordination, but keep lab results manually scoped.".into(),
            updated_at: 1_776_430_000_000_000,
        },
        DataSharingPreferenceView {
            source_cluster: "personal".into(),
            target_cluster: "research".into(),
            allowed: false,
            blocked_zomes: vec![],
            reason: "Research exports require explicit per-flow approval.".into(),
            updated_at: 1_776_440_000_000_000,
        },
    ]
}

pub fn mock_preference_log() -> Vec<PreferenceChangeLogView> {
    vec![
        PreferenceChangeLogView {
            source_cluster: "personal".into(),
            target_cluster: "research".into(),
            was_allowed: true,
            now_allowed: false,
            changed_at: 1_776_440_000_000_000,
        },
        PreferenceChangeLogView {
            source_cluster: "personal".into(),
            target_cluster: "health".into(),
            was_allowed: true,
            now_allowed: true,
            changed_at: 1_776_430_000_000_000,
        },
    ]
}
