// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Realistic mock data for Identity frontend development.

use identity_leptos_types::*;

pub fn mock_did_document() -> DidDocumentView {
    DidDocumentView {
        id: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
        controller: "uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
        verification_methods: vec![
            VerificationMethodView {
                id: "did:mycelix:uhCAk...#keys-1".into(),
                type_name: "Ed25519VerificationKey2020".into(),
                controller: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
                public_key_multibase: "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK".into(),
            },
            VerificationMethodView {
                id: "did:mycelix:uhCAk...#kem-1".into(),
                type_name: "X25519KeyAgreementKey2020".into(),
                controller: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
                public_key_multibase: "z6LSbysY2xFMRpGMhb7tFTLMpeuPRaqaWM1yECx2AtzE3KCc".into(),
            },
        ],
        key_agreements: vec!["did:mycelix:uhCAk...#kem-1".into()],
        services: vec![
            ServiceEndpointView {
                id: "did:mycelix:uhCAk...#hearth".into(),
                type_name: "HearthService".into(),
                endpoint: "ws://localhost:8888".into(),
            },
        ],
        created: 1711900000,
        updated: 1774934400,
        version: 3,
        active: true,
    }
}

pub fn mock_mfa_state() -> MfaStateView {
    MfaStateView {
        did: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
        factors: vec![
            FactorView {
                factor_type: FactorType::PrimaryKeyPair,
                factor_id: "pk-ed25519-1".into(),
                enrolled_at: 1711900000,
                last_verified: 1774934400,
                effective_strength: 1.0,
                active: true,
                metadata: "{}".into(),
            },
            FactorView {
                factor_type: FactorType::Biometric,
                factor_id: "bio-fingerprint-1".into(),
                enrolled_at: 1712000000,
                last_verified: 1774800000,
                effective_strength: 0.85,
                active: true,
                metadata: r#"{"type":"fingerprint","device":"Pixel 8 Pro"}"#.into(),
            },
            FactorView {
                factor_type: FactorType::SocialRecovery,
                factor_id: "social-trustees-1".into(),
                enrolled_at: 1713000000,
                last_verified: 1774000000,
                effective_strength: 0.7,
                active: true,
                metadata: r#"{"trustees":3,"threshold":2}"#.into(),
            },
        ],
        assurance_level: AssuranceLevel::Verified,
        effective_strength: 0.85,
        category_count: 3,
        updated: 1774934400,
    }
}

pub fn mock_recovery_config() -> RecoveryConfigView {
    RecoveryConfigView {
        did: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
        trustees: vec![
            "did:mycelix:elena".into(),
            "did:mycelix:marcus".into(),
            "did:mycelix:ama".into(),
        ],
        threshold: 2,
        time_lock_secs: 86400,
        active: true,
        created: 1713000000,
    }
}

pub fn mock_credentials_held() -> Vec<CredentialView> {
    vec![
        CredentialView {
            id: "vc-edu-001".into(),
            subject_did: "did:mycelix:uhCAk...self".into(),
            issuer_did: "did:mycelix:university-of-cape-town".into(),
            credential_type: vec!["VerifiableCredential".into(), "EducationCredential".into()],
            claims: serde_json::json!({
                "degree": "BSc Computer Science",
                "institution": "University of Cape Town",
                "year": 2022,
                "honors": "cum laude"
            }),
            issued_at: 1672531200,
            expires_at: None,
            revoked: false,
            schema_id: Some("mycelix:schema:education:degree:v1".into()),
        },
        CredentialView {
            id: "vc-comm-001".into(),
            subject_did: "did:mycelix:uhCAk...self".into(),
            issuer_did: "did:mycelix:block-7-dao".into(),
            credential_type: vec!["VerifiableCredential".into(), "CommunityMembership".into()],
            claims: serde_json::json!({
                "community": "Block 7 Commons",
                "role": "Steward",
                "joined": "2024-01-15"
            }),
            issued_at: 1705276800,
            expires_at: Some(1800000000),
            revoked: false,
            schema_id: Some("mycelix:schema:community:membership:v1".into()),
        },
        CredentialView {
            id: "vc-health-001".into(),
            subject_did: "did:mycelix:uhCAk...self".into(),
            issuer_did: "did:mycelix:health-authority".into(),
            credential_type: vec!["VerifiableCredential".into(), "HealthCredential".into()],
            claims: serde_json::json!({
                "type": "Vaccination Record",
                "status": "Complete"
            }),
            issued_at: 1711900000,
            expires_at: Some(1743436000),
            revoked: false,
            schema_id: None,
        },
    ]
}

pub fn mock_credentials_issued() -> Vec<CredentialView> {
    vec![
        CredentialView {
            id: "vc-issued-001".into(),
            subject_did: "did:mycelix:kai".into(),
            issuer_did: "did:mycelix:uhCAk...self".into(),
            credential_type: vec!["VerifiableCredential".into(), "SkillEndorsement".into()],
            claims: serde_json::json!({
                "skill": "Permaculture Design",
                "level": "Intermediate"
            }),
            issued_at: 1774000000,
            expires_at: None,
            revoked: false,
            schema_id: None,
        },
    ]
}

pub fn mock_trust_credentials() -> Vec<TrustCredentialView> {
    vec![
        TrustCredentialView {
            id: "tc-001".into(),
            subject_did: "did:mycelix:uhCAk...self".into(),
            issuer_did: "did:mycelix:block-7-dao".into(),
            trust_tier: TrustTier::Standard,
            issued_at: 1774000000,
            expires_at: Some(1805536000),
            revoked: false,
        },
    ]
}

pub fn mock_reputation() -> ReputationView {
    ReputationView {
        agent: "uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
        composite_score: 0.62,
        domain_scores: vec![
            DomainScoreView { domain: "governance".into(), score: 0.71 },
            DomainScoreView { domain: "commons".into(), score: 0.58 },
            DomainScoreView { domain: "hearth".into(), score: 0.65 },
            DomainScoreView { domain: "finance".into(), score: 0.54 },
        ],
        consciousness_profile: ConsciousnessProfileView {
            identity: 0.75,
            reputation: 0.62,
            community: 0.58,
            engagement: 0.53,
        },
        trust_tier: TrustTier::Standard,
    }
}

pub fn mock_name() -> NameRegistryView {
    NameRegistryView {
        canonical: "river.mycelix".into(),
        owner_did: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD-7dNYBuu2Uj17j4Q5h0eF".into(),
        registered_at: 1712500000,
    }
}
