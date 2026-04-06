// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Export Formats
//!
//! Standards-compliant export of Praxis learning data for interoperability
//! with legacy LMS platforms, credential wallets, and institutional systems.
//!
//! ## Supported Formats
//!
//! - **xAPI (Experience API)**: Learning event statements for LMS integration
//! - **Open Badges 3.0**: W3C VC-aligned achievement badges
//! - **Full JSON Export**: Complete learner data (credentials, TEND, PoL)

use serde::{Deserialize, Serialize};

// ============== xAPI Statements ==============
//
// xAPI (formerly Tin Can API) is the standard for recording learning
// experiences across platforms. Each statement is Actor-Verb-Object.
// See: https://xapi.com/overview/

/// An xAPI statement representing a learning experience.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiStatement {
    /// Actor (who did it)
    pub actor: XapiActor,
    /// Verb (what they did)
    pub verb: XapiVerb,
    /// Object (what they did it to)
    pub object: XapiObject,
    /// Result (how it went)
    pub result: Option<XapiResult>,
    /// Context (additional metadata)
    pub context: Option<XapiContext>,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiActor {
    /// Agent identifier (DID or email)
    pub mbox: Option<String>,
    /// Agent name
    pub name: Option<String>,
    /// Object type (always "Agent")
    #[serde(rename = "objectType")]
    pub object_type: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiVerb {
    /// IRI identifying the verb
    pub id: String,
    /// Human-readable display
    pub display: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiObject {
    /// IRI of the learning object
    pub id: String,
    /// Object type
    #[serde(rename = "objectType")]
    pub object_type: String,
    /// Definition
    pub definition: Option<XapiDefinition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiDefinition {
    /// Human-readable name
    pub name: Option<std::collections::HashMap<String, String>>,
    /// Description
    pub description: Option<std::collections::HashMap<String, String>>,
    /// Activity type IRI
    #[serde(rename = "type")]
    pub activity_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiResult {
    /// Score
    pub score: Option<XapiScore>,
    /// Success/failure
    pub success: Option<bool>,
    /// Completion status
    pub completion: Option<bool>,
    /// Duration (ISO 8601 duration)
    pub duration: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiScore {
    /// Scaled score (0.0-1.0)
    pub scaled: Option<f64>,
    /// Raw score
    pub raw: Option<f64>,
    /// Minimum possible
    pub min: Option<f64>,
    /// Maximum possible
    pub max: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XapiContext {
    /// Platform identifier
    pub platform: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Extensions
    pub extensions: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Standard xAPI verb IRIs
pub mod xapi_verbs {
    pub const COMPLETED: &str = "http://adlnet.gov/expapi/verbs/completed";
    pub const PASSED: &str = "http://adlnet.gov/expapi/verbs/passed";
    pub const FAILED: &str = "http://adlnet.gov/expapi/verbs/failed";
    pub const MASTERED: &str = "http://adlnet.gov/expapi/verbs/mastered";
    pub const ATTEMPTED: &str = "http://adlnet.gov/expapi/verbs/attempted";
    pub const EXPERIENCED: &str = "http://adlnet.gov/expapi/verbs/experienced";
    pub const EARNED: &str = "https://w3id.org/xapi/tla/verbs/earned";
}

/// Convert a learning event into an xAPI statement.
pub fn learning_event_to_xapi(
    learner_id: &str,
    learner_name: Option<&str>,
    event_type: &str,
    topic_id: &str,
    topic_name: &str,
    quality_permille: u16,
    duration_seconds: u32,
    timestamp: &str,
) -> XapiStatement {
    let verb_id = match event_type {
        "SkillMastered" => xapi_verbs::MASTERED,
        "QuizPassed" => xapi_verbs::PASSED,
        "QuizFailed" => xapi_verbs::FAILED,
        "LessonComplete" => xapi_verbs::COMPLETED,
        "SrsReview" | "SrsGraduated" => xapi_verbs::EXPERIENCED,
        _ => xapi_verbs::ATTEMPTED,
    };

    let verb_display = match event_type {
        "SkillMastered" => "mastered",
        "QuizPassed" => "passed",
        "QuizFailed" => "failed",
        "LessonComplete" => "completed",
        _ => "attempted",
    };

    let mut display = std::collections::HashMap::new();
    display.insert("en-US".to_string(), verb_display.to_string());

    let mut name_map = std::collections::HashMap::new();
    name_map.insert("en-US".to_string(), topic_name.to_string());

    // ISO 8601 duration from seconds
    let hours = duration_seconds / 3600;
    let minutes = (duration_seconds % 3600) / 60;
    let secs = duration_seconds % 60;
    let duration = format!("PT{}H{}M{}S", hours, minutes, secs);

    let mut extensions = std::collections::HashMap::new();
    extensions.insert(
        "https://mycelix.net/xapi/extensions/pol-quality".to_string(),
        serde_json::json!(quality_permille),
    );
    extensions.insert(
        "https://mycelix.net/xapi/extensions/platform".to_string(),
        serde_json::json!("Praxis"),
    );

    XapiStatement {
        actor: XapiActor {
            mbox: Some(format!("mailto:{}@mycelix.net", learner_id)),
            name: learner_name.map(|n| n.to_string()),
            object_type: "Agent".to_string(),
        },
        verb: XapiVerb { id: verb_id.to_string(), display },
        object: XapiObject {
            id: format!("https://praxis.mycelix.net/curriculum/{}", topic_id),
            object_type: "Activity".to_string(),
            definition: Some(XapiDefinition {
                name: Some(name_map),
                description: None,
                activity_type: Some("http://adlnet.gov/expapi/activities/lesson".to_string()),
            }),
        },
        result: Some(XapiResult {
            score: Some(XapiScore {
                scaled: Some(quality_permille as f64 / 1000.0),
                raw: Some(quality_permille as f64),
                min: Some(0.0),
                max: Some(1000.0),
            }),
            success: Some(quality_permille >= 600),
            completion: Some(event_type == "LessonComplete" || event_type == "SkillMastered"),
            duration: Some(duration),
        }),
        context: Some(XapiContext {
            platform: Some("Praxis by Mycelix".to_string()),
            language: Some("en".to_string()),
            extensions: Some(extensions),
        }),
        timestamp: timestamp.to_string(),
    }
}

// ============== Open Badges 3.0 ==============
//
// Open Badges 3.0 aligns with W3C Verifiable Credentials.
// See: https://1edtech.github.io/openbadges-specification/ob_v3p0.html

/// An Open Badges 3.0 achievement credential.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenBadge {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Credential type
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    /// Issuer
    pub issuer: BadgeIssuer,
    /// Issuance date (ISO 8601)
    #[serde(rename = "issuanceDate")]
    pub issuance_date: String,
    /// Credential subject (the learner)
    #[serde(rename = "credentialSubject")]
    pub credential_subject: BadgeSubject,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BadgeIssuer {
    /// Issuer identifier
    pub id: String,
    /// Issuer name
    pub name: String,
    /// Issuer URL
    pub url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BadgeSubject {
    /// Recipient identifier
    pub id: String,
    /// Achievement earned
    pub achievement: BadgeAchievement,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BadgeAchievement {
    /// Achievement type
    #[serde(rename = "type")]
    pub achievement_type: Vec<String>,
    /// Achievement name
    pub name: String,
    /// Description
    pub description: String,
    /// Criteria for earning
    pub criteria: BadgeCriteria,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BadgeCriteria {
    /// Narrative description of criteria
    pub narrative: String,
}

/// Convert an Praxis credential to Open Badges 3.0 format.
pub fn credential_to_open_badge(
    learner_id: &str,
    credential_title: &str,
    credential_description: &str,
    issuer_name: &str,
    issuance_date: &str,
    pol_score: Option<f64>,
    vitality: Option<u16>,
) -> OpenBadge {
    let mut criteria_parts = vec![
        format!("Completed: {}", credential_title),
    ];
    if let Some(pol) = pol_score {
        criteria_parts.push(format!("Proof of Learning score: {:.0}%", pol * 100.0));
    }
    if let Some(v) = vitality {
        criteria_parts.push(format!("Credential vitality: {}%", v / 10));
    }

    OpenBadge {
        context: vec![
            "https://www.w3.org/2018/credentials/v1".to_string(),
            "https://purl.imsglobal.org/spec/ob/v3p0/context-3.0.3.json".to_string(),
        ],
        credential_type: vec![
            "VerifiableCredential".to_string(),
            "OpenBadgeCredential".to_string(),
        ],
        issuer: BadgeIssuer {
            id: format!("https://praxis.mycelix.net/issuers/{}", issuer_name.to_lowercase().replace(' ', "-")),
            name: issuer_name.to_string(),
            url: Some("https://praxis.mycelix.net".to_string()),
        },
        issuance_date: issuance_date.to_string(),
        credential_subject: BadgeSubject {
            id: format!("did:mycelix:{}", learner_id),
            achievement: BadgeAchievement {
                achievement_type: vec!["Achievement".to_string()],
                name: credential_title.to_string(),
                description: credential_description.to_string(),
                criteria: BadgeCriteria {
                    narrative: criteria_parts.join(". "),
                },
            },
        },
    }
}

// ============== Full Export ==============

/// Complete learner data export for portability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullExport {
    /// Export metadata
    pub metadata: ExportMetadata,
    /// Learning events as xAPI statements
    pub xapi_statements: Vec<XapiStatement>,
    /// Credentials as Open Badges
    pub open_badges: Vec<OpenBadge>,
    /// TEND transaction history
    pub tend_history: Vec<TendTransaction>,
    /// PoL scores by domain
    pub pol_scores: Vec<PolScore>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub version: String,
    pub exported_at: String,
    pub learner_id: String,
    pub platform: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TendTransaction {
    pub amount: f32,
    pub event_type: String,
    pub timestamp: String,
    pub synced: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolScore {
    pub domain: String,
    pub composite_score: f64,
    pub trajectory: f64,
    pub error_authenticity: f64,
    pub retention: f64,
    pub transfer: f64,
    pub contribution: f64,
    pub consistency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xapi_mastery_statement() {
        let stmt = learning_event_to_xapi(
            "agent123",
            Some("Maria"),
            "SkillMastered",
            "CAPS.Mathematics.Gr12.P1.CALC",
            "Calculus",
            850,
            3600,
            "2026-04-03T10:00:00Z",
        );

        assert_eq!(stmt.verb.id, xapi_verbs::MASTERED);
        assert_eq!(stmt.actor.name, Some("Maria".to_string()));
        assert!(stmt.result.as_ref().unwrap().success.unwrap());
        assert!(stmt.result.as_ref().unwrap().completion.unwrap());
        let score = stmt.result.as_ref().unwrap().score.as_ref().unwrap();
        assert!((score.scaled.unwrap() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_xapi_failed_quiz() {
        let stmt = learning_event_to_xapi(
            "agent456",
            None,
            "QuizFailed",
            "CAPS.Mathematics.Gr12.P2.TRIG",
            "Trigonometry",
            350,
            1800,
            "2026-04-03T14:00:00Z",
        );

        assert_eq!(stmt.verb.id, xapi_verbs::FAILED);
        assert!(!stmt.result.as_ref().unwrap().success.unwrap());
    }

    #[test]
    fn test_open_badge_generation() {
        let badge = credential_to_open_badge(
            "agent123",
            "CAPS Mathematics Grade 12",
            "Mastery of the full CAPS Grade 12 Mathematics curriculum",
            "Praxis",
            "2026-04-03T10:00:00Z",
            Some(0.87),
            Some(920),
        );

        assert!(badge.credential_type.contains(&"OpenBadgeCredential".to_string()));
        assert_eq!(badge.credential_subject.id, "did:mycelix:agent123");
        assert!(badge.credential_subject.achievement.criteria.narrative.contains("87%"));
        assert!(badge.credential_subject.achievement.criteria.narrative.contains("92%"));
    }

    #[test]
    fn test_xapi_duration_format() {
        let stmt = learning_event_to_xapi(
            "a", None, "LessonComplete", "t", "Test", 800, 5432, "2026-01-01T00:00:00Z",
        );
        let duration = stmt.result.unwrap().duration.unwrap();
        assert_eq!(duration, "PT1H30M32S");
    }

    #[test]
    fn test_open_badge_without_pol() {
        let badge = credential_to_open_badge(
            "agent", "Test Badge", "A test", "Issuer", "2026-01-01", None, None,
        );
        assert!(!badge.credential_subject.achievement.criteria.narrative.contains("Proof"));
    }

    #[test]
    fn test_full_export_serialization() {
        let export = FullExport {
            metadata: ExportMetadata {
                version: "1.0".to_string(),
                exported_at: "2026-04-03T10:00:00Z".to_string(),
                learner_id: "agent123".to_string(),
                platform: "Praxis".to_string(),
            },
            xapi_statements: vec![],
            open_badges: vec![],
            tend_history: vec![TendTransaction {
                amount: 0.5,
                event_type: "LessonComplete".to_string(),
                timestamp: "2026-04-03T10:00:00Z".to_string(),
                synced: false,
            }],
            pol_scores: vec![],
        };

        let json = serde_json::to_string(&export).unwrap();
        assert!(json.contains("\"version\":\"1.0\""));
        assert!(json.contains("\"amount\":0.5"));
    }
}
