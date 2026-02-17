//! Security Integrity Zome - Entry types for security logging and auditing
use hdi::prelude::*;

/// Security event types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SecurityEventType {
    /// XSS attempt detected
    XssAttempt,
    /// SQL injection attempt
    InjectionAttempt,
    /// Invalid IPFS CID submitted
    InvalidCid,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Profanity detected
    ProfanityDetected,
    /// Suspicious activity pattern
    SuspiciousActivity,
    /// Authorization failure
    AuthorizationFailure,
    /// Input validation failure
    ValidationFailure,
}

/// Security severity levels
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecuritySeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Security log entry for auditing and threat detection
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SecurityLog {
    /// Event type
    pub event_type: SecurityEventType,

    /// Severity level
    pub severity: SecuritySeverity,

    /// Agent who triggered the event
    pub agent: AgentPubKey,

    /// Zome/function where event occurred
    pub zome_name: String,
    pub function_name: String,

    /// Event description
    pub description: String,

    /// Input that triggered the event (sanitized)
    pub trigger_input: Option<String>,

    /// Timestamp
    pub timestamp: Timestamp,

    /// Additional metadata
    pub metadata: String,
}

/// Link types for security logs
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from agent to their security events
    AgentToSecurityLogs,

    /// Links from event type to logs
    EventTypeToLogs,

    /// All security logs anchor
    AllSecurityLogs,
}

/// Entry types for this integrity zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    SecurityLog(SecurityLog),
}

/// Validation function for SecurityLog entries
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::SecurityLog(log) => validate_security_log(&log),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            // All link types are valid for security logging
            match link_type {
                LinkTypes::AgentToSecurityLogs => Ok(ValidateCallbackResult::Valid),
                LinkTypes::EventTypeToLogs => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AllSecurityLogs => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate security log data
fn validate_security_log(log: &SecurityLog) -> ExternResult<ValidateCallbackResult> {
    // Description required
    if log.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Security log description cannot be empty".into(),
        ));
    }

    // Description length check
    if log.description.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Security log description too long (max 1000 characters)".into(),
        ));
    }

    // Zome and function names required
    if log.zome_name.is_empty() || log.function_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Zome and function names required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
