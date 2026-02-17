//! Notifications Integrity Zome - Entry types for user notifications
use hdi::prelude::*;
use std::collections::HashMap;

/// Notification type enumeration
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum NotificationType {
    NewMessage,
    MessageReply,
    TransactionCreated,
    TransactionUpdated,
    TransactionCompleted,
    PaymentReceived,
    PaymentRequired,
    ReviewReceived,
    ReviewReply,
    ListingExpiring,
    ListingFlagged,
    DisputeOpened,
    DisputeResolved,
    MATLScoreChanged,
    SystemAlert,
    SecurityAlert,
}

/// Notification priority levels
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NotificationPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Notification entry - stores a single notification for a user
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Notification {
    pub id: String,
    pub notification_type: NotificationType,
    pub priority: NotificationPriority,
    pub title: String,
    pub message: String,
    pub action_link: Option<String>,
    pub related_entity: Option<ActionHash>,
    pub from_agent: Option<AgentPubKey>,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub read: bool,
    pub dismissed: bool,
    pub read_at: Option<u64>,
    pub metadata: HashMap<String, String>,
}

/// User notification preferences
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct NotificationPreferences {
    pub agent: AgentPubKey,
    pub enabled_types: Vec<NotificationType>,
    pub min_priority: NotificationPriority,
    pub quiet_hours_start: Option<u8>,
    pub quiet_hours_end: Option<u8>,
    pub group_notifications: bool,
    pub auto_dismiss_after: Option<u64>,
    pub max_history: u32,
    pub updated_at: u64,
}

impl Default for NotificationPreferences {
    fn default() -> Self {
        Self {
            agent: AgentPubKey::from_raw_36(vec![0; 36]),
            enabled_types: vec![
                NotificationType::NewMessage,
                NotificationType::TransactionCreated,
                NotificationType::PaymentReceived,
                NotificationType::ReviewReceived,
                NotificationType::DisputeOpened,
                NotificationType::SecurityAlert,
            ],
            min_priority: NotificationPriority::Normal,
            quiet_hours_start: None,
            quiet_hours_end: None,
            group_notifications: true,
            auto_dismiss_after: Some(604800),
            max_history: 100,
            updated_at: 0,
        }
    }
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Notification(Notification),
    NotificationPreferences(NotificationPreferences),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToNotifications,
    AgentToPreferences,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::Notification(n) => {
                    if n.title.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("Notification title required".into()));
                    }
                    if n.message.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("Notification message required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
