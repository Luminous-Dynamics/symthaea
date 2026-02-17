use hdk::prelude::*;
use std::collections::HashMap;

/// Notification entry - stores a single notification for a user
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Notification {
    /// Unique ID for this notification
    pub id: String,

    /// Type of notification (NewMessage, TransactionCreated, etc.)
    pub notification_type: NotificationType,

    /// Priority level
    pub priority: NotificationPriority,

    /// Notification title
    pub title: String,

    /// Notification message
    pub message: String,

    /// Optional action link (e.g., "/transactions/123")
    pub action_link: Option<String>,

    /// Related entity hash (listing, transaction, message, etc.)
    pub related_entity: Option<ActionHash>,

    /// Agent who triggered this notification (sender, reviewer, etc.)
    pub from_agent: Option<AgentPubKey>,

    /// When the notification was created
    pub created_at: u64,

    /// When the notification expires (auto-dismiss)
    pub expires_at: Option<u64>,

    /// Whether the notification has been read
    pub read: bool,

    /// Whether the notification has been dismissed
    pub dismissed: bool,

    /// When it was read (if applicable)
    pub read_at: Option<u64>,

    /// Additional metadata (flexible key-value pairs)
    pub metadata: HashMap<String, String>,
}

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
    Critical = 4,  // Security issues, urgent disputes
    High = 3,      // Payments, transactions
    Normal = 2,    // Messages, reviews
    Low = 1,       // System updates, tips
}

/// User notification preferences
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct NotificationPreferences {
    /// Agent these preferences belong to
    pub agent: AgentPubKey,

    /// Enabled notification types
    pub enabled_types: Vec<NotificationType>,

    /// Minimum priority to show
    pub min_priority: NotificationPriority,

    /// Quiet hours (no notifications during these times)
    pub quiet_hours_start: Option<u8>,  // Hour (0-23)
    pub quiet_hours_end: Option<u8>,    // Hour (0-23)

    /// Group similar notifications
    pub group_notifications: bool,

    /// Auto-dismiss after X seconds
    pub auto_dismiss_after: Option<u64>,

    /// Maximum notifications to keep in history
    pub max_history: u32,

    /// Updated timestamp
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
            auto_dismiss_after: Some(604800), // 7 days
            max_history: 100,
            updated_at: 0,
        }
    }
}

/// Response for get_my_notifications
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NotificationListResponse {
    pub notifications: Vec<NotificationWithHash>,
    pub unread_count: u32,
    pub total_count: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NotificationWithHash {
    pub hash: ActionHash,
    pub notification: Notification,
}

/// Statistics about notifications
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NotificationStats {
    pub total: u32,
    pub unread: u32,
    pub by_type: HashMap<String, u32>,
    pub by_priority: HashMap<String, u32>,
}

// ============================================================================
// API ENDPOINTS
// ============================================================================

/// Create a new notification
#[hdk_extern]
pub fn create_notification(notification: Notification) -> ExternResult<ActionHash> {
    // Validate notification
    if notification.title.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Notification title cannot be empty".to_string()
        )));
    }

    if notification.message.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Notification message cannot be empty".to_string()
        )));
    }

    // Get user preferences to check if this notification should be created
    let my_agent = agent_info()?.agent_latest_pubkey;
    let prefs = get_or_create_preferences(my_agent.clone())?;

    // Check if notification type is enabled
    if !prefs.enabled_types.contains(&notification.notification_type) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Notification type is disabled in user preferences".to_string()
        )));
    }

    // Check if priority meets minimum threshold
    if notification.priority < prefs.min_priority {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Notification priority below user threshold".to_string()
        )));
    }

    // Check quiet hours (if configured)
    if let (Some(start), Some(end)) = (prefs.quiet_hours_start, prefs.quiet_hours_end) {
        let current_hour = get_current_hour();
        if is_in_quiet_hours(current_hour, start, end) && notification.priority < NotificationPriority::Critical {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot create non-critical notifications during quiet hours".to_string()
            )));
        }
    }

    // Create the notification entry
    let notification_hash = create_entry(EntryTypes::Notification(notification.clone()))?;

    // Create link from agent to notification for easy retrieval
    let my_agent_hash = hash_entry(my_agent)?;
    create_link(
        my_agent_hash.clone(),
        notification_hash.clone(),
        LinkTypes::AgentToNotifications,
        (),
    )?;

    // Emit signal for real-time delivery
    emit_signal(NotificationSignal::NewNotification {
        notification_hash: notification_hash.clone(),
        notification: notification.clone(),
    })?;

    Ok(notification_hash)
}

/// Get all notifications for the current agent
#[hdk_extern]
pub fn get_my_notifications(filter: NotificationFilter) -> ExternResult<NotificationListResponse> {
    let my_agent = agent_info()?.agent_latest_pubkey;
    let my_agent_hash = hash_entry(my_agent)?;

    // Get all notification links
    let links = get_links(
        GetLinksInputBuilder::try_new(my_agent_hash, LinkTypes::AgentToNotifications)?.build()
    )?;

    let mut notifications = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(notification) = record.entry().to_app_option::<Notification>()? {
                    // Apply filters
                    if should_include_notification(&notification, &filter) {
                        notifications.push(NotificationWithHash {
                            hash: action_hash,
                            notification,
                        });
                    }
                }
            }
        }
    }

    // Sort by created_at (most recent first)
    notifications.sort_by(|a, b| b.notification.created_at.cmp(&a.notification.created_at));

    // Apply limit if specified
    if let Some(limit) = filter.limit {
        notifications.truncate(limit as usize);
    }

    // Calculate counts
    let unread_count = notifications.iter().filter(|n| !n.notification.read).count() as u32;
    let total_count = notifications.len() as u32;

    Ok(NotificationListResponse {
        notifications,
        unread_count,
        total_count,
    })
}

/// Mark a notification as read
#[hdk_extern]
pub fn mark_notification_read(notification_hash: ActionHash) -> ExternResult<ActionHash> {
    // Get the original notification
    let record = get(notification_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Notification not found".to_string()
        )))?;

    let mut notification: Notification = record
        .entry()
        .to_app_option()?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to deserialize notification".to_string()
        )))?;

    // Update read status
    notification.read = true;
    notification.read_at = Some(sys_time()?.as_micros());

    // Create updated entry
    let updated_hash = update_entry(notification_hash, notification.clone())?;

    // Emit signal
    emit_signal(NotificationSignal::NotificationRead {
        notification_hash: updated_hash.clone(),
    })?;

    Ok(updated_hash)
}

/// Mark all notifications as read
#[hdk_extern]
pub fn mark_all_read(_: ()) -> ExternResult<u32> {
    let my_agent = agent_info()?.agent_latest_pubkey;
    let my_agent_hash = hash_entry(my_agent)?;

    let links = get_links(
        GetLinksInputBuilder::try_new(my_agent_hash, LinkTypes::AgentToNotifications)?.build()
    )?;

    let mut marked_count = 0u32;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut notification) = record.entry().to_app_option::<Notification>()? {
                    if !notification.read {
                        notification.read = true;
                        notification.read_at = Some(sys_time()?.as_micros());
                        update_entry(action_hash, notification)?;
                        marked_count += 1;
                    }
                }
            }
        }
    }

    // Emit signal
    emit_signal(NotificationSignal::AllNotificationsRead)?;

    Ok(marked_count)
}

/// Dismiss a notification (mark as dismissed)
#[hdk_extern]
pub fn dismiss_notification(notification_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(notification_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Notification not found".to_string()
        )))?;

    let mut notification: Notification = record
        .entry()
        .to_app_option()?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to deserialize notification".to_string()
        )))?;

    notification.dismissed = true;

    let updated_hash = update_entry(notification_hash, notification)?;

    emit_signal(NotificationSignal::NotificationDismissed {
        notification_hash: updated_hash.clone(),
    })?;

    Ok(updated_hash)
}

/// Update notification preferences
#[hdk_extern]
pub fn update_notification_preferences(
    preferences: NotificationPreferences,
) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_latest_pubkey;

    // Ensure preferences are for the current agent
    if preferences.agent != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot update preferences for another agent".to_string()
        )));
    }

    // Get existing preferences if any
    let my_agent_hash = hash_entry(my_agent.clone())?;
    let links = get_links(
        GetLinksInputBuilder::try_new(my_agent_hash.clone(), LinkTypes::AgentToPreferences)?.build()
    )?;

    let prefs_hash = if links.is_empty() {
        // Create new preferences
        let hash = create_entry(EntryTypes::NotificationPreferences(preferences.clone()))?;
        create_link(my_agent_hash, hash.clone(), LinkTypes::AgentToPreferences, ())?;
        hash
    } else {
        // Update existing preferences
        let existing_hash = links[0]
            .target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid preference link".to_string()
            )))?;

        update_entry(existing_hash, preferences.clone())?
    };

    Ok(prefs_hash)
}

/// Get notification statistics
#[hdk_extern]
pub fn get_notification_stats(_: ()) -> ExternResult<NotificationStats> {
    let my_agent = agent_info()?.agent_latest_pubkey;
    let my_agent_hash = hash_entry(my_agent)?;

    let links = get_links(
        GetLinksInputBuilder::try_new(my_agent_hash, LinkTypes::AgentToNotifications)?.build()
    )?;

    let mut total = 0u32;
    let mut unread = 0u32;
    let mut by_type: HashMap<String, u32> = HashMap::new();
    let mut by_priority: HashMap<String, u32> = HashMap::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(notification) = record.entry().to_app_option::<Notification>()? {
                    if !notification.dismissed {
                        total += 1;

                        if !notification.read {
                            unread += 1;
                        }

                        // Count by type
                        let type_key = format!("{:?}", notification.notification_type);
                        *by_type.entry(type_key).or_insert(0) += 1;

                        // Count by priority
                        let priority_key = format!("{:?}", notification.priority);
                        *by_priority.entry(priority_key).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    Ok(NotificationStats {
        total,
        unread,
        by_type,
        by_priority,
    })
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Get or create default preferences for an agent
fn get_or_create_preferences(agent: AgentPubKey) -> ExternResult<NotificationPreferences> {
    let agent_hash = hash_entry(agent.clone())?;
    let links = get_links(
        GetLinksInputBuilder::try_new(agent_hash.clone(), LinkTypes::AgentToPreferences)?.build()
    )?;

    if links.is_empty() {
        // Create default preferences
        let mut prefs = NotificationPreferences::default();
        prefs.agent = agent;
        prefs.updated_at = sys_time()?.as_micros();

        let prefs_hash = create_entry(EntryTypes::NotificationPreferences(prefs.clone()))?;
        create_link(agent_hash, prefs_hash, LinkTypes::AgentToPreferences, ())?;

        Ok(prefs)
    } else {
        // Get existing preferences
        let prefs_hash = links[0]
            .target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid preference link".to_string()
            )))?;

        let record = get(prefs_hash, GetOptions::default())?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Preferences not found".to_string()
            )))?;

        record
            .entry()
            .to_app_option()?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Failed to deserialize preferences".to_string()
            )))
    }
}

/// Check if a notification should be included based on filter
fn should_include_notification(notification: &Notification, filter: &NotificationFilter) -> bool {
    // Check if dismissed (don't show dismissed by default unless explicitly requested)
    if notification.dismissed && !filter.include_dismissed.unwrap_or(false) {
        return false;
    }

    // Check read status
    if let Some(only_unread) = filter.only_unread {
        if only_unread && notification.read {
            return false;
        }
    }

    // Check notification type
    if let Some(ref types) = filter.notification_types {
        if !types.contains(&notification.notification_type) {
            return false;
        }
    }

    // Check priority
    if let Some(min_priority) = &filter.min_priority {
        if notification.priority < *min_priority {
            return false;
        }
    }

    // Check expiration
    if let Some(expires_at) = notification.expires_at {
        let now = match sys_time() {
            Ok(time) => time.as_micros(),
            Err(_) => return true, // Include if we can't check time
        };
        if now > expires_at {
            return false;
        }
    }

    true
}

/// Get current hour (0-23) from system time
fn get_current_hour() -> u8 {
    let now = match sys_time() {
        Ok(time) => time.as_micros(),
        Err(_) => return 12, // Default to noon if can't get time
    };

    // Convert microseconds to hours
    let hours_since_epoch = now / 1_000_000 / 60 / 60;
    (hours_since_epoch % 24) as u8
}

/// Check if current hour is within quiet hours
fn is_in_quiet_hours(current: u8, start: u8, end: u8) -> bool {
    if start < end {
        // Normal range (e.g., 22:00 to 07:00 next day)
        current >= start && current < end
    } else {
        // Wrap around midnight (e.g., 07:00 to 22:00)
        current >= start || current < end
    }
}

// ============================================================================
// TYPES FOR SIGNALS AND FILTERS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum NotificationSignal {
    NewNotification {
        notification_hash: ActionHash,
        notification: Notification,
    },
    NotificationRead {
        notification_hash: ActionHash,
    },
    NotificationDismissed {
        notification_hash: ActionHash,
    },
    AllNotificationsRead,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct NotificationFilter {
    pub only_unread: Option<bool>,
    pub notification_types: Option<Vec<NotificationType>>,
    pub min_priority: Option<NotificationPriority>,
    pub include_dismissed: Option<bool>,
    pub limit: Option<u32>,
}

// ============================================================================
// ENTRY AND LINK TYPE DEFINITIONS
// ============================================================================

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
