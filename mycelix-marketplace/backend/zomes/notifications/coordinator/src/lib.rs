//! Notifications Coordinator Zome - Business logic for user notifications
use hdk::prelude::*;
use notifications_integrity::*;
use std::collections::HashMap;

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NotificationStats {
    pub total: u32,
    pub unread: u32,
    pub by_type: HashMap<String, u32>,
    pub by_priority: HashMap<String, u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct NotificationFilter {
    pub only_unread: Option<bool>,
    pub notification_types: Option<Vec<NotificationType>>,
    pub min_priority: Option<NotificationPriority>,
    pub include_dismissed: Option<bool>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum NotificationSignal {
    NewNotification { notification_hash: ActionHash, notification: Notification },
    NotificationRead { notification_hash: ActionHash },
    NotificationDismissed { notification_hash: ActionHash },
    AllNotificationsRead,
}

fn now_micros() -> ExternResult<u64> {
    Ok(sys_time()?.as_micros() as u64)
}

fn get_agent_notification_links(agent: AgentPubKey) -> ExternResult<Vec<Link>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToNotifications)?;
    get_links(LinkQuery::new(agent, filter), GetStrategy::default())
}

fn get_agent_preferences_links(agent: AgentPubKey) -> ExternResult<Vec<Link>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToPreferences)?;
    get_links(LinkQuery::new(agent, filter), GetStrategy::default())
}

#[hdk_extern]
pub fn create_notification(notification: Notification) -> ExternResult<ActionHash> {
    if notification.title.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Notification title cannot be empty".into())));
    }
    if notification.message.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Notification message cannot be empty".into())));
    }

    let my_agent = agent_info()?.agent_initial_pubkey;
    let prefs = get_or_create_preferences(my_agent.clone())?;

    if !prefs.enabled_types.contains(&notification.notification_type) {
        return Err(wasm_error!(WasmErrorInner::Guest("Notification type disabled".into())));
    }
    if notification.priority < prefs.min_priority {
        return Err(wasm_error!(WasmErrorInner::Guest("Priority below threshold".into())));
    }

    let notification_hash = create_entry(EntryTypes::Notification(notification.clone()))?;
    create_link(my_agent, notification_hash.clone(), LinkTypes::AgentToNotifications, ())?;

    emit_signal(NotificationSignal::NewNotification {
        notification_hash: notification_hash.clone(),
        notification,
    })?;

    Ok(notification_hash)
}

#[hdk_extern]
pub fn get_my_notifications(filter: NotificationFilter) -> ExternResult<NotificationListResponse> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let links = get_agent_notification_links(my_agent)?;

    let mut notifications = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(notification) = record.entry().to_app_option::<Notification>().map_err(|e| wasm_error!(e))? {
                    if should_include_notification(&notification, &filter)? {
                        notifications.push(NotificationWithHash { hash: action_hash, notification });
                    }
                }
            }
        }
    }

    notifications.sort_by(|a, b| b.notification.created_at.cmp(&a.notification.created_at));
    if let Some(limit) = filter.limit {
        notifications.truncate(limit as usize);
    }

    let unread_count = notifications.iter().filter(|n| !n.notification.read).count() as u32;
    let total_count = notifications.len() as u32;

    Ok(NotificationListResponse { notifications, unread_count, total_count })
}

#[hdk_extern]
pub fn mark_notification_read(notification_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(notification_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Notification not found".into())))?;

    let mut notification: Notification = record.entry().to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to deserialize".into())))?;

    notification.read = true;
    notification.read_at = Some(now_micros()?);

    let updated_hash = update_entry(notification_hash, EntryTypes::Notification(notification))?;
    emit_signal(NotificationSignal::NotificationRead { notification_hash: updated_hash.clone() })?;
    Ok(updated_hash)
}

#[hdk_extern]
pub fn mark_all_read(_: ()) -> ExternResult<u32> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let links = get_agent_notification_links(my_agent)?;

    let mut marked_count = 0u32;
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut notification) = record.entry().to_app_option::<Notification>().map_err(|e| wasm_error!(e))? {
                    if !notification.read {
                        notification.read = true;
                        notification.read_at = Some(now_micros()?);
                        update_entry(action_hash, EntryTypes::Notification(notification))?;
                        marked_count += 1;
                    }
                }
            }
        }
    }

    emit_signal(NotificationSignal::AllNotificationsRead)?;
    Ok(marked_count)
}

#[hdk_extern]
pub fn dismiss_notification(notification_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(notification_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Notification not found".into())))?;

    let mut notification: Notification = record.entry().to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to deserialize".into())))?;

    notification.dismissed = true;
    let updated_hash = update_entry(notification_hash, EntryTypes::Notification(notification))?;
    emit_signal(NotificationSignal::NotificationDismissed { notification_hash: updated_hash.clone() })?;
    Ok(updated_hash)
}

#[hdk_extern]
pub fn update_notification_preferences(preferences: NotificationPreferences) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    if preferences.agent != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest("Cannot update preferences for another agent".into())));
    }

    let links = get_agent_preferences_links(my_agent.clone())?;

    if links.is_empty() {
        let hash = create_entry(EntryTypes::NotificationPreferences(preferences))?;
        create_link(my_agent, hash.clone(), LinkTypes::AgentToPreferences, ())?;
        Ok(hash)
    } else {
        let existing_hash = links[0].target.clone().into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid preference link".into())))?;
        update_entry(existing_hash, EntryTypes::NotificationPreferences(preferences))
    }
}

#[hdk_extern]
pub fn get_notification_stats(_: ()) -> ExternResult<NotificationStats> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let links = get_agent_notification_links(my_agent)?;

    let mut total = 0u32;
    let mut unread = 0u32;
    let mut by_type: HashMap<String, u32> = HashMap::new();
    let mut by_priority: HashMap<String, u32> = HashMap::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(notification) = record.entry().to_app_option::<Notification>().map_err(|e| wasm_error!(e))? {
                    if !notification.dismissed {
                        total += 1;
                        if !notification.read { unread += 1; }
                        *by_type.entry(format!("{:?}", notification.notification_type)).or_insert(0) += 1;
                        *by_priority.entry(format!("{:?}", notification.priority)).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    Ok(NotificationStats { total, unread, by_type, by_priority })
}

fn get_or_create_preferences(agent: AgentPubKey) -> ExternResult<NotificationPreferences> {
    let links = get_agent_preferences_links(agent.clone())?;

    if links.is_empty() {
        let mut prefs = NotificationPreferences::default();
        prefs.agent = agent.clone();
        prefs.updated_at = now_micros()?;
        let prefs_hash = create_entry(EntryTypes::NotificationPreferences(prefs.clone()))?;
        create_link(agent, prefs_hash, LinkTypes::AgentToPreferences, ())?;
        Ok(prefs)
    } else {
        let prefs_hash = links[0].target.clone().into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid preference link".into())))?;
        let record = get(prefs_hash, GetOptions::default())?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Preferences not found".into())))?;
        record.entry().to_app_option().map_err(|e| wasm_error!(e))?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to deserialize preferences".into())))
    }
}

fn should_include_notification(notification: &Notification, filter: &NotificationFilter) -> ExternResult<bool> {
    if notification.dismissed && !filter.include_dismissed.unwrap_or(false) { return Ok(false); }
    if let Some(only_unread) = filter.only_unread {
        if only_unread && notification.read { return Ok(false); }
    }
    if let Some(ref types) = filter.notification_types {
        if !types.contains(&notification.notification_type) { return Ok(false); }
    }
    if let Some(min_priority) = &filter.min_priority {
        if notification.priority < *min_priority { return Ok(false); }
    }
    if let Some(expires_at) = notification.expires_at {
        if now_micros()? > expires_at { return Ok(false); }
    }
    Ok(true)
}
