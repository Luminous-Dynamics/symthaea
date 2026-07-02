# 🔔 Notification System - Architecture

**Purpose**: Real-time, intelligent notification system that keeps users informed of important marketplace events without overwhelming them.

---

## 🎯 Vision

The Mycelix-Marketplace notification system is **the most intelligent notification system ever built for a marketplace**, featuring:

- 🎯 **Smart Prioritization** - Critical notifications first, noise filtered
- 🔕 **User Control** - Granular preferences for every notification type
- ⚡ **Real-Time** - Instant delivery via WebSocket + DHT signals
- 📊 **Actionable** - Every notification has a clear action (view, respond, resolve)
- 🧠 **Learning** - System learns from user behavior to reduce noise
- 🔐 **Privacy-First** - End-to-end encrypted notification content
- 📱 **Multi-Channel** - In-app, push, email, SMS (future)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Event Sources                          │
│  (Listings, Transactions, Messages, Arbitration, etc.)  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────┐
│              Notification Generator                      │
│  - Creates notification from event                       │
│  - Applies user preferences                              │
│  - Determines priority                                   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────┐
│            Notification Storage (DHT)                    │
│  - Stores notification entries                           │
│  - Links to user                                         │
│  - Tracks read/unread status                            │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────┐
│          Delivery Channels                               │
│  ├─ In-App (WebSocket signals)                          │
│  ├─ Push Notifications (future)                         │
│  ├─ Email (future)                                      │
│  └─ SMS (future)                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Data Model

### Notification Entry

```rust
#[hdk_entry_helper]
pub struct Notification {
    /// Unique notification ID
    pub notification_id: String,

    /// Recipient agent
    pub recipient: AgentPubKey,

    /// Notification type
    pub notification_type: NotificationType,

    /// Priority level
    pub priority: NotificationPriority,

    /// Title (brief summary)
    pub title: String,

    /// Message (detailed information)
    pub message: String,

    /// Related entity
    pub entity_type: Option<EntityType>,
    pub entity_hash: Option<ActionHash>,

    /// Action information
    pub action_type: Option<NotificationAction>,
    pub action_url: Option<String>,

    /// Metadata
    pub created_at: u64,
    pub read_at: Option<u64>,
    pub dismissed_at: Option<u64>,
    pub expires_at: Option<u64>,

    /// Grouping (for batch notifications)
    pub group_id: Option<String>,
    pub group_count: u32,
}
```

### Notification Types

```rust
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum NotificationType {
    // Messaging
    NewMessage,
    ConversationStarted,

    // Transactions
    TransactionCreated,
    TransactionPaid,
    TransactionShipped,
    TransactionCompleted,
    TransactionCancelled,

    // Listings
    ListingSold,
    ListingExpiring,
    PriceDropAlert,
    BackInStock,

    // Reputation
    ReviewReceived,
    MatlScoreChanged,

    // Arbitration
    DisputeCreated,
    VoteCast,
    DisputeResolved,

    // System
    SystemAnnouncement,
    SecurityAlert,
    MaintenanceNotice,
}
```

### Priority Levels

```rust
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum NotificationPriority {
    Critical,   // Immediate attention required (security, disputes)
    High,       // Important (transactions, messages)
    Normal,     // Standard (reviews, updates)
    Low,        // Informational (price drops, announcements)
}
```

### Notification Actions

```rust
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum NotificationAction {
    ViewMessage,
    ViewTransaction,
    ViewDispute,
    ViewListing,
    ViewReview,
    PayNow,
    ShipNow,
    VoteNow,
    Dismiss,
    None,
}
```

### User Preferences

```rust
#[hdk_entry_helper]
pub struct NotificationPreferences {
    /// User agent
    pub agent: AgentPubKey,

    /// Enabled notification types
    pub enabled_types: Vec<NotificationType>,

    /// Disabled notification types
    pub disabled_types: Vec<NotificationType>,

    /// Channel preferences
    pub in_app_enabled: bool,
    pub push_enabled: bool,
    pub email_enabled: bool,
    pub sms_enabled: bool,

    /// Quiet hours (no non-critical notifications)
    pub quiet_hours_start: Option<u32>,  // Hour of day (0-23)
    pub quiet_hours_end: Option<u32>,

    /// Grouping preferences
    pub group_similar: bool,
    pub group_threshold: u32,  // Group if N+ similar notifications

    /// Priority filters
    pub minimum_priority: NotificationPriority,

    /// Updated timestamp
    pub updated_at: u64,
}
```

---

## 🎯 API Endpoints

### 1. Create Notification

```rust
#[hdk_extern]
pub fn create_notification(input: CreateNotificationInput) -> ExternResult<NotificationOutput>
```

**Called by**: Other zomes when events occur

**Input**:
```rust
pub struct CreateNotificationInput {
    pub recipient: AgentPubKey,
    pub notification_type: NotificationType,
    pub priority: NotificationPriority,
    pub title: String,
    pub message: String,
    pub entity_type: Option<EntityType>,
    pub entity_hash: Option<ActionHash>,
    pub action_type: Option<NotificationAction>,
    pub action_url: Option<String>,
    pub expires_in_hours: Option<u32>,
}
```

**Output**:
```rust
pub struct NotificationOutput {
    pub notification_hash: ActionHash,
    pub notification: Notification,
    pub delivered: bool,  // Was it delivered immediately?
}
```

### 2. Get My Notifications

```rust
#[hdk_extern]
pub fn get_my_notifications(filters: NotificationFilters) -> ExternResult<NotificationsResponse>
```

**Input**:
```rust
pub struct NotificationFilters {
    pub types: Option<Vec<NotificationType>>,
    pub priorities: Option<Vec<NotificationPriority>>,
    pub read_status: Option<ReadStatus>,  // Unread | Read | All
    pub entity_type: Option<EntityType>,
    pub limit: u32,
    pub offset: u32,
}
```

**Output**:
```rust
pub struct NotificationsResponse {
    pub notifications: Vec<NotificationOutput>,
    pub unread_count: u32,
    pub total_count: u32,
}
```

### 3. Mark as Read

```rust
#[hdk_extern]
pub fn mark_notification_read(notification_hash: ActionHash) -> ExternResult<NotificationOutput>
```

**Effect**: Updates `read_at` timestamp, decrements unread count

### 4. Mark All as Read

```rust
#[hdk_extern]
pub fn mark_all_read(filters: Option<NotificationFilters>) -> ExternResult<u32>
```

**Returns**: Number of notifications marked as read

### 5. Dismiss Notification

```rust
#[hdk_extern]
pub fn dismiss_notification(notification_hash: ActionHash) -> ExternResult<NotificationOutput>
```

**Effect**: Sets `dismissed_at`, hides from list

### 6. Update Preferences

```rust
#[hdk_extern]
pub fn update_notification_preferences(
    preferences: NotificationPreferences
) -> ExternResult<NotificationPreferences>
```

**Effect**: Updates user's notification settings

### 7. Get Notification Statistics

```rust
#[hdk_extern]
pub fn get_notification_stats() -> ExternResult<NotificationStats>
```

**Output**:
```rust
pub struct NotificationStats {
    pub total_notifications: u32,
    pub unread_count: u32,
    pub by_type: HashMap<NotificationType, u32>,
    pub by_priority: HashMap<NotificationPriority, u32>,
    pub today_count: u32,
    pub this_week_count: u32,
}
```

---

## 🔔 Notification Triggers

### Messaging Zome Integration

```rust
// In messaging zome after sending message
pub fn send_message(input: SendMessageInput) -> ExternResult<MessageOutput> {
    // ... create message ...

    // Trigger notification
    call_remote(
        None,
        "notifications".into(),
        "create_notification".into(),
        None,
        &CreateNotificationInput {
            recipient: input.recipient.clone(),
            notification_type: NotificationType::NewMessage,
            priority: NotificationPriority::High,
            title: format!("New message from {}", sender_name),
            message: "You have a new message".to_string(),
            entity_type: Some(EntityType::Message),
            entity_hash: Some(message_hash.clone()),
            action_type: Some(NotificationAction::ViewMessage),
            action_url: Some(format!("/messages/{}", conversation_id)),
            expires_in_hours: Some(168),  // 7 days
        },
    )?;

    Ok(message_output)
}
```

### Transaction Status Changes

```rust
// When transaction status changes
pub fn update_transaction_status(
    transaction_hash: ActionHash,
    new_status: TransactionStatus,
) -> ExternResult<TransactionOutput> {
    // ... update status ...

    // Notify buyer
    call_remote(
        None,
        "notifications".into(),
        "create_notification".into(),
        None,
        &CreateNotificationInput {
            recipient: transaction.buyer.clone(),
            notification_type: match new_status {
                TransactionStatus::Shipped => NotificationType::TransactionShipped,
                TransactionStatus::Completed => NotificationType::TransactionCompleted,
                _ => NotificationType::TransactionPaid,
            },
            priority: NotificationPriority::High,
            title: format!("Transaction {}", status_to_string(&new_status)),
            message: format!("Your order has been {}", status_to_string(&new_status)),
            entity_type: Some(EntityType::Transaction),
            entity_hash: Some(transaction_hash.clone()),
            action_type: Some(NotificationAction::ViewTransaction),
            action_url: Some(format!("/transactions/{}", transaction_hash)),
            expires_in_hours: Some(72),
        },
    )?;

    Ok(transaction_output)
}
```

### Review Notifications

```rust
// When receiving a review
pub fn submit_review(input: ReviewInput) -> ExternResult<ReviewOutput> {
    // ... create review ...

    // Notify reviewed agent
    call_remote(
        None,
        "notifications".into(),
        "create_notification".into(),
        None,
        &CreateNotificationInput {
            recipient: input.reviewee.clone(),
            notification_type: NotificationType::ReviewReceived,
            priority: NotificationPriority::Normal,
            title: "New review received".to_string(),
            message: format!("You received a {} star review", input.rating),
            entity_type: Some(EntityType::Review),
            entity_hash: Some(review_hash.clone()),
            action_type: Some(NotificationAction::ViewReview),
            action_url: Some(format!("/reviews/{}", review_hash)),
            expires_in_hours: Some(720),  // 30 days
        },
    )?;

    Ok(review_output)
}
```

---

## ⚡ Real-Time Delivery

### WebSocket Signal Protocol

```typescript
// Frontend subscribes to signals
holochain.on('signal', (signal) => {
  if (signal.type === 'notification') {
    const notification = signal.data;

    // Display toast/banner
    showNotificationBanner(notification);

    // Update notification center
    updateNotificationCenter(notification);

    // Play sound (if enabled)
    if (notification.priority === 'Critical' || notification.priority === 'High') {
      playNotificationSound();
    }

    // Show badge count
    updateUnreadBadge(signal.unread_count);
  }
});
```

### Signal Emission (Backend)

```rust
// After creating notification
let signal = NotificationSignal {
    notification_type: notification.notification_type.clone(),
    notification_hash: notification_hash.clone(),
    priority: notification.priority.clone(),
    title: notification.title.clone(),
    message: notification.message.clone(),
    action_url: notification.action_url.clone(),
    unread_count: get_unread_count(&notification.recipient)?,
};

emit_signal(&signal)?;
```

---

## 🎨 Smart Features

### 1. Notification Grouping

**Problem**: 10 messages from same person = 10 notifications (annoying!)

**Solution**: Group similar notifications

```rust
// Check if should group
if preferences.group_similar {
    let similar = find_similar_notifications(
        &notification.recipient,
        &notification.notification_type,
        notification.created_at - 3600,  // Last hour
    )?;

    if similar.len() >= preferences.group_threshold {
        // Create grouped notification
        return create_grouped_notification(similar, new_notification);
    }
}
```

**Result**:
```
Instead of:
- "New message from Alice"
- "New message from Alice"
- "New message from Alice"

Show:
- "3 new messages from Alice"
```

### 2. Priority-Based Filtering

```rust
// Apply priority filter
pub fn should_deliver(
    notification: &Notification,
    preferences: &NotificationPreferences,
    current_time: u64,
) -> bool {
    // Always deliver critical
    if notification.priority == NotificationPriority::Critical {
        return true;
    }

    // Check if below minimum priority
    if notification.priority < preferences.minimum_priority {
        return false;
    }

    // Check quiet hours
    if is_quiet_hours(current_time, preferences) {
        return notification.priority >= NotificationPriority::High;
    }

    // Check if type is disabled
    if preferences.disabled_types.contains(&notification.notification_type) {
        return false;
    }

    true
}
```

### 3. Smart Expiration

```rust
// Auto-expire old notifications
pub fn cleanup_expired_notifications() -> ExternResult<u32> {
    let now = sys_time()?;
    let all_notifications = get_all_notifications()?;

    let mut expired_count = 0;

    for notification in all_notifications {
        if let Some(expires_at) = notification.expires_at {
            if now > expires_at {
                dismiss_notification(notification.notification_hash)?;
                expired_count += 1;
            }
        }
    }

    Ok(expired_count)
}
```

### 4. Actionable Notifications

Every notification has a clear action:

```typescript
function renderNotification(notification: Notification) {
  return (
    <div className="notification" priority={notification.priority}>
      <div className="content">
        <h4>{notification.title}</h4>
        <p>{notification.message}</p>
        <small>{formatTime(notification.created_at)}</small>
      </div>

      <div className="actions">
        {notification.action_type === 'ViewMessage' && (
          <button onClick={() => navigate(notification.action_url)}>
            View Message
          </button>
        )}
        {notification.action_type === 'PayNow' && (
          <button onClick={() => navigate(notification.action_url)}>
            Pay Now
          </button>
        )}
        <button onClick={() => dismiss(notification.notification_hash)}>
          Dismiss
        </button>
      </div>
    </div>
  );
}
```

---

## 📊 Monitoring Integration

### New Metrics

```rust
pub enum MetricType {
    // ... existing metrics ...

    /// Notification created
    NotificationCreated,

    /// Notification delivered
    NotificationDelivered,

    /// Notification read
    NotificationRead,

    /// Notification dismissed
    NotificationDismissed,

    /// Notification action clicked
    NotificationActionClicked,
}
```

### Analytics Queries

```javascript
// Get notification delivery rate
const delivered = await getMetric('NotificationDelivered', timeWindow);
const created = await getMetric('NotificationCreated', timeWindow);
const deliveryRate = (delivered / created) * 100;

// Get read rate
const read = await getMetric('NotificationRead', timeWindow);
const readRate = (read / delivered) * 100;

// Get action click-through rate
const clicked = await getMetric('NotificationActionClicked', timeWindow);
const ctr = (clicked / delivered) * 100;

console.log(`Delivery: ${deliveryRate}% | Read: ${readRate}% | CTR: ${ctr}%`);
```

---

## 🎯 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Notification creation** | < 50ms | Including preference check |
| **Real-time delivery** | < 100ms | WebSocket signal emission |
| **Get notifications** | < 100ms | With pagination |
| **Mark as read** | < 50ms | Update entry only |
| **Memory usage** | < 100MB | For 10K notifications |

---

## 🔮 Future Enhancements

### Phase 5: Advanced Features

1. **Push Notifications** - Mobile push via FCM/APNS
2. **Email Notifications** - Digest emails for important events
3. **SMS Notifications** - Critical alerts via SMS
4. **Notification Templates** - Customizable notification messages
5. **Rich Media** - Images, videos in notifications
6. **Smart Scheduling** - ML-based optimal delivery times
7. **Notification Insights** - Personal analytics on notification patterns

---

## 📈 Success Metrics

**Technical**:
- Delivery success rate > 99.5%
- Real-time latency < 100ms (p95)
- False positive rate < 5% (unwanted notifications)
- User engagement rate > 60% (clicked/viewed)

**User Experience**:
- Unread notifications cleared < 1 hour average
- User satisfaction > 4.5/5
- Notification opt-out rate < 10%
- Action click-through rate > 30%

---

**Status**: Ready for implementation 🚀
**Dependencies**: None (standalone utility module)
**Priority**: High (critical for UX)

---

*"The best notification system is invisible until you need it, then indispensable."* 🔔
