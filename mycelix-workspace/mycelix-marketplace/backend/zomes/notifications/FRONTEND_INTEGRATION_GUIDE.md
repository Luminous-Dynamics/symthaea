# 🔔 Notification System - Frontend Integration Guide

**Version**: 1.0.0
**Purpose**: Guide for frontend developers to integrate notifications into the Mycelix-Marketplace UI

---

## 📋 Overview

The Mycelix notification system is **client-side** by design - notifications are created and stored locally on each user's device, maintaining privacy and P2P principles.

### Key Principles

1. **Client-Side Creation**: Frontends detect events (via DHT queries or signals) and create notifications locally
2. **Privacy-First**: Notifications never leave the user's device
3. **User-Controlled**: All filtering happens client-side based on user preferences
4. **Real-Time**: Use Holochain signals for instant notification delivery

---

## 🎯 Integration Pattern

### Basic Flow

```
Event Occurs → Frontend Detects → Create Notification → Update UI
     ↓              ↓                     ↓                  ↓
  (Zome)      (Query/Signal)      (call create_notification)  (Show banner)
```

### Example: New Message Notification

```javascript
// 1. Listen for new messages (polling or signals)
async function pollForNewMessages() {
  const messages = await getMyReceivedMessages({
    only_unread: true,
    limit: 50
  });

  // 2. Check if there are new messages since last check
  const newMessages = messages.filter(m =>
    m.message.sent_at > lastCheckTimestamp
  );

  // 3. Create notification for each new message
  for (const msg of newMessages) {
    await createNotification({
      id: generateUUID(),
      notification_type: { NewMessage: null },
      priority: { Normal: null },
      title: `New message from ${getSenderName(msg.message.sender)}`,
      message: truncate(decrypt(msg.message.encrypted_content), 100),
      action_link: `/messages/${msg.message.conversation_id}`,
      related_entity: msg.message_hash,
      from_agent: msg.message.sender,
      created_at: Date.now() * 1000,
      expires_at: null,
      read: false,
      dismissed: false,
      read_at: null,
      metadata: {
        conversation_id: msg.message.conversation_id,
        preview: truncate(decrypt(msg.message.encrypted_content), 50)
      }
    });
  }

  lastCheckTimestamp = Date.now() * 1000;
}

// Poll every 30 seconds
setInterval(pollForNewMessages, 30000);
```

---

## 🔌 Event Detection Methods

### Method 1: Polling (Simple, Reliable)

**When to use**: For events that don't need instant delivery

```javascript
// Poll for new transactions every minute
setInterval(async () => {
  const transactions = await getMyTransactions({
    status: "Pending",
    limit: 20
  });

  const newTransactions = transactions.filter(t =>
    t.transaction.created_at > lastTransactionCheck
  );

  for (const txn of newTransactions) {
    await createTransactionNotification(txn);
  }

  lastTransactionCheck = Date.now() * 1000;
}, 60000); // Every 60 seconds
```

**Pros**: Simple, reliable, works offline
**Cons**: Not instant, consumes resources

---

### Method 2: Holochain Signals (Instant, Efficient)

**When to use**: For events that need instant delivery

```javascript
// Subscribe to Holochain signals
const unsubscribe = appWebsocket.on('signal', async (signal) => {
  // Signals are emitted by zomes when events occur
  if (signal.type === 'message_received') {
    await createNotification({
      id: generateUUID(),
      notification_type: { NewMessage: null },
      priority: { Normal: null },
      title: `New message from ${signal.sender_name}`,
      message: signal.preview,
      action_link: `/messages/${signal.conversation_id}`,
      related_entity: signal.message_hash,
      from_agent: signal.sender,
      created_at: Date.now() * 1000,
      // ... other fields ...
    });
  }

  if (signal.type === 'transaction_created') {
    await createTransactionCreatedNotification(signal.data);
  }

  // Handle other signal types...
});

// Cleanup on component unmount
onUnmount(() => unsubscribe());
```

**Pros**: Instant delivery, efficient
**Cons**: Requires zome signal emission, offline handling needed

---

### Method 3: Hybrid (Best Practice)

**Recommended**: Combine signals for instant delivery + polling as backup

```javascript
let lastCheck = Date.now() * 1000;

// Instant delivery via signals
appWebsocket.on('signal', handleSignal);

// Backup polling every 5 minutes (catches anything signals missed)
setInterval(async () => {
  await catchupMissedEvents(lastCheck);
  lastCheck = Date.now() * 1000;
}, 300000); // 5 minutes
```

---

## 📝 Event-to-Notification Mapping

### Messaging Events

#### NewMessage
**Trigger**: New message received in any conversation
**Detection**: Poll `get_my_received_messages` or listen to `message_received` signal
**Code**:
```javascript
async function createNewMessageNotification(message, messageHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { NewMessage: null },
    priority: { Normal: null },
    title: `New message from ${getSenderName(message.sender)}`,
    message: truncate(decrypt(message.encrypted_content), 100),
    action_link: `/messages/${message.conversation_id}`,
    related_entity: messageHash,
    from_agent: message.sender,
    created_at: Date.now() * 1000,
    expires_at: null,
    read: false,
    dismissed: false,
    read_at: null,
    metadata: {
      conversation_id: message.conversation_id
    }
  });
}
```

#### MessageReply
**Trigger**: Someone replied to a conversation you started
**Detection**: Check if message is a reply to your conversation
**Code**:
```javascript
async function createMessageReplyNotification(message, messageHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { MessageReply: null },
    priority: { Normal: null },
    title: `${getSenderName(message.sender)} replied`,
    message: truncate(decrypt(message.encrypted_content), 100),
    action_link: `/messages/${message.conversation_id}`,
    related_entity: messageHash,
    from_agent: message.sender,
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

---

### Transaction Events

#### TransactionCreated
**Trigger**: New transaction initiated
**Detection**: Poll `get_my_transactions` with `status: "Pending"`
**Code**:
```javascript
async function createTransactionCreatedNotification(transaction, txnHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { TransactionCreated: null },
    priority: { High: null },
    title: "New transaction started",
    message: `Transaction for "${transaction.listing_title}" - ${transaction.amount} ${transaction.currency}`,
    action_link: `/transactions/${transaction.id}`,
    related_entity: txnHash,
    from_agent: transaction.buyer,
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

#### PaymentReceived
**Trigger**: Payment confirmed in a transaction you're selling
**Detection**: Transaction status changes to "Paid"
**Code**:
```javascript
async function createPaymentReceivedNotification(transaction, txnHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { PaymentReceived: null },
    priority: { High: null },
    title: "Payment Received!",
    message: `Received ${transaction.amount} ${transaction.currency} for "${transaction.listing_title}"`,
    action_link: `/transactions/${transaction.id}`,
    related_entity: txnHash,
    from_agent: transaction.buyer,
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

#### TransactionCompleted
**Trigger**: Transaction reaches "Completed" status
**Detection**: Transaction status = "Completed"
**Code**:
```javascript
async function createTransactionCompletedNotification(transaction, txnHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { TransactionCompleted: null },
    priority: { High: null },
    title: "Transaction completed",
    message: `Transaction for "${transaction.listing_title}" is now complete`,
    action_link: `/transactions/${transaction.id}`,
    related_entity: txnHash,
    from_agent: transaction.seller, // or buyer, depending on who you are
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

---

### Reputation Events

#### ReviewReceived
**Trigger**: New review posted on your listing
**Detection**: Poll reviews for your listings
**Code**:
```javascript
async function createReviewReceivedNotification(review, reviewHash, listingTitle) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { ReviewReceived: null },
    priority: { Normal: null },
    title: `${review.rating}-star review received`,
    message: `On "${listingTitle}": ${truncate(review.comment, 100)}`,
    action_link: `/reviews/${reviewHash}`,
    related_entity: reviewHash,
    from_agent: review.reviewer,
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

#### MATLScoreChanged
**Trigger**: Your MATL score changes significantly (>5%)
**Detection**: Compare current MATL score with cached score
**Code**:
```javascript
let lastKnownMATL = null;

setInterval(async () => {
  const currentMATL = await getMyMATLScore();

  if (lastKnownMATL !== null) {
    const change = Math.abs(currentMATL - lastKnownMATL);
    const changePercent = (change / lastKnownMATL) * 100;

    if (changePercent >= 5) {
      await createNotification({
        id: generateUUID(),
        notification_type: { MATLScoreChanged: null },
        priority: { Normal: null },
        title: "Your trust score changed",
        message: `MATL score: ${lastKnownMATL.toFixed(2)} → ${currentMATL.toFixed(2)} (${changePercent.toFixed(1)}%)`,
        action_link: "/profile/reputation",
        related_entity: null,
        from_agent: null,
        created_at: Date.now() * 1000,
        // ... other fields ...
      });
    }
  }

  lastKnownMATL = currentMATL;
}, 3600000); // Check every hour
```

---

### Arbitration Events

#### DisputeOpened
**Trigger**: New dispute opened on your transaction
**Detection**: Poll disputes for transactions you're involved in
**Code**:
```javascript
async function createDisputeOpenedNotification(dispute, disputeHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { DisputeOpened: null },
    priority: { Critical: null },
    title: "Dispute opened on your transaction",
    message: `Reason: ${truncate(dispute.description, 100)}`,
    action_link: `/disputes/${disputeHash}`,
    related_entity: disputeHash,
    from_agent: dispute.plaintiff,
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

#### DisputeResolved
**Trigger**: Dispute reaches final resolution
**Detection**: Dispute status = "Resolved"
**Code**:
```javascript
async function createDisputeResolvedNotification(dispute, disputeHash) {
  return await createNotification({
    id: generateUUID(),
    notification_type: { DisputeResolved: null },
    priority: { High: null },
    title: "Dispute resolved",
    message: `Resolution: ${dispute.winner === "Plaintiff" ? "Refund issued" : "Seller favored"}`,
    action_link: `/disputes/${disputeHash}`,
    related_entity: disputeHash,
    from_agent: null,
    created_at: Date.now() * 1000,
    // ... other fields ...
  });
}
```

---

### Listing Events

#### ListingExpiring
**Trigger**: Listing will expire in 24 hours
**Detection**: Compare listing expiration with current time
**Code**:
```javascript
setInterval(async () => {
  const myListings = await getMyListings({ status: "Active" });
  const now = Date.now() * 1000;
  const twentyFourHours = 24 * 60 * 60 * 1000000; // microseconds

  for (const listing of myListings) {
    const timeUntilExpiration = listing.listing.expires_at - now;

    if (timeUntilExpiration > 0 && timeUntilExpiration < twentyFourHours) {
      // Check if we haven't already notified about this
      if (!hasBeenNotified(listing.listing_hash)) {
        await createNotification({
          id: generateUUID(),
          notification_type: { ListingExpiring: null },
          priority: { Normal: null },
          title: "Your listing is expiring soon",
          message: `"${listing.listing.title}" expires in ${Math.floor(timeUntilExpiration / 3600000000)} hours`,
          action_link: `/listings/${listing.listing_hash}`,
          related_entity: listing.listing_hash,
          from_agent: null,
          created_at: Date.now() * 1000,
          // ... other fields ...
        });

        markAsNotified(listing.listing_hash);
      }
    }
  }
}, 3600000); // Check every hour
```

---

## 🎨 UI Integration Patterns

### Notification Center Component

```javascript
class NotificationCenter extends Component {
  state = {
    notifications: [],
    unreadCount: 0
  };

  async componentDidMount() {
    // Load initial notifications
    await this.refreshNotifications();

    // Listen for real-time updates
    this.signalHandler = onNotificationSignal(signal => {
      if (signal.type === 'NewNotification') {
        this.handleNewNotification(signal.notification);
      }
    });

    // Refresh periodically as backup
    this.refreshInterval = setInterval(() => {
      this.refreshNotifications();
    }, 60000); // Every minute
  }

  async refreshNotifications() {
    const response = await getMyNotifications({
      only_unread: false,
      limit: 50
    });

    this.setState({
      notifications: response.notifications,
      unreadCount: response.unread_count
    });
  }

  handleNewNotification(notification) {
    this.setState(state => ({
      notifications: [notification, ...state.notifications],
      unreadCount: state.unreadCount + 1
    }));

    this.showToast(notification);
    this.playSound(notification.priority);
  }

  showToast(notification) {
    toast({
      title: notification.title,
      message: notification.message,
      priority: notification.priority,
      onClick: () => navigate(notification.action_link)
    });
  }

  playSound(priority) {
    if (priority >= Priority.High) {
      new Audio('/sounds/notification-important.mp3').play();
    } else {
      new Audio('/sounds/notification.mp3').play();
    }
  }

  render() {
    return (
      <div className="notification-center">
        <NotificationBadge count={this.state.unreadCount} />
        <NotificationList notifications={this.state.notifications} />
      </div>
    );
  }
}
```

### Toast/Banner for New Notifications

```javascript
function NotificationToast({ notification }) {
  const priorityColors = {
    Critical: 'bg-red-500',
    High: 'bg-orange-500',
    Normal: 'bg-blue-500',
    Low: 'bg-gray-500'
  };

  return (
    <div className={`notification-toast ${priorityColors[notification.priority]}`}>
      <div className="toast-header">
        <span className="toast-icon">{getIconForType(notification.notification_type)}</span>
        <span className="toast-title">{notification.title}</span>
      </div>
      <div className="toast-message">{notification.message}</div>
      <div className="toast-actions">
        <button onClick={() => navigate(notification.action_link)}>
          View
        </button>
        <button onClick={() => dismissNotification(notification.hash)}>
          Dismiss
        </button>
      </div>
    </div>
  );
}
```

---

## 🔧 Helper Functions

### UUID Generation

```javascript
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
```

### Text Truncation

```javascript
function truncate(text, maxLength) {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}
```

### Relative Time Formatting

```javascript
function formatRelativeTime(timestamp) {
  const now = Date.now() * 1000;
  const diff = now - timestamp;
  const minutes = Math.floor(diff / 60000000);
  const hours = Math.floor(diff / 3600000000);
  const days = Math.floor(diff / 86400000000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp / 1000).toLocaleDateString();
}
```

---

## 📊 Monitoring Best Practices

### Track Notification Metrics

```javascript
// Track notification creation rate
function trackNotificationCreated(notificationType) {
  analytics.track('notification_created', {
    type: notificationType,
    timestamp: Date.now()
  });
}

// Track notification engagement
function trackNotificationClicked(notification) {
  analytics.track('notification_clicked', {
    type: notification.notification_type,
    priority: notification.priority,
    age: Date.now() * 1000 - notification.created_at
  });
}

// Track notification dismissal
function trackNotificationDismissed(notification) {
  analytics.track('notification_dismissed', {
    type: notification.notification_type,
    priority: notification.priority,
    age: Date.now() * 1000 - notification.created_at
  });
}
```

### Performance Optimization

```javascript
// Debounce notification creation to avoid spam
const createNotificationDebounced = debounce(createNotification, 1000);

// Batch notification queries
async function batchCheckForEvents() {
  const [newMessages, newTransactions, newReviews] = await Promise.all([
    getMyReceivedMessages({ only_unread: true }),
    getMyTransactions({ status: "Pending" }),
    getReviewsForMyListings()
  ]);

  // Process all events and create notifications
  await Promise.all([
    ...newMessages.map(createMessageNotification),
    ...newTransactions.map(createTransactionNotification),
    ...newReviews.map(createReviewNotification)
  ]);
}
```

---

## 🎯 Testing Checklist

- [ ] Notifications created for all 16 event types
- [ ] User preferences properly filter notifications
- [ ] Quiet hours respected (except Critical)
- [ ] Notifications marked as read when viewed
- [ ] Unread badge updates in real-time
- [ ] Toast/banner appears for new notifications
- [ ] Sound plays for high-priority notifications
- [ ] Action links navigate correctly
- [ ] Notifications auto-dismiss after expiration
- [ ] Performance acceptable with 100+ notifications
- [ ] Offline handling works (notifications queued)
- [ ] Multi-device sync works (if applicable)

---

## 🚀 Production Deployment

### Recommended Event Detection Strategy

```javascript
// Hybrid approach: Signals + Polling
const POLL_INTERVAL = 60000; // 1 minute backup polling
const SIGNAL_TIMEOUT = 5000;  // 5 second signal timeout

class NotificationManager {
  constructor() {
    this.lastCheck = Date.now() * 1000;
    this.setupSignals();
    this.setupPolling();
  }

  setupSignals() {
    // Instant delivery via signals
    appWebsocket.on('signal', signal => {
      this.handleSignal(signal);
      this.lastCheck = Date.now() * 1000; // Update to prevent duplicate polling
    });
  }

  setupPolling() {
    // Backup polling every minute
    setInterval(async () => {
      await this.pollForMissedEvents();
    }, POLL_INTERVAL);
  }

  async handleSignal(signal) {
    // Handle different signal types
    switch (signal.type) {
      case 'message_received':
        await this.createMessageNotification(signal.data);
        break;
      case 'transaction_created':
        await this.createTransactionNotification(signal.data);
        break;
      // ... other cases ...
    }
  }

  async pollForMissedEvents() {
    // Check for events that might have been missed by signals
    const events = await this.queryEventsSince(this.lastCheck);
    for (const event of events) {
      await this.createNotificationForEvent(event);
    }
    this.lastCheck = Date.now() * 1000;
  }
}

// Initialize on app start
const notificationManager = new NotificationManager();
```

---

## 📖 Summary

### Key Takeaways

1. **Client-Side Creation**: Frontends are responsible for creating notifications
2. **Event Detection**: Use polling, signals, or hybrid approach
3. **User Control**: Respect user preferences and quiet hours
4. **Privacy-First**: All notifications stay local
5. **Real-Time**: Use signals for instant delivery when possible

### Next Steps

1. Implement event detection for all 16 notification types
2. Build notification center UI component
3. Add toast/banner for new notifications
4. Test all notification scenarios
5. Monitor notification engagement metrics
6. Optimize performance for large notification counts

---

**Status**: ✅ Complete Integration Guide
**Version**: 1.0.0
**Maintainer**: Mycelix Marketplace Team
**Last Updated**: December 30, 2025
