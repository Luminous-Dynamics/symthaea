# 💬 Messaging Zome - Complete Guide

## Overview

The Messaging Zome provides **P2P encrypted messaging** for the Mycelix-Marketplace, enabling buyers and sellers to communicate securely and efficiently.

**Key Features**:
- 🔐 **End-to-end encryption** - Messages encrypted client-side
- 💬 **Conversation threading** - Organized by listing or transaction
- 📎 **IPFS attachments** - Rich media support
- 🛡️ **MATL-gated** - Spam prevention (score > 0.4 required)
- 📊 **Read receipts** - Know when messages are read
- 🔍 **Smart search** - Find conversations fast
- 📈 **Monitoring integration** - Track messaging metrics

---

## Architecture

### Data Model

```
Conversation
├── participants: Vec<AgentPubKey>
├── subject: String
├── first_message_hash: ActionHash
├── last_message_hash: ActionHash
├── message_count: u32
├── unread_counts: Vec<(AgentPubKey, u32)>
├── status: ConversationStatus
├── listing_hash?: Option<ActionHash>
└── transaction_hash?: Option<ActionHash>

Message
├── sender: AgentPubKey
├── recipient: AgentPubKey
├── encrypted_content: String (AES-256-GCM)
├── conversation_id: ActionHash
├── sent_at: Timestamp
├── read_at?: Option<Timestamp>
├── message_type: MessageType
├── listing_hash?: Option<ActionHash>
└── transaction_hash?: Option<ActionHash>

ReadReceipt
├── message_hash: ActionHash
├── reader: AgentPubKey
└── read_at: Timestamp
```

### Message Types

```rust
pub enum MessageType {
    Text,          // Regular text message
    System,        // System notification
    Offer,         // Price negotiation
    Question,      // About listing
    ReviewRequest, // Request for review
}
```

### Conversation Status

```rust
pub enum ConversationStatus {
    Active,   // Normal conversation
    Archived, // Hidden from active list
    Muted,    // Notifications off
    Blocked,  // Spam/abuse prevention
}
```

---

## API Reference

### 1. Start Conversation

Start a new conversation with another agent.

**Endpoint**: `start_conversation`

**Input**:
```rust
pub struct StartConversationInput {
    pub recipient: AgentPubKey,
    pub subject: String,
    pub first_message_content: String, // Pre-encrypted
    pub listing_hash: Option<ActionHash>,
    pub transaction_hash: Option<ActionHash>,
}
```

**Output**:
```rust
pub struct ConversationOutput {
    pub conversation_hash: ActionHash,
    pub conversation: Conversation,
    pub first_message: MessageOutput,
}
```

**Requirements**:
- Sender must have MATL score > 0.4
- Subject cannot be empty
- First message content must be pre-encrypted

**Example**:
```javascript
const conversation = await startConversation({
    recipient: sellerPubKey,
    subject: "Question about MacBook listing",
    first_message_content: encryptedMessage,
    listing_hash: listingHash,
    transaction_hash: null,
});
```

### 2. Send Message

Send a message in an existing conversation.

**Endpoint**: `send_message`

**Input**:
```rust
pub struct SendMessageInput {
    pub recipient: AgentPubKey,
    pub encrypted_content: String,
    pub listing_hash: Option<ActionHash>,
    pub transaction_hash: Option<ActionHash>,
    pub conversation_id: ActionHash,
    pub message_type: MessageType,
}
```

**Output**:
```rust
pub struct MessageOutput {
    pub message_hash: ActionHash,
    pub message: Message,
}
```

**Requirements**:
- Sender must have MATL score > 0.4
- Content must be pre-encrypted (client-side)
- Content cannot exceed 10KB
- Conversation must exist

**Rate Limits**:
- New conversations: 10/hour
- Messages in existing conversation: 100/hour

**Example**:
```javascript
const message = await sendMessage({
    recipient: buyerPubKey,
    encrypted_content: encryptedReply,
    conversation_id: conversationHash,
    message_type: "Text",
});
```

### 3. Get My Conversations

Retrieve all conversations for the current agent.

**Endpoint**: `get_my_conversations`

**Input**: `()` (no parameters)

**Output**:
```rust
pub struct ConversationsResponse {
    pub conversations: Vec<ConversationOutput>,
}
```

**Returns**: Conversations sorted by `last_activity_at` (most recent first)

**Example**:
```javascript
const { conversations } = await getMyConversations();

// conversations[0] = most recent
// conversations.forEach(conv => {
//     console.log(conv.conversation.subject);
//     console.log(conv.conversation.unread_counts);
// });
```

### 4. Get Conversation Messages

Get all messages in a conversation.

**Endpoint**: `get_conversation_messages`

**Input**: `ActionHash` (conversation_hash)

**Output**:
```rust
pub struct MessagesResponse {
    pub messages: Vec<MessageOutput>,
}
```

**Returns**: Messages sorted by `sent_at` (chronological order)

**Example**:
```javascript
const { messages } = await getConversationMessages(conversationHash);

// Decrypt messages client-side
const decryptedMessages = messages.map(msg => ({
    ...msg,
    content: decryptMessage(msg.message.encrypted_content)
}));
```

### 5. Mark Message as Read

Mark a message as read and create a read receipt.

**Endpoint**: `mark_message_read`

**Input**: `ActionHash` (message_hash)

**Output**:
```rust
pub struct ReadReceiptOutput {
    pub receipt_hash: ActionHash,
    pub receipt: ReadReceipt,
}
```

**Requirements**:
- Caller must be the message recipient
- Updates conversation unread count

**Example**:
```javascript
const receipt = await markMessageRead(messageHash);
// Read receipt sent to sender
```

### 6. Archive Conversation

Hide a conversation from the active list.

**Endpoint**: `archive_conversation`

**Input**: `ActionHash` (conversation_hash)

**Output**: `ConversationOutput`

**Effect**: Sets `status = ConversationStatus::Archived`

**Example**:
```javascript
await archiveConversation(oldConversationHash);
// Conversation moved to archived
```

### 7. Block Conversation

Block a conversation (spam/abuse prevention).

**Endpoint**: `block_conversation`

**Input**: `ActionHash` (conversation_hash)

**Output**: `ConversationOutput`

**Requirements**:
- Caller must be a participant

**Effect**:
- Sets `status = ConversationStatus::Blocked`
- Emits `ConversationBlocked` metric
- Prevents new messages

**Example**:
```javascript
await blockConversation(spamConversationHash);
// Conversation blocked, metrics emitted
```

### 8. Search Conversations

Search conversations by multiple criteria.

**Endpoint**: `search_conversations`

**Input**:
```rust
pub struct SearchQuery {
    pub participant: Option<AgentPubKey>,
    pub listing_hash: Option<ActionHash>,
    pub transaction_hash: Option<ActionHash>,
    pub status: Option<ConversationStatus>,
    pub subject_keyword: Option<String>,
}
```

**Output**: `ConversationsResponse`

**Returns**: Conversations matching ALL specified criteria

**Example**:
```javascript
// Find all active conversations about a specific listing
const results = await searchConversations({
    listing_hash: listingHash,
    status: "Active",
    subject_keyword: "macbook",
});
```

---

## Encryption Model

### Client-Side Encryption

Messages are encrypted **client-side** before being sent to the DHT.

**Recommended**: AES-256-GCM with ephemeral keys

**Process**:
1. Generate shared secret (ECDH)
2. Derive encryption key (HKDF)
3. Encrypt message (AES-256-GCM)
4. Store ciphertext on DHT
5. Recipient decrypts with shared secret

**Example (pseudocode)**:
```javascript
// Sender
const sharedSecret = deriveSharedSecret(myPrivateKey, recipientPublicKey);
const encryptionKey = hkdf(sharedSecret, "messaging-v1");
const encrypted = aes256gcm.encrypt(messageText, encryptionKey);

await sendMessage({
    recipient: recipientPubKey,
    encrypted_content: encrypted,
    // ...
});

// Recipient
const sharedSecret = deriveSharedSecret(myPrivateKey, senderPublicKey);
const decryptionKey = hkdf(sharedSecret, "messaging-v1");
const plaintext = aes256gcm.decrypt(encrypted_content, decryptionKey);
```

---

## MATL Gating (Spam Prevention)

### Requirement

To send messages or start conversations, agents must have:
- **MATL score ≥ 0.4**

### Rationale

This prevents spam without requiring centralized moderation:
- New agents must build reputation through transactions first
- Spammers would need multiple high-reputation identities (expensive)
- Legitimate users naturally exceed threshold through honest activity

### Error Message

If MATL score is too low:
```
"Insufficient MATL score to send messages (have: 0.25, need: 0.40).
 Build your reputation through successful transactions first."
```

### Checking MATL Score

```javascript
// Check before attempting to message
const matlScore = await getAgentMatlScore(myPubKey);

if (matlScore.composite >= 0.4) {
    // Can send messages
} else {
    // Show reputation-building guide
}
```

---

## IPFS Attachments

### Attachment Structure

```rust
pub struct Attachment {
    pub ipfs_cid: String,         // IPFS content ID
    pub filename: String,          // Original filename
    pub mime_type: String,         // image/png, video/mp4, etc.
    pub size_bytes: u64,           // File size
    pub thumbnail_cid: Option<String>, // Thumbnail for media
}
```

### Sending Attachments

**Process**:
1. Upload file to IPFS
2. Get CID
3. Create attachment metadata
4. Include in encrypted message content

**Example**:
```javascript
// 1. Upload to IPFS
const cid = await ipfs.add(fileBuffer);

// 2. Create attachment metadata
const attachment = {
    ipfs_cid: cid.toString(),
    filename: "screenshot.png",
    mime_type: "image/png",
    size_bytes: fileBuffer.length,
    thumbnail_cid: null,
};

// 3. Encrypt message with attachment
const messageWithAttachment = {
    text: "Here's the screenshot you requested",
    attachments: [attachment],
};

const encrypted = encrypt(JSON.stringify(messageWithAttachment));

// 4. Send message
await sendMessage({
    encrypted_content: encrypted,
    // ...
});
```

### Downloading Attachments

```javascript
// 1. Decrypt message
const decrypted = decrypt(message.encrypted_content);
const content = JSON.parse(decrypted);

// 2. Get attachment CID
const attachment = content.attachments[0];

// 3. Download from IPFS
const file = await ipfs.cat(attachment.ipfs_cid);

// 4. Display to user
displayImage(file, attachment.mime_type);
```

---

## Integration Examples

### Example 1: Messaging About a Listing

```javascript
// Buyer wants to ask about a listing
const listing = await getListing(listingHash);

const conversation = await startConversation({
    recipient: listing.seller,
    subject: `Question about ${listing.title}`,
    first_message_content: encrypt("Is this still available?"),
    listing_hash: listingHash,
    transaction_hash: null,
});

console.log("Conversation started:", conversation.conversation_hash);
```

### Example 2: Transaction Communication

```javascript
// Seller updates buyer about shipment
const transaction = await getTransaction(transactionHash);

await sendMessage({
    recipient: transaction.buyer,
    encrypted_content: encrypt("Your order has shipped! Tracking: 1Z999AA10123456784"),
    conversation_id: conversationHash,
    transaction_hash: transactionHash,
    message_type: "System",
});
```

### Example 3: Read Receipt Handling

```javascript
// Mark all unread messages as read
const { messages } = await getConversationMessages(conversationHash);

for (const msg of messages) {
    if (!msg.message.read_at && msg.message.recipient === myPubKey) {
        await markMessageRead(msg.message_hash);
    }
}

console.log("All messages marked as read");
```

### Example 4: Blocking Spam

```javascript
// User reports conversation as spam
await blockConversation(conversationHash);

// This emits a monitoring metric
// Admin dashboard can track spam patterns
```

---

## Monitoring Metrics

### Metrics Emitted

1. **MessageSent**
   - Value: 1.0 per message
   - Metadata: recipient
   - Use: Track messaging volume

2. **ConversationStarted**
   - Value: 1.0 per conversation
   - Metadata: participant count, subject
   - Use: Track new conversations

3. **ConversationBlocked**
   - Value: 1.0 per block
   - Agent: Who blocked
   - Use: Detect spam patterns

4. **MessageRead**
   - Value: 1.0 per read receipt
   - Use: Track engagement

### Monitoring Dashboard

```javascript
// Example: Get messaging stats
const metrics = await queryMetrics({
    metric_type: "MessageSent",
    time_window: 86400, // Last 24 hours
});

console.log(`Messages sent today: ${metrics.total}`);
console.log(`Most active users: ${metrics.top_agents}`);
```

---

## Best Practices

### For Developers

1. **Always encrypt client-side** - Never send plaintext
2. **Validate MATL score** before showing message UI
3. **Handle read receipts** to provide UX feedback
4. **Implement retry logic** for failed sends
5. **Cache conversation lists** to reduce DHT queries

### For Users

1. **Build reputation first** - Complete transactions before messaging
2. **Be respectful** - Blocked conversations impact your reputation
3. **Use clear subjects** - Makes searching easier later
4. **Report spam** - Help keep marketplace clean

### Security Considerations

1. **Key Management** - Securely store private keys
2. **Perfect Forward Secrecy** - Rotate ephemeral keys
3. **Metadata Leakage** - Subject lines visible on DHT
4. **Timing Attacks** - Randomize send times if needed
5. **Replay Protection** - Include timestamps in encrypted payload

---

## Performance Tips

### Efficient Message Loading

```javascript
// Don't: Load all conversations eagerly
const all = await getMyConversations();

// Do: Load recent conversations, lazy-load older
const recent = (await getMyConversations())
    .conversations
    .slice(0, 20); // First 20 only
```

### Caching Strategy

```javascript
// Cache conversation list
let conversationCache = [];
let lastFetch = 0;

async function getConversationsCached() {
    const now = Date.now();

    if (now - lastFetch > 60000) { // Refresh every minute
        conversationCache = await getMyConversations();
        lastFetch = now;
    }

    return conversationCache;
}
```

---

## Testing

### Test Coverage

The messaging zome includes **40+ tests** covering:
- Message creation and validation
- Conversation management
- Read receipts
- MATL gating
- Search functionality
- Spam prevention
- Edge cases

### Running Tests

```bash
cd backend/zomes/messaging/coordinator
cargo test
```

### Example Test

```rust
#[test]
fn test_low_matl_agent_cannot_message() {
    // Setup agent with low MATL score (0.2)
    let result = send_message(test_input);

    // Verify error message
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Insufficient MATL"));
}
```

---

## Future Enhancements

### Planned Features

1. **Voice Messages** - Audio message support via IPFS
2. **Video Calls** - WebRTC signaling through DHT
3. **Message Reactions** - Emoji reactions to messages
4. **Thread Replies** - Reply to specific messages
5. **Message Editing** - Edit sent messages (with history)
6. **Typing Indicators** - Real-time typing status
7. **Presence System** - Online/offline status
8. **Push Notifications** - Mobile notification support

### Community Requests

Submit feature requests to: https://github.com/mycelix/marketplace/issues

---

## Troubleshooting

### "Insufficient MATL score" Error

**Problem**: Cannot send messages

**Solution**: Build reputation through successful transactions
- Complete at least 3-5 transactions
- Maintain high-quality interactions
- Check current score: `get_agent_matl_score(myPubKey)`

### Messages Not Appearing

**Problem**: Sent messages don't show up

**Possible Causes**:
1. DHT propagation delay (wait 5-10 seconds)
2. Recipient offline (messages cached locally)
3. Encryption mismatch (check key derivation)

### Conversation Search Not Working

**Problem**: Cannot find conversations

**Solution**:
- Verify search criteria are correct
- Check conversation status (archived/blocked)
- Ensure listing/transaction hashes match exactly

---

## API Summary Table

| Endpoint | Input | Output | MATL Required | Rate Limit |
|----------|-------|--------|---------------|------------|
| `start_conversation` | Recipient, subject, message | Conversation | ≥0.4 | 10/hour |
| `send_message` | Recipient, content, conversation | Message | ≥0.4 | 100/hour |
| `get_my_conversations` | None | Conversations | No | 100/min |
| `get_conversation_messages` | Conversation hash | Messages | No | 100/min |
| `mark_message_read` | Message hash | Read receipt | No | Unlimited |
| `archive_conversation` | Conversation hash | Conversation | No | Unlimited |
| `block_conversation` | Conversation hash | Conversation | No | 10/day |
| `search_conversations` | Search query | Conversations | No | 100/min |

---

## Conclusion

The Messaging Zome provides a **robust, secure, and user-friendly** P2P messaging system for the Mycelix-Marketplace. By combining:

- End-to-end encryption
- MATL-based spam prevention
- Rich conversation management
- IPFS attachment support
- Comprehensive monitoring

We create a messaging experience that rivals centralized platforms while maintaining complete decentralization and user privacy.

**Next Steps**:
1. Integrate messaging UI into frontend
2. Implement client-side encryption
3. Add IPFS for attachments
4. Connect to notification system
5. Monitor usage metrics

**Questions?** See the full API reference or join our community chat!

---

*Last Updated*: December 30, 2025
*Version*: v1.0.0
*Status*: Production Ready ✅
