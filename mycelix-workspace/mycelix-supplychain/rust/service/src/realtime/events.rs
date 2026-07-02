// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-Time Event Types
//!
//! Defines all events that can be sent/received via WebSocket

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Client-to-server messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ClientMessage {
    /// Subscribe to a channel
    Subscribe(SubscribeRequest),

    /// Unsubscribe from a channel
    Unsubscribe(UnsubscribeRequest),

    /// Join a document for collaborative editing
    JoinDocument(JoinDocumentRequest),

    /// Leave a document
    LeaveDocument(LeaveDocumentRequest),

    /// Broadcast cursor/selection position
    CursorMove(CursorUpdate),

    /// Document field update
    FieldUpdate(FieldUpdateRequest),

    /// Ping for keepalive
    Ping,
}

/// Server-to-client messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ServerMessage {
    /// Connection established
    Connected(ConnectionInfo),

    /// Subscription confirmed
    Subscribed(SubscriptionInfo),

    /// Unsubscription confirmed
    Unsubscribed { channel: String },

    /// Document join confirmed
    DocumentJoined(DocumentInfo),

    /// Document left
    DocumentLeft { document_id: Uuid },

    /// Another user joined the document
    UserJoined(UserPresence),

    /// Another user left the document
    UserLeft { user_id: Uuid, document_id: Uuid },

    /// User cursor/selection update
    UserCursor(CursorUpdate),

    /// Field was updated by another user
    FieldUpdated(FieldUpdate),

    /// Document was saved
    DocumentSaved { document_id: Uuid, version: i32 },

    /// Generic business event
    BusinessEvent(BusinessEvent),

    /// Error message
    Error(ErrorMessage),

    /// Pong response
    Pong,
}

// ============================================================================
// Request Types
// ============================================================================

/// Subscribe request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub channel: String,
}

/// Unsubscribe request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeRequest {
    pub channel: String,
}

/// Join document request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinDocumentRequest {
    pub document_type: DocumentType,
    pub document_id: Uuid,
}

/// Leave document request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveDocumentRequest {
    pub document_id: Uuid,
}

/// Field update request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldUpdateRequest {
    pub document_id: Uuid,
    pub field_path: String,
    pub value: serde_json::Value,
    pub version: i32,
}

// ============================================================================
// Response Types
// ============================================================================

/// Connection info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub session_id: Uuid,
    pub user_id: Uuid,
    pub tenant_id: Uuid,
    pub connected_at: DateTime<Utc>,
}

/// Subscription info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionInfo {
    pub channel: String,
    pub subscribed_at: DateTime<Utc>,
}

/// Document info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub document_type: DocumentType,
    pub document_id: Uuid,
    pub version: i32,
    pub current_users: Vec<UserPresence>,
}

/// User presence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    pub user_id: Uuid,
    pub user_name: String,
    pub color: String,
    pub joined_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
}

/// Cursor update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorUpdate {
    pub user_id: Uuid,
    pub document_id: Uuid,
    pub field_path: Option<String>,
    pub selection_start: Option<u32>,
    pub selection_end: Option<u32>,
}

/// Field update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldUpdate {
    pub document_id: Uuid,
    pub field_path: String,
    pub value: serde_json::Value,
    pub updated_by: Uuid,
    pub updated_at: DateTime<Utc>,
    pub version: i32,
}

/// Error message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

// ============================================================================
// Business Events
// ============================================================================

/// Business event wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessEvent {
    pub event_type: BusinessEventType,
    pub entity_type: String,
    pub entity_id: Uuid,
    pub data: serde_json::Value,
    pub actor_id: Uuid,
    pub timestamp: DateTime<Utc>,
}

/// Business event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BusinessEventType {
    // Invoice events
    InvoiceCreated,
    InvoiceUpdated,
    InvoiceSent,
    InvoicePaid,
    InvoiceOverdue,

    // Bill events
    BillCreated,
    BillUpdated,
    BillApproved,
    BillRejected,
    BillPaid,

    // Payment events
    PaymentReceived,
    PaymentSent,
    PaymentFailed,

    // Inventory events
    StockLow,
    StockReplenished,
    ItemReceived,
    ItemShipped,

    // Purchase events
    PurchaseOrderCreated,
    PurchaseOrderApproved,
    PurchaseOrderReceived,

    // Sales events
    SalesOrderCreated,
    SalesOrderConfirmed,
    SalesOrderShipped,

    // Bank events
    BankTransactionImported,
    TransactionMatched,

    // General
    CommentAdded,
    DocumentUploaded,
    ApprovalRequired,
}

/// Document types for collaboration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DocumentType {
    Invoice,
    Bill,
    PurchaseOrder,
    SalesOrder,
    Quote,
    JournalEntry,
    Contact,
    Item,
}

impl DocumentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentType::Invoice => "INVOICE",
            DocumentType::Bill => "BILL",
            DocumentType::PurchaseOrder => "PURCHASE_ORDER",
            DocumentType::SalesOrder => "SALES_ORDER",
            DocumentType::Quote => "QUOTE",
            DocumentType::JournalEntry => "JOURNAL_ENTRY",
            DocumentType::Contact => "CONTACT",
            DocumentType::Item => "ITEM",
        }
    }
}

// ============================================================================
// Channel Definitions
// ============================================================================

/// Channel types for pub/sub
#[derive(Debug, Clone)]
pub enum Channel {
    /// Tenant-wide notifications
    TenantNotifications(Uuid),

    /// User-specific notifications
    UserNotifications(Uuid),

    /// Dashboard updates
    Dashboard(Uuid),

    /// Module-specific updates (e.g., finance, inventory)
    Module { tenant_id: Uuid, module: String },

    /// Document-specific updates
    Document { document_type: DocumentType, document_id: Uuid },
}

impl Channel {
    pub fn to_string(&self) -> String {
        match self {
            Channel::TenantNotifications(id) => format!("tenant:{}:notifications", id),
            Channel::UserNotifications(id) => format!("user:{}:notifications", id),
            Channel::Dashboard(id) => format!("tenant:{}:dashboard", id),
            Channel::Module { tenant_id, module } => format!("tenant:{}:module:{}", tenant_id, module),
            Channel::Document { document_type, document_id } => {
                format!("doc:{}:{}", document_type.as_str(), document_id)
            }
        }
    }

    pub fn from_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(':').collect();

        match parts.as_slice() {
            ["tenant", id, "notifications"] => {
                Uuid::parse_str(id).ok().map(Channel::TenantNotifications)
            }
            ["user", id, "notifications"] => {
                Uuid::parse_str(id).ok().map(Channel::UserNotifications)
            }
            ["tenant", id, "dashboard"] => Uuid::parse_str(id).ok().map(Channel::Dashboard),
            ["tenant", id, "module", module] => Uuid::parse_str(id).ok().map(|tenant_id| Channel::Module {
                tenant_id,
                module: module.to_string(),
            }),
            ["doc", doc_type, id] => {
                let document_type = match *doc_type {
                    "INVOICE" => Some(DocumentType::Invoice),
                    "BILL" => Some(DocumentType::Bill),
                    "PURCHASE_ORDER" => Some(DocumentType::PurchaseOrder),
                    "SALES_ORDER" => Some(DocumentType::SalesOrder),
                    "QUOTE" => Some(DocumentType::Quote),
                    "JOURNAL_ENTRY" => Some(DocumentType::JournalEntry),
                    "CONTACT" => Some(DocumentType::Contact),
                    "ITEM" => Some(DocumentType::Item),
                    _ => None,
                }?;
                let document_id = Uuid::parse_str(id).ok()?;
                Some(Channel::Document { document_type, document_id })
            }
            _ => None,
        }
    }
}
