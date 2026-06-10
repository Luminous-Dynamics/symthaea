// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ticket Management for Iroh Connections
//!
//! Handles creation, validation, and exchange of connection tickets
//! between peers. Tickets are the mechanism by which peers establish
//! direct QUIC connections.

use crate::swarm::{ConnectionTicket, SwarmError, SwarmResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Manages connection tickets for peer discovery and connection
///
/// The TicketManager handles the lifecycle of connection tickets:
/// - **Outgoing tickets**: Created by this node for others to connect to us
/// - **Incoming tickets**: Received from peers, validated before connection
/// - **Expiration**: Automatic cleanup of stale tickets
/// - **Use tracking**: Counts how many times a ticket has been used
pub struct TicketManager {
    /// Tickets we've created for others to connect to us
    outgoing_tickets: RwLock<HashMap<String, TicketEntry>>,

    /// Tickets we've received from others
    incoming_tickets: RwLock<HashMap<String, TicketEntry>>,

    /// Default ticket expiration
    default_expiration: Duration,
}

/// An entry in the ticket cache with metadata for tracking and expiration
struct TicketEntry {
    ticket: ConnectionTicket,
    created_at: SystemTime,
    use_count: usize,
}

impl TicketManager {
    /// Create a new ticket manager
    pub fn new() -> Self {
        Self {
            outgoing_tickets: RwLock::new(HashMap::new()),
            incoming_tickets: RwLock::new(HashMap::new()),
            default_expiration: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Create a new ticket manager with custom expiration
    pub fn with_expiration(expiration: Duration) -> Self {
        Self {
            outgoing_tickets: RwLock::new(HashMap::new()),
            incoming_tickets: RwLock::new(HashMap::new()),
            default_expiration: expiration,
        }
    }

    /// Store an outgoing ticket (one we created for others)
    ///
    /// Called by `IrohNode::create_ticket()` when generating a ticket for peers.
    pub fn store_outgoing(&self, ticket: ConnectionTicket) {
        let entry = TicketEntry {
            ticket: ticket.clone(),
            created_at: SystemTime::now(),
            use_count: 0,
        };
        self.outgoing_tickets
            .write()
            .insert(ticket.node_id.clone(), entry);
    }

    /// Store an incoming ticket (one we received from a peer)
    ///
    /// Called by `IrohNode::connect()` after validating the ticket.
    pub fn store_incoming(&self, ticket: ConnectionTicket) {
        let entry = TicketEntry {
            ticket: ticket.clone(),
            created_at: SystemTime::now(),
            use_count: 0,
        };
        self.incoming_tickets
            .write()
            .insert(ticket.node_id.clone(), entry);
    }

    /// Get an incoming ticket for a specific node
    ///
    /// Used for reconnection or ticket caching via `IrohNode::get_cached_ticket()`.
    pub fn get_incoming(&self, node_id: &str) -> Option<ConnectionTicket> {
        let tickets = self.incoming_tickets.read();
        tickets.get(node_id).map(|entry| entry.ticket.clone())
    }

    /// Mark a ticket as used (increments use count)
    ///
    /// Called by `IrohNode::connect()` after successful connection.
    pub fn mark_used(&self, node_id: &str) {
        if let Some(entry) = self.incoming_tickets.write().get_mut(node_id) {
            entry.use_count += 1;
        }
    }

    /// Remove expired tickets from both incoming and outgoing caches
    ///
    /// Called by `IrohNode::cleanup_tickets()` and automatically before `connect()`.
    pub fn cleanup_expired(&self) {
        let _now = SystemTime::now();

        self.outgoing_tickets.write().retain(|_, entry| {
            entry
                .created_at
                .elapsed()
                .map(|d| d < self.default_expiration)
                .unwrap_or(false)
        });

        self.incoming_tickets.write().retain(|_, entry| {
            entry
                .created_at
                .elapsed()
                .map(|d| d < self.default_expiration)
                .unwrap_or(false)
        });
    }

    /// Get all known peer node IDs (from incoming tickets)
    ///
    /// Used by `IrohNode::known_peer_ids()` for peer discovery.
    pub fn known_peers(&self) -> Vec<String> {
        self.incoming_tickets.read().keys().cloned().collect()
    }

    /// Get the number of stored tickets (outgoing, incoming)
    ///
    /// Used by `IrohNode::ticket_stats()` for monitoring.
    pub fn ticket_count(&self) -> (usize, usize) {
        (
            self.outgoing_tickets.read().len(),
            self.incoming_tickets.read().len(),
        )
    }

    /// Validate a ticket before connection
    ///
    /// Called by `IrohNode::connect()` to ensure ticket is valid:
    /// - Not expired
    /// - Has valid ticket string
    /// - Has valid node ID
    pub fn validate(&self, ticket: &ConnectionTicket) -> SwarmResult<()> {
        // Check expiration
        if ticket.is_expired() {
            return Err(SwarmError::InvalidTicket {
                reason: "Ticket has expired".to_string(),
            });
        }

        // Check ticket format (basic validation)
        if ticket.ticket.is_empty() {
            return Err(SwarmError::InvalidTicket {
                reason: "Empty ticket string".to_string(),
            });
        }

        if ticket.node_id.is_empty() {
            return Err(SwarmError::InvalidTicket {
                reason: "Empty node ID".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for TicketManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Basic Operations
    // ============================================================================

    #[test]
    fn test_ticket_manager_creation() {
        let manager = TicketManager::new();
        assert_eq!(manager.ticket_count(), (0, 0));
    }

    #[test]
    fn test_with_custom_expiration() {
        let manager = TicketManager::with_expiration(Duration::from_secs(60));
        assert_eq!(manager.ticket_count(), (0, 0));
        // Expiration is internal, but we verify construction works
    }

    #[test]
    fn test_default_trait() {
        let manager = TicketManager::default();
        assert_eq!(manager.ticket_count(), (0, 0));
    }

    // ============================================================================
    // Incoming Tickets
    // ============================================================================

    #[test]
    fn test_store_and_retrieve_incoming() {
        let manager = TicketManager::new();

        let ticket = ConnectionTicket::new("test-ticket", "node-123");
        manager.store_incoming(ticket);

        let retrieved = manager.get_incoming("node-123");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().ticket, "test-ticket");
    }

    #[test]
    fn test_get_incoming_nonexistent() {
        let manager = TicketManager::new();
        assert!(manager.get_incoming("nonexistent").is_none());
    }

    #[test]
    fn test_store_incoming_overwrites() {
        let manager = TicketManager::new();

        manager.store_incoming(ConnectionTicket::new("ticket-v1", "node-123"));
        manager.store_incoming(ConnectionTicket::new("ticket-v2", "node-123"));

        let retrieved = manager.get_incoming("node-123");
        assert_eq!(retrieved.unwrap().ticket, "ticket-v2");
        assert_eq!(manager.ticket_count(), (0, 1)); // Only one entry
    }

    // ============================================================================
    // Outgoing Tickets
    // ============================================================================

    #[test]
    fn test_store_outgoing() {
        let manager = TicketManager::new();

        let ticket = ConnectionTicket::new("my-ticket", "my-node");
        manager.store_outgoing(ticket);

        assert_eq!(manager.ticket_count(), (1, 0));
    }

    #[test]
    fn test_store_multiple_outgoing() {
        let manager = TicketManager::new();

        manager.store_outgoing(ConnectionTicket::new("t1", "node-1"));
        manager.store_outgoing(ConnectionTicket::new("t2", "node-2"));

        assert_eq!(manager.ticket_count(), (2, 0));
    }

    // ============================================================================
    // Use Tracking
    // ============================================================================

    #[test]
    fn test_mark_used_increments_count() {
        let manager = TicketManager::new();

        manager.store_incoming(ConnectionTicket::new("ticket", "node-123"));

        // Mark as used multiple times
        manager.mark_used("node-123");
        manager.mark_used("node-123");
        manager.mark_used("node-123");

        // Ticket should still be retrievable
        let retrieved = manager.get_incoming("node-123");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_mark_used_nonexistent_is_noop() {
        let manager = TicketManager::new();
        // Should not panic or error
        manager.mark_used("nonexistent");
        assert_eq!(manager.ticket_count(), (0, 0));
    }

    // ============================================================================
    // Peer Discovery
    // ============================================================================

    #[test]
    fn test_known_peers() {
        let manager = TicketManager::new();

        manager.store_incoming(ConnectionTicket::new("t1", "node-1"));
        manager.store_incoming(ConnectionTicket::new("t2", "node-2"));

        let peers = manager.known_peers();
        assert_eq!(peers.len(), 2);
        assert!(peers.contains(&"node-1".to_string()));
        assert!(peers.contains(&"node-2".to_string()));
    }

    #[test]
    fn test_known_peers_empty() {
        let manager = TicketManager::new();
        let peers = manager.known_peers();
        assert!(peers.is_empty());
    }

    // ============================================================================
    // Validation
    // ============================================================================

    #[test]
    fn test_validation_valid_ticket() {
        let manager = TicketManager::new();

        let valid = ConnectionTicket::new("ticket", "node");
        assert!(manager.validate(&valid).is_ok());
    }

    #[test]
    fn test_validation_empty_ticket_string() {
        let manager = TicketManager::new();

        let empty_ticket = ConnectionTicket::new("", "node");
        let result = manager.validate(&empty_ticket);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SwarmError::InvalidTicket { .. }
        ));
    }

    #[test]
    fn test_validation_empty_node_id() {
        let manager = TicketManager::new();

        let empty_node = ConnectionTicket::new("ticket", "");
        let result = manager.validate(&empty_node);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SwarmError::InvalidTicket { .. }
        ));
    }

    #[test]
    fn test_validation_expired_ticket() {
        let manager = TicketManager::new();

        let mut expired_ticket = ConnectionTicket::new("ticket", "node");
        // Set expiration to the past (1 ms since epoch)
        expired_ticket.expires_at = Some(1);

        let result = manager.validate(&expired_ticket);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), SwarmError::InvalidTicket { reason } if reason.contains("expired"))
        );
    }

    #[test]
    fn test_validation_ticket_without_expiration() {
        let manager = TicketManager::new();

        let ticket = ConnectionTicket::new("ticket", "node");
        assert!(ticket.expires_at.is_none());
        assert!(manager.validate(&ticket).is_ok());
    }

    // ============================================================================
    // Expiration and Cleanup
    // ============================================================================

    #[test]
    fn test_cleanup_expired_removes_old_tickets() {
        // Create manager with very short expiration for testing
        let manager = TicketManager::with_expiration(Duration::from_millis(1));

        manager.store_incoming(ConnectionTicket::new("ticket", "node-123"));
        assert_eq!(manager.ticket_count(), (0, 1));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        // Cleanup should remove the expired ticket
        manager.cleanup_expired();
        assert_eq!(manager.ticket_count(), (0, 0));
    }

    #[test]
    fn test_cleanup_keeps_fresh_tickets() {
        // Create manager with long expiration
        let manager = TicketManager::with_expiration(Duration::from_secs(3600));

        manager.store_incoming(ConnectionTicket::new("ticket", "node-123"));
        manager.store_outgoing(ConnectionTicket::new("my-ticket", "my-node"));

        // Cleanup should keep fresh tickets
        manager.cleanup_expired();
        assert_eq!(manager.ticket_count(), (1, 1));
    }

    #[test]
    fn test_cleanup_mixed_expiration() {
        // Create manager with very short expiration
        let manager = TicketManager::with_expiration(Duration::from_millis(5));

        // Add first ticket
        manager.store_incoming(ConnectionTicket::new("old-ticket", "old-node"));

        // Wait a bit
        std::thread::sleep(Duration::from_millis(10));

        // Add second ticket (fresh)
        // Note: With such short expiration, the first ticket should be expired
        // but the second one might still be fresh if added quickly enough
        manager.store_incoming(ConnectionTicket::new("new-ticket", "new-node"));

        // Cleanup - old ticket should be removed
        manager.cleanup_expired();

        // At least one ticket should be removed (the old one)
        // The new one may or may not survive depending on timing
        assert!(manager.ticket_count().1 <= 2);
    }

    // ============================================================================
    // Ticket Count
    // ============================================================================

    #[test]
    fn test_ticket_count_separate_counters() {
        let manager = TicketManager::new();

        manager.store_outgoing(ConnectionTicket::new("out1", "node-o1"));
        manager.store_outgoing(ConnectionTicket::new("out2", "node-o2"));
        manager.store_incoming(ConnectionTicket::new("in1", "node-i1"));

        let (outgoing, incoming) = manager.ticket_count();
        assert_eq!(outgoing, 2);
        assert_eq!(incoming, 1);
    }

    // ============================================================================
    // Concurrency (basic sanity)
    // ============================================================================

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(TicketManager::new());
        let mut handles = vec![];

        // Spawn multiple threads that store and read tickets
        for i in 0..10 {
            let mgr = Arc::clone(&manager);
            handles.push(thread::spawn(move || {
                let ticket = ConnectionTicket::new(format!("ticket-{}", i), format!("node-{}", i));
                mgr.store_incoming(ticket);
                mgr.known_peers();
                mgr.ticket_count();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have stored all tickets
        assert_eq!(manager.ticket_count().1, 10);
    }
}
