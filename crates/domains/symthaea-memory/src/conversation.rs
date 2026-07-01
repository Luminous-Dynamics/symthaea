// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conversation Memory - SQLite-based Persistent Conversation Storage
//!
//! This module provides persistent storage for conversations, enabling:
//! - Session resumption across restarts
//! - Semantic search for similar past conversations
//! - Causal learning from action→outcome pairs
//! - Φ tracking across conversation turns
//!
//! ## Design Philosophy
//!
//! Unlike the complex multi-database architecture in `databases/`, this module
//! uses simple SQLite for conversation persistence. This is intentional:
//! - Conversations are structured, relational data (not vectors)
//! - SQLite is battle-tested, zero-config, and embedded
//! - HDC embeddings are stored as BLOBs for similarity search
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::memory::ConversationMemory;
//!
//! let mut memory = ConversationMemory::new("symthaea_memory.db")?;
//!
//! // Start a new session
//! let session_id = memory.start_session();
//!
//! // Add turns as conversation progresses
//! memory.add_turn("user", "How do I install vim?", 0.45, &embedding)?;
//! memory.add_turn("assistant", "Use: nix-env -iA nixpkgs.vim", 0.52, &embedding)?;
//!
//! // Resume later
//! let turns = memory.resume_session(&session_id)?;
//! ```

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use symthaea_core::hdc::ContinuousHV;
use uuid::Uuid;

/// A single turn in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Turn number within conversation
    pub turn_number: usize,
    /// Role: "user" or "assistant"
    pub role: String,
    /// Content of the turn
    pub content: String,
    /// Φ (integrated information) at this turn
    pub phi: f32,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

/// Summary of a conversation for listing/search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    /// Unique conversation ID
    pub id: String,
    /// When created
    pub created_at: DateTime<Utc>,
    /// When last updated
    pub updated_at: DateTime<Utc>,
    /// Number of turns
    pub turn_count: usize,
    /// Average Φ during conversation
    pub phi_average: Option<f32>,
    /// Topic summary (LLM-generated or extracted)
    pub topic_summary: Option<String>,
}

/// A causal learning record (action → outcome)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLearning {
    /// Action that was taken
    pub action: String,
    /// Outcome of the action
    pub outcome: String,
    /// Φ before action
    pub phi_before: f32,
    /// Φ after action
    pub phi_after: f32,
    /// Extracted pattern/learning
    pub pattern: String,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

/// SQLite-based conversation memory
pub struct ConversationMemory {
    /// Database connection
    conn: Connection,
    /// Current active conversation ID
    current_conversation_id: Option<String>,
    /// Current turn count
    turn_count: usize,
}

impl ConversationMemory {
    /// Create or open a conversation memory database
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        // Initialize schema
        conn.execute_batch(
            r#"
            -- Main conversations table
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                phi_average REAL,
                topic_summary TEXT,
                hypervector BLOB,
                metadata TEXT
            );

            -- Individual turns within conversations
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                phi_at_turn REAL,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            -- Causal learning records (action → outcome)
            CREATE TABLE IF NOT EXISTS causal_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                action_taken TEXT,
                outcome TEXT,
                phi_before REAL,
                phi_after REAL,
                learned_pattern TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at);
            CREATE INDEX IF NOT EXISTS idx_turns_conversation
                ON conversation_turns(conversation_id, turn_number);
            CREATE INDEX IF NOT EXISTS idx_causal_conversation
                ON causal_chains(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_turns_role
                ON conversation_turns(role);
            "#,
        )?;

        Ok(Self {
            conn,
            current_conversation_id: None,
            turn_count: 0,
        })
    }

    /// Start a new conversation session
    ///
    /// Returns the new conversation ID, or an error if the session could not be created
    pub fn start_session(&mut self) -> Result<String> {
        let id = Uuid::new_v4().to_string();

        self.conn
            .execute("INSERT INTO conversations (id) VALUES (?1)", params![&id])
            .map_err(|e| anyhow!("Failed to create conversation: {e}"))?;

        self.current_conversation_id = Some(id.clone());
        self.turn_count = 0;

        tracing::info!(conversation_id = %id, "Started new conversation session");

        Ok(id)
    }

    /// Resume an existing conversation session
    ///
    /// Returns all previous turns in the conversation
    pub fn resume_session(&mut self, conversation_id: &str) -> Result<Vec<ConversationTurn>> {
        // Verify conversation exists
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM conversations WHERE id = ?1",
                params![conversation_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);

        if !exists {
            return Err(anyhow!("Conversation not found: {conversation_id}"));
        }

        self.current_conversation_id = Some(conversation_id.to_string());

        // Load all turns
        let mut stmt = self.conn.prepare(
            r#"
            SELECT turn_number, role, content, phi_at_turn, created_at
            FROM conversation_turns
            WHERE conversation_id = ?1
            ORDER BY turn_number
            "#,
        )?;

        let turns: Vec<ConversationTurn> = stmt
            .query_map(params![conversation_id], |row| {
                Ok(ConversationTurn {
                    turn_number: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    phi: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0) as f32,
                    created_at: row
                        .get::<_, String>(4)
                        .ok()
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        self.turn_count = turns.len();

        tracing::info!(
            conversation_id = %conversation_id,
            turns = %self.turn_count,
            "Resumed conversation session"
        );

        Ok(turns)
    }

    /// Add a turn to the current conversation
    pub fn add_turn(
        &mut self,
        role: &str,
        content: &str,
        phi: f32,
        embedding: Option<&ContinuousHV>,
    ) -> Result<()> {
        let conv_id = self
            .current_conversation_id
            .as_ref()
            .ok_or_else(|| anyhow!("No active conversation - call start_session() first"))?;

        self.turn_count += 1;

        // Serialize embedding if provided
        let embedding_blob: Option<Vec<u8>> = embedding.and_then(|e| {
            bincode::serialize(&e.values)
                .map_err(|err| {
                    tracing::warn!(error = %err, "Failed to serialize conversation embedding — storing without vector");
                    err
                })
                .ok()
        });

        self.conn.execute(
            r#"
            INSERT INTO conversation_turns
                (conversation_id, turn_number, role, content, phi_at_turn, embedding)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                conv_id,
                self.turn_count,
                role,
                content,
                phi as f64,
                embedding_blob,
            ],
        )?;

        // Update conversation timestamp and compute running average Φ
        self.conn.execute(
            r#"
            UPDATE conversations
            SET updated_at = datetime('now'),
                phi_average = (
                    SELECT AVG(phi_at_turn)
                    FROM conversation_turns
                    WHERE conversation_id = ?1
                )
            WHERE id = ?1
            "#,
            params![conv_id],
        )?;

        tracing::debug!(
            conversation_id = %conv_id,
            turn = %self.turn_count,
            role = %role,
            phi = %phi,
            "Added conversation turn"
        );

        Ok(())
    }

    /// Get the current conversation ID
    pub fn current_conversation(&self) -> Option<&str> {
        self.current_conversation_id.as_deref()
    }

    /// Get the current turn count
    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    /// Find similar past conversations using HDC embedding similarity
    ///
    /// Compares the query embedding against stored conversation embeddings
    pub fn find_similar(
        &self,
        query_embedding: &ContinuousHV,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, hypervector FROM conversations WHERE hypervector IS NOT NULL")?;

        let mut similarities: Vec<(String, f32)> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, blob)| {
                let values: Vec<f32> = bincode::deserialize(&blob)
                    .map_err(|e| {
                        tracing::warn!(conversation_id = %id, error = %e, "Corrupted embedding BLOB in conversation — skipping");
                        e
                    })
                    .ok()?;
                let hv = ContinuousHV { values };
                let sim = query_embedding.similarity(&hv);
                Some((id, sim))
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// Record a causal learning (action → outcome)
    ///
    /// This captures what actions lead to what outcomes, allowing the system
    /// to learn from experience. Φ changes indicate integration impact.
    pub fn record_causal_learning(
        &self,
        action: &str,
        outcome: &str,
        phi_before: f32,
        phi_after: f32,
    ) -> Result<()> {
        let conv_id = self
            .current_conversation_id
            .as_ref()
            .ok_or_else(|| anyhow!("No active conversation"))?;

        // Extract pattern from Φ change
        let pattern = if phi_after > phi_before + 0.05 {
            format!(
                "POSITIVE: '{}' improved integration (+{:.2}Φ)",
                action,
                phi_after - phi_before
            )
        } else if phi_after < phi_before - 0.05 {
            format!(
                "NEGATIVE: '{}' reduced integration ({:.2}Φ)",
                action,
                phi_after - phi_before
            )
        } else {
            format!("NEUTRAL: '{action}' had minimal Φ impact")
        };

        self.conn.execute(
            r#"
            INSERT INTO causal_chains
                (conversation_id, action_taken, outcome, phi_before, phi_after, learned_pattern)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                conv_id,
                action,
                outcome,
                phi_before as f64,
                phi_after as f64,
                &pattern,
            ],
        )?;

        tracing::info!(
            action = %action,
            phi_before = %phi_before,
            phi_after = %phi_after,
            pattern = %pattern,
            "Recorded causal learning"
        );

        Ok(())
    }

    /// Get recent causal learnings for analysis
    pub fn get_recent_learnings(&self, limit: usize) -> Result<Vec<CausalLearning>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT action_taken, outcome, phi_before, phi_after, learned_pattern, created_at
            FROM causal_chains
            ORDER BY created_at DESC
            LIMIT ?1
            "#,
        )?;

        let learnings: Vec<CausalLearning> = stmt
            .query_map(params![limit as i64], |row| {
                Ok(CausalLearning {
                    action: row.get(0)?,
                    outcome: row.get(1)?,
                    phi_before: row.get::<_, f64>(2)? as f32,
                    phi_after: row.get::<_, f64>(3)? as f32,
                    pattern: row.get(4)?,
                    created_at: row
                        .get::<_, String>(5)
                        .ok()
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(learnings)
    }

    /// List recent conversations
    pub fn list_conversations(&self, limit: usize) -> Result<Vec<ConversationSummary>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT
                c.id,
                c.created_at,
                c.updated_at,
                c.phi_average,
                c.topic_summary,
                COUNT(t.id) as turn_count
            FROM conversations c
            LEFT JOIN conversation_turns t ON c.id = t.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?1
            "#,
        )?;

        let conversations: Vec<ConversationSummary> = stmt
            .query_map(params![limit as i64], |row| {
                Ok(ConversationSummary {
                    id: row.get(0)?,
                    created_at: row
                        .get::<_, String>(1)
                        .ok()
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                    updated_at: row
                        .get::<_, String>(2)
                        .ok()
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                    phi_average: row.get::<_, Option<f64>>(3)?.map(|v| v as f32),
                    topic_summary: row.get(4)?,
                    turn_count: row.get::<_, i64>(5)? as usize,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(conversations)
    }

    /// Update the topic summary for a conversation (typically from LLM)
    pub fn set_topic_summary(&self, conversation_id: &str, summary: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations SET topic_summary = ?1 WHERE id = ?2",
            params![summary, conversation_id],
        )?;
        Ok(())
    }

    /// Set the conversation-level embedding for similarity search
    pub fn set_conversation_embedding(
        &self,
        conversation_id: &str,
        embedding: &ContinuousHV,
    ) -> Result<()> {
        let blob = bincode::serialize(&embedding.values)?;
        self.conn.execute(
            "UPDATE conversations SET hypervector = ?1 WHERE id = ?2",
            params![blob, conversation_id],
        )?;
        Ok(())
    }

    /// Get database statistics
    pub fn stats(&self) -> Result<ConversationMemoryStats> {
        let conversation_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM conversations", [], |row| row.get(0))?;

        let turn_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM conversation_turns", [], |row| {
                    row.get(0)
                })?;

        let causal_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM causal_chains", [], |row| row.get(0))?;

        let avg_phi: Option<f64> = self
            .conn
            .query_row(
                "SELECT AVG(phi_at_turn) FROM conversation_turns WHERE phi_at_turn IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .ok();

        Ok(ConversationMemoryStats {
            conversation_count: conversation_count as usize,
            turn_count: turn_count as usize,
            causal_learning_count: causal_count as usize,
            average_phi: avg_phi.map(|v| v as f32),
        })
    }
}

/// Statistics about the conversation memory
#[derive(Debug, Clone)]
pub struct ConversationMemoryStats {
    pub conversation_count: usize,
    pub turn_count: usize,
    pub causal_learning_count: usize,
    pub average_phi: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::HDC_DIMENSION;
    use tempfile::NamedTempFile;

    #[test]
    fn test_conversation_lifecycle() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file for test");
        let mut memory = ConversationMemory::new(temp_file.path()).unwrap();

        // Start session
        let session_id = memory.start_session().unwrap();
        assert!(!session_id.is_empty());
        assert_eq!(memory.turn_count(), 0);

        // Add turns
        memory
            .add_turn("user", "Hello, how are you?", 0.45, None)
            .unwrap();
        assert_eq!(memory.turn_count(), 1);

        memory
            .add_turn("assistant", "I'm doing well, thank you!", 0.52, None)
            .unwrap();
        assert_eq!(memory.turn_count(), 2);

        // Verify current conversation
        assert_eq!(memory.current_conversation(), Some(session_id.as_str()));
    }

    #[test]
    fn test_session_resume() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file for test");
        let session_id;

        // First session
        {
            let mut memory = ConversationMemory::new(temp_file.path()).unwrap();
            session_id = memory.start_session().unwrap();
            memory.add_turn("user", "Test message", 0.5, None).unwrap();
        }

        // Resume session
        {
            let mut memory = ConversationMemory::new(temp_file.path()).unwrap();
            let turns = memory.resume_session(&session_id).unwrap();

            assert_eq!(turns.len(), 1);
            assert_eq!(turns[0].role, "user");
            assert_eq!(turns[0].content, "Test message");
            assert_eq!(memory.turn_count(), 1);
        }
    }

    #[test]
    fn test_causal_learning() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file for test");
        let mut memory = ConversationMemory::new(temp_file.path()).unwrap();

        memory.start_session().unwrap();

        // Record positive learning
        memory
            .record_causal_learning("nix-env -i vim", "Success: vim installed", 0.45, 0.55)
            .unwrap();

        let learnings = memory.get_recent_learnings(10).unwrap();
        assert_eq!(learnings.len(), 1);
        assert!(learnings[0].pattern.contains("POSITIVE"));
    }

    #[test]
    fn test_similarity_search() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file for test");
        let mut memory = ConversationMemory::new(temp_file.path()).unwrap();

        let session_id = memory.start_session().unwrap();

        // Set conversation embedding
        let embedding = ContinuousHV::random(HDC_DIMENSION, 42);
        memory
            .set_conversation_embedding(&session_id, &embedding)
            .unwrap();

        // Search for similar
        let similar = memory.find_similar(&embedding, 5).unwrap();
        assert_eq!(similar.len(), 1);
        assert_eq!(similar[0].0, session_id);
        assert!((similar[0].1 - 1.0).abs() < 0.01); // Self-similarity should be ~1.0
    }

    #[test]
    fn test_stats() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file for test");
        let mut memory = ConversationMemory::new(temp_file.path()).unwrap();

        memory.start_session().unwrap();
        memory.add_turn("user", "Test 1", 0.4, None).unwrap();
        memory
            .add_turn("assistant", "Response 1", 0.5, None)
            .unwrap();

        let stats = memory.stats().unwrap();
        assert_eq!(stats.conversation_count, 1);
        assert_eq!(stats.turn_count, 2);
        assert!(stats.average_phi.is_some());
    }
}
