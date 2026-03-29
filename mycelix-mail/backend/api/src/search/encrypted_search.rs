// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Encrypted Search
//!
//! Search over encrypted data using blind indexing and secure search tokens

use argon2::{Argon2, PasswordHasher};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashSet;
use uuid::Uuid;

/// Encrypted search index entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindIndex {
    /// Blinded search token
    pub token: Vec<u8>,
    /// Document ID
    pub document_id: Uuid,
    /// Field name
    pub field: String,
    /// Word position (for phrase searches)
    pub position: u32,
}

/// Encrypted search service
pub struct EncryptedSearchService {
    pool: PgPool,
    /// Master key for blind indexing (derived from user's key)
    blind_index_key: [u8; 32],
}

impl EncryptedSearchService {
    pub fn new(pool: PgPool, user_key: &[u8]) -> Self {
        // Derive blind index key from user's key
        let mut hasher = blake3::Hasher::new_keyed(&[0u8; 32]);
        hasher.update(b"blind_index_key");
        hasher.update(user_key);
        let blind_index_key: [u8; 32] = hasher.finalize().into();

        Self {
            pool,
            blind_index_key,
        }
    }

    /// Index document for encrypted search
    pub async fn index_document(
        &self,
        document_id: Uuid,
        fields: Vec<(&str, &str)>,
    ) -> Result<(), sqlx::Error> {
        let mut tokens = Vec::new();

        for (field, content) in fields {
            let words = self.tokenize(content);

            for (position, word) in words.iter().enumerate() {
                let token = self.create_blind_index(word);
                tokens.push(BlindIndex {
                    token,
                    document_id,
                    field: field.to_string(),
                    position: position as u32,
                });
            }
        }

        // Batch insert
        for token in tokens {
            sqlx::query(
                r#"
                INSERT INTO encrypted_search_index (token, document_id, field, position)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (token, document_id, field, position) DO NOTHING
                "#,
            )
            .bind(&token.token)
            .bind(token.document_id)
            .bind(&token.field)
            .bind(token.position as i32)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    /// Search encrypted documents
    pub async fn search(
        &self,
        query: &str,
        limit: u32,
    ) -> Result<Vec<SearchResult>, sqlx::Error> {
        let query_words = self.tokenize(query);
        let query_tokens: Vec<Vec<u8>> = query_words
            .iter()
            .map(|w| self.create_blind_index(w))
            .collect();

        if query_tokens.is_empty() {
            return Ok(vec![]);
        }

        // For single word, simple lookup
        if query_tokens.len() == 1 {
            let results = sqlx::query_as::<_, (Uuid, i64)>(
                r#"
                SELECT document_id, COUNT(*) as matches
                FROM encrypted_search_index
                WHERE token = $1
                GROUP BY document_id
                ORDER BY matches DESC
                LIMIT $2
                "#,
            )
            .bind(&query_tokens[0])
            .bind(limit as i32)
            .fetch_all(&self.pool)
            .await?;

            return Ok(results
                .into_iter()
                .map(|(id, matches)| SearchResult {
                    document_id: id,
                    score: matches as f32,
                    field_matches: vec![],
                })
                .collect());
        }

        // For multiple words, find documents containing all words
        // Use set intersection
        let mut document_sets: Vec<HashSet<Uuid>> = Vec::new();

        for token in &query_tokens {
            let docs: Vec<Uuid> = sqlx::query_scalar(
                "SELECT DISTINCT document_id FROM encrypted_search_index WHERE token = $1",
            )
            .bind(token)
            .fetch_all(&self.pool)
            .await?;

            document_sets.push(docs.into_iter().collect());
        }

        // Intersect all sets
        let mut result_set = document_sets.pop().unwrap_or_default();
        for set in document_sets {
            result_set = result_set.intersection(&set).copied().collect();
        }

        // Score by match count
        let mut results: Vec<SearchResult> = Vec::new();
        for doc_id in result_set.into_iter().take(limit as usize) {
            let match_count: i64 = sqlx::query_scalar(
                r#"
                SELECT COUNT(*)
                FROM encrypted_search_index
                WHERE document_id = $1 AND token = ANY($2)
                "#,
            )
            .bind(doc_id)
            .bind(&query_tokens)
            .fetch_one(&self.pool)
            .await?;

            results.push(SearchResult {
                document_id: doc_id,
                score: match_count as f32,
                field_matches: vec![],
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    /// Search with phrase matching (words must appear consecutively)
    pub async fn phrase_search(
        &self,
        phrase: &str,
        limit: u32,
    ) -> Result<Vec<SearchResult>, sqlx::Error> {
        let words = self.tokenize(phrase);
        if words.len() < 2 {
            return self.search(phrase, limit).await;
        }

        let tokens: Vec<Vec<u8>> = words
            .iter()
            .map(|w| self.create_blind_index(w))
            .collect();

        // Find documents where tokens appear consecutively
        let first_token = &tokens[0];

        let candidates: Vec<(Uuid, String, i32)> = sqlx::query_as(
            r#"
            SELECT document_id, field, position
            FROM encrypted_search_index
            WHERE token = $1
            "#,
        )
        .bind(first_token)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();

        for (doc_id, field, start_pos) in candidates {
            let mut is_phrase_match = true;

            for (i, token) in tokens.iter().enumerate().skip(1) {
                let exists: bool = sqlx::query_scalar(
                    r#"
                    SELECT EXISTS(
                        SELECT 1 FROM encrypted_search_index
                        WHERE document_id = $1
                          AND field = $2
                          AND token = $3
                          AND position = $4
                    )
                    "#,
                )
                .bind(doc_id)
                .bind(&field)
                .bind(token)
                .bind(start_pos + i as i32)
                .fetch_one(&self.pool)
                .await?;

                if !exists {
                    is_phrase_match = false;
                    break;
                }
            }

            if is_phrase_match {
                results.push(SearchResult {
                    document_id: doc_id,
                    score: tokens.len() as f32,
                    field_matches: vec![field],
                });
            }

            if results.len() >= limit as usize {
                break;
            }
        }

        Ok(results)
    }

    /// Remove document from index
    pub async fn remove_document(&self, document_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM encrypted_search_index WHERE document_id = $1")
            .bind(document_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Create blind index token using keyed hash
    fn create_blind_index(&self, word: &str) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new_keyed(&self.blind_index_key);
        hasher.update(word.as_bytes());
        hasher.finalize().as_bytes()[..16].to_vec() // Truncate to 16 bytes
    }

    /// Tokenize text for indexing
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 2 && s.len() <= 50)
            .map(|s| s.to_string())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub document_id: Uuid,
    pub score: f32,
    pub field_matches: Vec<String>,
}

/// Migration for encrypted search index
pub const ENCRYPTED_SEARCH_MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS encrypted_search_index (
    token BYTEA NOT NULL,
    document_id UUID NOT NULL,
    field VARCHAR(50) NOT NULL,
    position INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (token, document_id, field, position)
);

CREATE INDEX IF NOT EXISTS idx_encrypted_search_token ON encrypted_search_index(token);
CREATE INDEX IF NOT EXISTS idx_encrypted_search_document ON encrypted_search_index(document_id);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let service = EncryptedSearchService::new(
            // Mock pool - won't be used in this test
            unsafe { std::mem::zeroed() },
            b"test_key",
        );

        let tokens = service.tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // Too short
    }

    #[test]
    fn test_blind_index_deterministic() {
        let service = EncryptedSearchService::new(
            unsafe { std::mem::zeroed() },
            b"test_key",
        );

        let token1 = service.create_blind_index("hello");
        let token2 = service.create_blind_index("hello");
        let token3 = service.create_blind_index("world");

        assert_eq!(token1, token2);
        assert_ne!(token1, token3);
    }
}
