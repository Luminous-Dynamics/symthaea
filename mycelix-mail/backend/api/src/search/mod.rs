// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Search Engine
//!
//! Full-text search with operators, saved searches, and search history

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Search Engine
// ============================================================================

pub struct SearchEngine {
    pool: PgPool,
}

impl SearchEngine {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Parse and execute a search query with operators
    pub async fn search(
        &self,
        user_id: Uuid,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchResults, SearchError> {
        let parsed = self.parse_query(query)?;

        // Build SQL query from parsed operators
        let mut sql = String::from(
            r#"
            SELECT e.id, e.subject, e.from_address, e.from_name, e.body_text,
                   e.received_at, e.folder, e.is_read, e.has_attachments,
                   ts_rank(search_vector, plainto_tsquery('english', $2)) as rank
            FROM emails e
            WHERE e.user_id = $1
            "#
        );

        let mut param_count = 2;
        let mut conditions = Vec::new();

        // Full-text search on main query
        if !parsed.text_query.is_empty() {
            conditions.push(format!(
                "search_vector @@ plainto_tsquery('english', ${})",
                param_count
            ));
        }

        // Apply operator filters
        if let Some(from) = &parsed.from {
            param_count += 1;
            conditions.push(format!("LOWER(from_address) LIKE LOWER(${})", param_count));
        }

        if let Some(to) = &parsed.to {
            param_count += 1;
            conditions.push(format!("${}::text = ANY(to_addresses)", param_count));
        }

        if let Some(subject) = &parsed.subject {
            param_count += 1;
            conditions.push(format!("LOWER(subject) LIKE LOWER(${})", param_count));
        }

        if parsed.has_attachment {
            conditions.push("has_attachments = true".to_string());
        }

        if parsed.is_unread {
            conditions.push("is_read = false".to_string());
        }

        if parsed.is_starred {
            conditions.push("is_starred = true".to_string());
        }

        if let Some(folder) = &parsed.in_folder {
            param_count += 1;
            conditions.push(format!("LOWER(folder) = LOWER(${})", param_count));
        }

        if let Some(after) = &parsed.after {
            param_count += 1;
            conditions.push(format!("received_at >= ${}", param_count));
        }

        if let Some(before) = &parsed.before {
            param_count += 1;
            conditions.push(format!("received_at <= ${}", param_count));
        }

        if let Some(label) = &parsed.label {
            param_count += 1;
            conditions.push(format!("${}::text = ANY(labels)", param_count));
        }

        if !conditions.is_empty() {
            sql.push_str(" AND ");
            sql.push_str(&conditions.join(" AND "));
        }

        sql.push_str(" ORDER BY ");
        match options.sort_by {
            SortBy::Relevance => sql.push_str("rank DESC, received_at DESC"),
            SortBy::Date => sql.push_str("received_at DESC"),
            SortBy::DateAsc => sql.push_str("received_at ASC"),
            SortBy::Sender => sql.push_str("from_address ASC"),
        }

        sql.push_str(&format!(" LIMIT {} OFFSET {}", options.limit, options.offset));

        // Execute query (simplified - actual implementation would bind all params)
        let results: Vec<SearchResult> = sqlx::query_as(&sql)
            .bind(user_id)
            .bind(&parsed.text_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| SearchError::Database(e.to_string()))?;

        // Log search for history
        self.log_search(user_id, query).await?;

        Ok(SearchResults {
            results,
            total_count: results.len() as i64,
            query: query.to_string(),
            parsed_query: parsed,
        })
    }

    /// Parse search query into structured operators
    fn parse_query(&self, query: &str) -> Result<ParsedQuery, SearchError> {
        let mut parsed = ParsedQuery::default();
        let mut remaining_text = Vec::new();

        // Regex patterns for operators
        let operators = [
            ("from:", "from"),
            ("to:", "to"),
            ("subject:", "subject"),
            ("in:", "folder"),
            ("folder:", "folder"),
            ("label:", "label"),
            ("has:attachment", "has_attachment"),
            ("has:star", "is_starred"),
            ("is:unread", "is_unread"),
            ("is:read", "is_read"),
            ("is:starred", "is_starred"),
            ("after:", "after"),
            ("before:", "before"),
            ("older_than:", "before"),
            ("newer_than:", "after"),
        ];

        let mut i = 0;
        let chars: Vec<char> = query.chars().collect();

        while i < chars.len() {
            let remaining = &query[i..];

            let mut matched = false;
            for (op, field) in &operators {
                if remaining.to_lowercase().starts_with(*op) {
                    let value_start = i + op.len();
                    let value = self.extract_value(&query[value_start..]);

                    match *field {
                        "from" => parsed.from = Some(value.clone()),
                        "to" => parsed.to = Some(value.clone()),
                        "subject" => parsed.subject = Some(value.clone()),
                        "folder" => parsed.in_folder = Some(value.clone()),
                        "label" => parsed.label = Some(value.clone()),
                        "has_attachment" => parsed.has_attachment = true,
                        "is_starred" => parsed.is_starred = true,
                        "is_unread" => parsed.is_unread = true,
                        "is_read" => parsed.is_unread = false,
                        "after" => {
                            if let Ok(date) = self.parse_date(&value) {
                                parsed.after = Some(date);
                            }
                        }
                        "before" => {
                            if let Ok(date) = self.parse_date(&value) {
                                parsed.before = Some(date);
                            }
                        }
                        _ => {}
                    }

                    i = value_start + value.len();
                    if i < query.len() && query.chars().nth(i) == Some(' ') {
                        i += 1;
                    }
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Collect regular text
                let mut word = String::new();
                while i < chars.len() && chars[i] != ' ' {
                    word.push(chars[i]);
                    i += 1;
                }
                if !word.is_empty() {
                    remaining_text.push(word);
                }
                if i < chars.len() {
                    i += 1;
                }
            }
        }

        parsed.text_query = remaining_text.join(" ");
        Ok(parsed)
    }

    fn extract_value(&self, s: &str) -> String {
        if s.starts_with('"') {
            // Quoted value
            if let Some(end) = s[1..].find('"') {
                return s[1..end + 1].to_string();
            }
        }
        // Unquoted - take until space
        s.split_whitespace().next().unwrap_or("").to_string()
    }

    fn parse_date(&self, s: &str) -> Result<DateTime<Utc>, SearchError> {
        // Support various date formats
        if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            return Ok(date.and_hms_opt(0, 0, 0).unwrap().and_utc());
        }

        // Relative dates
        let now = Utc::now();
        match s.to_lowercase().as_str() {
            "today" => Ok(now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc()),
            "yesterday" => Ok((now - chrono::Duration::days(1)).date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc()),
            "week" | "1w" => Ok(now - chrono::Duration::weeks(1)),
            "month" | "1m" => Ok(now - chrono::Duration::days(30)),
            "year" | "1y" => Ok(now - chrono::Duration::days(365)),
            _ => Err(SearchError::InvalidDate(s.to_string())),
        }
    }

    async fn log_search(&self, user_id: Uuid, query: &str) -> Result<(), SearchError> {
        sqlx::query(
            "INSERT INTO search_history (id, user_id, query, searched_at) VALUES ($1, $2, $3, NOW())"
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(query)
        .execute(&self.pool)
        .await
        .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Saved Searches
// ============================================================================

pub struct SavedSearchService {
    pool: PgPool,
}

impl SavedSearchService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Save a search query for quick access
    pub async fn save_search(
        &self,
        user_id: Uuid,
        name: &str,
        query: &str,
    ) -> Result<SavedSearch, SearchError> {
        let id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO saved_searches (id, user_id, name, query, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#
        )
        .bind(id)
        .bind(user_id)
        .bind(name)
        .bind(query)
        .execute(&self.pool)
        .await
        .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(SavedSearch {
            id,
            user_id,
            name: name.to_string(),
            query: query.to_string(),
            created_at: Utc::now(),
            last_used_at: None,
            use_count: 0,
        })
    }

    /// Get all saved searches for a user
    pub async fn get_saved_searches(&self, user_id: Uuid) -> Result<Vec<SavedSearch>, SearchError> {
        let searches: Vec<SavedSearch> = sqlx::query_as(
            "SELECT * FROM saved_searches WHERE user_id = $1 ORDER BY use_count DESC, name ASC"
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(searches)
    }

    /// Delete a saved search
    pub async fn delete_saved_search(&self, user_id: Uuid, search_id: Uuid) -> Result<(), SearchError> {
        sqlx::query("DELETE FROM saved_searches WHERE id = $1 AND user_id = $2")
            .bind(search_id)
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(())
    }

    /// Increment usage counter when saved search is used
    pub async fn record_usage(&self, search_id: Uuid) -> Result<(), SearchError> {
        sqlx::query(
            "UPDATE saved_searches SET use_count = use_count + 1, last_used_at = NOW() WHERE id = $1"
        )
        .bind(search_id)
        .execute(&self.pool)
        .await
        .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Search History
// ============================================================================

pub struct SearchHistoryService {
    pool: PgPool,
}

impl SearchHistoryService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get recent search history
    pub async fn get_history(
        &self,
        user_id: Uuid,
        limit: i64,
    ) -> Result<Vec<SearchHistoryEntry>, SearchError> {
        let history: Vec<SearchHistoryEntry> = sqlx::query_as(
            r#"
            SELECT DISTINCT ON (query) id, user_id, query, searched_at
            FROM search_history
            WHERE user_id = $1
            ORDER BY query, searched_at DESC
            LIMIT $2
            "#
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(history)
    }

    /// Clear search history
    pub async fn clear_history(&self, user_id: Uuid) -> Result<(), SearchError> {
        sqlx::query("DELETE FROM search_history WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| SearchError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get search suggestions based on history and common patterns
    pub async fn get_suggestions(
        &self,
        user_id: Uuid,
        prefix: &str,
    ) -> Result<Vec<String>, SearchError> {
        let mut suggestions = Vec::new();

        // Operator suggestions
        let operators = [
            "from:", "to:", "subject:", "has:attachment", "is:unread",
            "is:starred", "in:", "label:", "after:", "before:",
        ];

        for op in operators {
            if op.starts_with(&prefix.to_lowercase()) {
                suggestions.push(op.to_string());
            }
        }

        // History-based suggestions
        let history: Vec<(String,)> = sqlx::query_as(
            r#"
            SELECT DISTINCT query FROM search_history
            WHERE user_id = $1 AND LOWER(query) LIKE LOWER($2)
            ORDER BY query
            LIMIT 5
            "#
        )
        .bind(user_id)
        .bind(format!("{}%", prefix))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SearchError::Database(e.to_string()))?;

        for (query,) in history {
            if !suggestions.contains(&query) {
                suggestions.push(query);
            }
        }

        Ok(suggestions)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedQuery {
    pub text_query: String,
    pub from: Option<String>,
    pub to: Option<String>,
    pub subject: Option<String>,
    pub in_folder: Option<String>,
    pub label: Option<String>,
    pub has_attachment: bool,
    pub is_unread: bool,
    pub is_starred: bool,
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    pub limit: i64,
    pub offset: i64,
    pub sort_by: SortBy,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 50,
            offset: 0,
            sort_by: SortBy::Relevance,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SortBy {
    Relevance,
    Date,
    DateAsc,
    Sender,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SearchResult {
    pub id: Uuid,
    pub subject: String,
    pub from_address: String,
    pub from_name: Option<String>,
    pub body_text: String,
    pub received_at: DateTime<Utc>,
    pub folder: String,
    pub is_read: bool,
    pub has_attachments: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub total_count: i64,
    pub query: String,
    pub parsed_query: ParsedQuery,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SavedSearch {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub query: String,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub use_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SearchHistoryEntry {
    pub id: Uuid,
    pub user_id: Uuid,
    pub query: String,
    pub searched_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    #[error("Invalid date: {0}")]
    InvalidDate(String),
}
