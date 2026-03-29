// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Repository implementations for database access
//!
//! Clean repository pattern with async SQLx queries

use super::models::*;
use super::{DbError, DbResult};
use sqlx::PgPool;
use uuid::Uuid;

// ============================================================================
// User Repository
// ============================================================================

pub struct UserRepository {
    pool: PgPool,
}

impl UserRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find_by_id(&self, id: Uuid) -> DbResult<Option<User>> {
        let user = sqlx::query_as::<_, User>(
            r#"
            SELECT * FROM users WHERE id = $1 AND is_active = true
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(user)
    }

    pub async fn find_by_email(&self, email: &str) -> DbResult<Option<User>> {
        let user = sqlx::query_as::<_, User>(
            r#"
            SELECT * FROM users WHERE email = $1 AND is_active = true
            "#,
        )
        .bind(email)
        .fetch_optional(&self.pool)
        .await?;

        Ok(user)
    }

    pub async fn create(&self, data: CreateUser) -> DbResult<User> {
        let user = sqlx::query_as::<_, User>(
            r#"
            INSERT INTO users (email, display_name, avatar_url, holochain_agent_id, public_key)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            "#,
        )
        .bind(&data.email)
        .bind(&data.display_name)
        .bind(&data.avatar_url)
        .bind(&data.holochain_agent_id)
        .bind(&data.public_key)
        .fetch_one(&self.pool)
        .await?;

        Ok(user)
    }

    pub async fn update(&self, id: Uuid, data: UpdateUser) -> DbResult<User> {
        let user = sqlx::query_as::<_, User>(
            r#"
            UPDATE users SET
                display_name = COALESCE($2, display_name),
                avatar_url = COALESCE($3, avatar_url),
                holochain_agent_id = COALESCE($4, holochain_agent_id),
                public_key = COALESCE($5, public_key),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&data.display_name)
        .bind(&data.avatar_url)
        .bind(&data.holochain_agent_id)
        .bind(&data.public_key)
        .fetch_one(&self.pool)
        .await?;

        Ok(user)
    }

    pub async fn update_last_login(&self, id: Uuid) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE users SET last_login_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn deactivate(&self, id: Uuid) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE users SET is_active = false, updated_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Email Repository
// ============================================================================

pub struct EmailRepository {
    pool: PgPool,
}

impl EmailRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find_by_id(&self, id: Uuid) -> DbResult<Option<Email>> {
        let email = sqlx::query_as::<_, Email>(
            r#"
            SELECT * FROM emails WHERE id = $1 AND is_deleted = false
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(email)
    }

    pub async fn find_by_user(
        &self,
        user_id: Uuid,
        filter: EmailFilter,
        pagination: Pagination,
    ) -> DbResult<PaginatedResult<Email>> {
        // Build dynamic query based on filters
        let mut query = String::from(
            r#"
            SELECT * FROM emails
            WHERE user_id = $1 AND is_deleted = false
            "#,
        );
        let mut count_query = String::from(
            r#"
            SELECT COUNT(*) FROM emails
            WHERE user_id = $1 AND is_deleted = false
            "#,
        );

        let mut conditions = Vec::new();
        let mut param_idx = 2;

        if filter.is_read.is_some() {
            conditions.push(format!("AND is_read = ${}", param_idx));
            param_idx += 1;
        }
        if filter.is_starred.is_some() {
            conditions.push(format!("AND is_starred = ${}", param_idx));
            param_idx += 1;
        }
        if filter.is_archived.is_some() {
            conditions.push(format!("AND is_archived = ${}", param_idx));
            param_idx += 1;
        }
        if filter.min_trust_score.is_some() {
            conditions.push(format!("AND trust_score >= ${}", param_idx));
            param_idx += 1;
        }

        let condition_str = conditions.join(" ");
        query.push_str(&condition_str);
        count_query.push_str(&condition_str);

        query.push_str(&format!(
            " ORDER BY received_at DESC NULLS LAST LIMIT ${} OFFSET ${}",
            param_idx,
            param_idx + 1
        ));

        // Execute count query
        let mut count_builder = sqlx::query_scalar::<_, i64>(&count_query).bind(user_id);

        if let Some(is_read) = filter.is_read {
            count_builder = count_builder.bind(is_read);
        }
        if let Some(is_starred) = filter.is_starred {
            count_builder = count_builder.bind(is_starred);
        }
        if let Some(is_archived) = filter.is_archived {
            count_builder = count_builder.bind(is_archived);
        }
        if let Some(min_trust) = filter.min_trust_score {
            count_builder = count_builder.bind(min_trust);
        }

        let total = count_builder.fetch_one(&self.pool).await?;

        // Execute main query
        let mut query_builder = sqlx::query_as::<_, Email>(&query).bind(user_id);

        if let Some(is_read) = filter.is_read {
            query_builder = query_builder.bind(is_read);
        }
        if let Some(is_starred) = filter.is_starred {
            query_builder = query_builder.bind(is_starred);
        }
        if let Some(is_archived) = filter.is_archived {
            query_builder = query_builder.bind(is_archived);
        }
        if let Some(min_trust) = filter.min_trust_score {
            query_builder = query_builder.bind(min_trust);
        }

        let emails = query_builder
            .bind(pagination.limit() as i32)
            .bind(pagination.offset() as i32)
            .fetch_all(&self.pool)
            .await?;

        Ok(PaginatedResult::new(emails, total, &pagination))
    }

    pub async fn create(&self, data: CreateEmail) -> DbResult<Email> {
        let message_id = format!("<{}@mycelix.mail>", Uuid::new_v4());

        let email = sqlx::query_as::<_, Email>(
            r#"
            INSERT INTO emails (
                user_id, thread_id, message_id, in_reply_to, subject,
                body_text, body_html, from_address, to_addresses,
                cc_addresses, bcc_addresses, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'draft')
            RETURNING *
            "#,
        )
        .bind(data.user_id)
        .bind(data.thread_id)
        .bind(&message_id)
        .bind(&data.in_reply_to)
        .bind(&data.subject)
        .bind(&data.body_text)
        .bind(&data.body_html)
        .bind(&data.from_address)
        .bind(&data.to_addresses)
        .bind(&data.cc_addresses)
        .bind(&data.bcc_addresses)
        .fetch_one(&self.pool)
        .await?;

        Ok(email)
    }

    pub async fn update(&self, id: Uuid, data: UpdateEmail) -> DbResult<Email> {
        let email = sqlx::query_as::<_, Email>(
            r#"
            UPDATE emails SET
                subject = COALESCE($2, subject),
                body_text = COALESCE($3, body_text),
                body_html = COALESCE($4, body_html),
                is_read = COALESCE($5, is_read),
                is_starred = COALESCE($6, is_starred),
                is_archived = COALESCE($7, is_archived),
                labels = COALESCE($8, labels),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&data.subject)
        .bind(&data.body_text)
        .bind(&data.body_html)
        .bind(data.is_read)
        .bind(data.is_starred)
        .bind(data.is_archived)
        .bind(&data.labels)
        .fetch_one(&self.pool)
        .await?;

        Ok(email)
    }

    pub async fn mark_read(&self, id: Uuid, is_read: bool) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE emails SET is_read = $2, updated_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(is_read)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn soft_delete(&self, id: Uuid) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE emails SET is_deleted = true, updated_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_trust_score(&self, id: Uuid, score: f64) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE emails SET trust_score = $2, updated_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(score)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Contact Repository
// ============================================================================

pub struct ContactRepository {
    pool: PgPool,
}

impl ContactRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find_by_id(&self, id: Uuid) -> DbResult<Option<Contact>> {
        let contact = sqlx::query_as::<_, Contact>(
            r#"
            SELECT * FROM contacts WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(contact)
    }

    pub async fn find_by_user(&self, user_id: Uuid, pagination: Pagination) -> DbResult<PaginatedResult<Contact>> {
        let total = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT COUNT(*) FROM contacts WHERE user_id = $1
            "#,
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await?;

        let contacts = sqlx::query_as::<_, Contact>(
            r#"
            SELECT * FROM contacts
            WHERE user_id = $1
            ORDER BY interaction_count DESC, display_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(user_id)
        .bind(pagination.limit() as i32)
        .bind(pagination.offset() as i32)
        .fetch_all(&self.pool)
        .await?;

        Ok(PaginatedResult::new(contacts, total, &pagination))
    }

    pub async fn find_by_email(&self, user_id: Uuid, email: &str) -> DbResult<Option<Contact>> {
        let contact = sqlx::query_as::<_, Contact>(
            r#"
            SELECT * FROM contacts WHERE user_id = $1 AND email = $2
            "#,
        )
        .bind(user_id)
        .bind(email)
        .fetch_optional(&self.pool)
        .await?;

        Ok(contact)
    }

    pub async fn create(&self, data: CreateContact) -> DbResult<Contact> {
        let contact = sqlx::query_as::<_, Contact>(
            r#"
            INSERT INTO contacts (user_id, email, display_name, holochain_agent_id)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            "#,
        )
        .bind(data.user_id)
        .bind(&data.email)
        .bind(&data.display_name)
        .bind(&data.holochain_agent_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(contact)
    }

    pub async fn update(&self, id: Uuid, data: UpdateContact) -> DbResult<Contact> {
        let contact = sqlx::query_as::<_, Contact>(
            r#"
            UPDATE contacts SET
                display_name = COALESCE($2, display_name),
                avatar_url = COALESCE($3, avatar_url),
                holochain_agent_id = COALESCE($4, holochain_agent_id),
                trust_score = COALESCE($5, trust_score),
                is_blocked = COALESCE($6, is_blocked),
                notes = COALESCE($7, notes),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&data.display_name)
        .bind(&data.avatar_url)
        .bind(&data.holochain_agent_id)
        .bind(data.trust_score)
        .bind(data.is_blocked)
        .bind(&data.notes)
        .fetch_one(&self.pool)
        .await?;

        Ok(contact)
    }

    pub async fn increment_interaction(&self, id: Uuid) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE contacts SET
                interaction_count = interaction_count + 1,
                last_interaction_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_trust_score(&self, id: Uuid, score: f64) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE contacts SET trust_score = $2, updated_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(score)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete(&self, id: Uuid) -> DbResult<()> {
        sqlx::query(
            r#"
            DELETE FROM contacts WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Trust Attestation Repository
// ============================================================================

pub struct TrustAttestationRepository {
    pool: PgPool,
}

impl TrustAttestationRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find_by_id(&self, id: Uuid) -> DbResult<Option<TrustAttestation>> {
        let attestation = sqlx::query_as::<_, TrustAttestation>(
            r#"
            SELECT * FROM trust_attestations WHERE id = $1 AND revoked_at IS NULL
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(attestation)
    }

    pub async fn find_for_agent(&self, agent_id: &str) -> DbResult<Vec<TrustAttestation>> {
        let attestations = sqlx::query_as::<_, TrustAttestation>(
            r#"
            SELECT * FROM trust_attestations
            WHERE to_agent_id = $1
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY created_at DESC
            "#,
        )
        .bind(agent_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(attestations)
    }

    pub async fn create(&self, data: CreateTrustAttestation) -> DbResult<TrustAttestation> {
        let attestation = sqlx::query_as::<_, TrustAttestation>(
            r#"
            INSERT INTO trust_attestations (
                from_user_id, to_agent_id, trust_level, context,
                signature, holochain_hash, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            "#,
        )
        .bind(data.from_user_id)
        .bind(&data.to_agent_id)
        .bind(data.trust_level)
        .bind(&data.context)
        .bind(&data.signature)
        .bind(&data.holochain_hash)
        .bind(data.expires_at)
        .fetch_one(&self.pool)
        .await?;

        Ok(attestation)
    }

    pub async fn revoke(&self, id: Uuid) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE trust_attestations SET revoked_at = NOW() WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn calculate_aggregate_trust(&self, agent_id: &str) -> DbResult<f64> {
        let result = sqlx::query_scalar::<_, Option<f64>>(
            r#"
            SELECT AVG(trust_level) FROM trust_attestations
            WHERE to_agent_id = $1
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > NOW())
            "#,
        )
        .bind(agent_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.unwrap_or(0.0))
    }
}
