// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contact Management & CRM
//!
//! Contact sync, relationship tracking, enrichment, and mini-CRM features

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Contact Service
// ============================================================================

pub struct ContactService {
    pool: PgPool,
    enrichment: ContactEnrichment,
}

impl ContactService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: pool.clone(),
            enrichment: ContactEnrichment::new(pool),
        }
    }

    /// Create or update a contact
    pub async fn upsert_contact(
        &self,
        user_id: Uuid,
        contact: ContactInput,
    ) -> Result<Contact, ContactError> {
        let contact_id = contact.id.unwrap_or_else(Uuid::new_v4);

        sqlx::query(
            r#"
            INSERT INTO contacts (id, user_id, email, name, company, title, phone,
                                  avatar_url, notes, tags, custom_fields, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
            ON CONFLICT (user_id, email) DO UPDATE SET
                name = COALESCE(EXCLUDED.name, contacts.name),
                company = COALESCE(EXCLUDED.company, contacts.company),
                title = COALESCE(EXCLUDED.title, contacts.title),
                phone = COALESCE(EXCLUDED.phone, contacts.phone),
                avatar_url = COALESCE(EXCLUDED.avatar_url, contacts.avatar_url),
                notes = COALESCE(EXCLUDED.notes, contacts.notes),
                tags = EXCLUDED.tags,
                custom_fields = EXCLUDED.custom_fields,
                updated_at = NOW()
            RETURNING id
            "#,
        )
        .bind(contact_id)
        .bind(user_id)
        .bind(&contact.email)
        .bind(&contact.name)
        .bind(&contact.company)
        .bind(&contact.title)
        .bind(&contact.phone)
        .bind(&contact.avatar_url)
        .bind(&contact.notes)
        .bind(&contact.tags)
        .bind(serde_json::to_value(&contact.custom_fields).unwrap_or_default())
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        self.get_contact(user_id, contact_id).await
    }

    /// Get a contact by ID
    pub async fn get_contact(
        &self,
        user_id: Uuid,
        contact_id: Uuid,
    ) -> Result<Contact, ContactError> {
        let contact: Contact = sqlx::query_as(
            r#"
            SELECT id, user_id, email, name, company, title, phone, avatar_url,
                   notes, tags, custom_fields, relationship_strength, last_interaction,
                   interaction_count, created_at, updated_at
            FROM contacts
            WHERE id = $1 AND user_id = $2
            "#,
        )
        .bind(contact_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contact)
    }

    /// Search contacts
    pub async fn search_contacts(
        &self,
        user_id: Uuid,
        query: &str,
        limit: i64,
    ) -> Result<Vec<Contact>, ContactError> {
        let pattern = format!("%{}%", query.to_lowercase());

        let contacts: Vec<Contact> = sqlx::query_as(
            r#"
            SELECT id, user_id, email, name, company, title, phone, avatar_url,
                   notes, tags, custom_fields, relationship_strength, last_interaction,
                   interaction_count, created_at, updated_at
            FROM contacts
            WHERE user_id = $1
            AND (LOWER(email) LIKE $2 OR LOWER(name) LIKE $2 OR LOWER(company) LIKE $2)
            ORDER BY relationship_strength DESC, last_interaction DESC NULLS LAST
            LIMIT $3
            "#,
        )
        .bind(user_id)
        .bind(&pattern)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contacts)
    }

    /// Get contacts by tag
    pub async fn get_contacts_by_tag(
        &self,
        user_id: Uuid,
        tag: &str,
    ) -> Result<Vec<Contact>, ContactError> {
        let contacts: Vec<Contact> = sqlx::query_as(
            r#"
            SELECT id, user_id, email, name, company, title, phone, avatar_url,
                   notes, tags, custom_fields, relationship_strength, last_interaction,
                   interaction_count, created_at, updated_at
            FROM contacts
            WHERE user_id = $1 AND $2 = ANY(tags)
            ORDER BY name ASC
            "#,
        )
        .bind(user_id)
        .bind(tag)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contacts)
    }

    /// Get all tags for a user
    pub async fn get_all_tags(&self, user_id: Uuid) -> Result<Vec<TagWithCount>, ContactError> {
        let tags: Vec<TagWithCount> = sqlx::query_as(
            r#"
            SELECT UNNEST(tags) as tag, COUNT(*) as count
            FROM contacts
            WHERE user_id = $1
            GROUP BY tag
            ORDER BY count DESC
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(tags)
    }

    /// Delete a contact
    pub async fn delete_contact(
        &self,
        user_id: Uuid,
        contact_id: Uuid,
    ) -> Result<(), ContactError> {
        sqlx::query("DELETE FROM contacts WHERE id = $1 AND user_id = $2")
            .bind(contact_id)
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(())
    }

    /// Merge two contacts
    pub async fn merge_contacts(
        &self,
        user_id: Uuid,
        primary_id: Uuid,
        secondary_id: Uuid,
    ) -> Result<Contact, ContactError> {
        // Get both contacts
        let primary = self.get_contact(user_id, primary_id).await?;
        let secondary = self.get_contact(user_id, secondary_id).await?;

        // Merge data (primary takes precedence, fill in gaps from secondary)
        let merged_name = primary.name.or(secondary.name);
        let merged_company = primary.company.or(secondary.company);
        let merged_title = primary.title.or(secondary.title);
        let merged_phone = primary.phone.or(secondary.phone);

        // Merge tags
        let mut merged_tags: Vec<String> = primary.tags;
        for tag in secondary.tags {
            if !merged_tags.contains(&tag) {
                merged_tags.push(tag);
            }
        }

        // Update primary contact
        sqlx::query(
            r#"
            UPDATE contacts SET
                name = $3, company = $4, title = $5, phone = $6, tags = $7,
                interaction_count = interaction_count + $8,
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            "#,
        )
        .bind(primary_id)
        .bind(user_id)
        .bind(&merged_name)
        .bind(&merged_company)
        .bind(&merged_title)
        .bind(&merged_phone)
        .bind(&merged_tags)
        .bind(secondary.interaction_count)
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        // Update email references
        sqlx::query(
            "UPDATE emails SET contact_id = $1 WHERE contact_id = $2 AND user_id = $3",
        )
        .bind(primary_id)
        .bind(secondary_id)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .ok();

        // Delete secondary contact
        self.delete_contact(user_id, secondary_id).await?;

        self.get_contact(user_id, primary_id).await
    }

    /// Auto-create contact from email interaction
    pub async fn create_from_email(
        &self,
        user_id: Uuid,
        email_address: &str,
        name: Option<&str>,
    ) -> Result<Uuid, ContactError> {
        let contact_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO contacts (id, user_id, email, name, interaction_count, last_interaction, created_at, updated_at)
            VALUES ($1, $2, $3, $4, 1, NOW(), NOW(), NOW())
            ON CONFLICT (user_id, email) DO UPDATE SET
                interaction_count = contacts.interaction_count + 1,
                last_interaction = NOW()
            RETURNING id
            "#,
        )
        .bind(contact_id)
        .bind(user_id)
        .bind(email_address)
        .bind(name)
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contact_id)
    }
}

// ============================================================================
// Relationship Tracking
// ============================================================================

pub struct RelationshipTracker {
    pool: PgPool,
}

impl RelationshipTracker {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Record an interaction with a contact
    pub async fn record_interaction(
        &self,
        user_id: Uuid,
        contact_email: &str,
        interaction_type: InteractionType,
    ) -> Result<(), ContactError> {
        // Update contact stats
        sqlx::query(
            r#"
            UPDATE contacts SET
                interaction_count = interaction_count + 1,
                last_interaction = NOW()
            WHERE user_id = $1 AND email = $2
            "#,
        )
        .bind(user_id)
        .bind(contact_email)
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        // Record interaction detail
        sqlx::query(
            r#"
            INSERT INTO contact_interactions (id, user_id, contact_email, interaction_type, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(contact_email)
        .bind(interaction_type.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        // Recalculate relationship strength
        self.recalculate_strength(user_id, contact_email).await?;

        Ok(())
    }

    async fn recalculate_strength(
        &self,
        user_id: Uuid,
        contact_email: &str,
    ) -> Result<(), ContactError> {
        // Calculate based on:
        // - Recency of interactions
        // - Frequency of interactions
        // - Type of interactions (sent emails weight more than received)

        let stats: Option<RelationshipStats> = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) as total_interactions,
                COUNT(*) FILTER (WHERE interaction_type = 'email_sent') as emails_sent,
                COUNT(*) FILTER (WHERE interaction_type = 'email_received') as emails_received,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as recent_interactions,
                MAX(created_at) as last_interaction
            FROM contact_interactions
            WHERE user_id = $1 AND contact_email = $2
            "#,
        )
        .bind(user_id)
        .bind(contact_email)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        let strength = if let Some(stats) = stats {
            let base_score = (stats.total_interactions as f64).min(100.0);
            let recency_bonus = (stats.recent_interactions as f64 * 2.0).min(50.0);
            let sent_bonus = (stats.emails_sent as f64 * 1.5).min(30.0);

            ((base_score + recency_bonus + sent_bonus) / 180.0 * 100.0).min(100.0)
        } else {
            0.0
        };

        sqlx::query(
            "UPDATE contacts SET relationship_strength = $3 WHERE user_id = $1 AND email = $2",
        )
        .bind(user_id)
        .bind(contact_email)
        .bind(strength)
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get interaction history for a contact
    pub async fn get_interaction_history(
        &self,
        user_id: Uuid,
        contact_email: &str,
        limit: i64,
    ) -> Result<Vec<Interaction>, ContactError> {
        let interactions: Vec<Interaction> = sqlx::query_as(
            r#"
            SELECT id, interaction_type, created_at, email_id, notes
            FROM contact_interactions
            WHERE user_id = $1 AND contact_email = $2
            ORDER BY created_at DESC
            LIMIT $3
            "#,
        )
        .bind(user_id)
        .bind(contact_email)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(interactions)
    }

    /// Get top contacts by relationship strength
    pub async fn get_top_contacts(
        &self,
        user_id: Uuid,
        limit: i64,
    ) -> Result<Vec<Contact>, ContactError> {
        let contacts: Vec<Contact> = sqlx::query_as(
            r#"
            SELECT id, user_id, email, name, company, title, phone, avatar_url,
                   notes, tags, custom_fields, relationship_strength, last_interaction,
                   interaction_count, created_at, updated_at
            FROM contacts
            WHERE user_id = $1 AND relationship_strength > 0
            ORDER BY relationship_strength DESC
            LIMIT $2
            "#,
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contacts)
    }

    /// Get contacts that haven't been contacted recently
    pub async fn get_stale_contacts(
        &self,
        user_id: Uuid,
        days: i32,
        limit: i64,
    ) -> Result<Vec<Contact>, ContactError> {
        let contacts: Vec<Contact> = sqlx::query_as(
            r#"
            SELECT id, user_id, email, name, company, title, phone, avatar_url,
                   notes, tags, custom_fields, relationship_strength, last_interaction,
                   interaction_count, created_at, updated_at
            FROM contacts
            WHERE user_id = $1
            AND last_interaction < NOW() - INTERVAL '1 day' * $2
            AND relationship_strength > 20
            ORDER BY relationship_strength DESC
            LIMIT $3
            "#,
        )
        .bind(user_id)
        .bind(days)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contacts)
    }
}

// ============================================================================
// Contact Enrichment
// ============================================================================

pub struct ContactEnrichment {
    pool: PgPool,
}

impl ContactEnrichment {
    fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Enrich contact with additional data
    pub async fn enrich_contact(
        &self,
        user_id: Uuid,
        contact_id: Uuid,
    ) -> Result<EnrichmentResult, ContactError> {
        let contact: (String,) = sqlx::query_as(
            "SELECT email FROM contacts WHERE id = $1 AND user_id = $2",
        )
        .bind(contact_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        let email = contact.0;
        let domain = email.split('@').last().unwrap_or("");

        // Would call external APIs here (Clearbit, FullContact, etc.)
        // For now, just extract what we can from the domain

        let enrichment = EnrichmentResult {
            company_name: self.lookup_company(domain).await,
            company_domain: Some(domain.to_string()),
            linkedin_url: None,
            twitter_handle: None,
            location: None,
            industry: None,
        };

        // Update contact with enriched data
        if let Some(ref company) = enrichment.company_name {
            sqlx::query(
                "UPDATE contacts SET company = COALESCE(company, $3) WHERE id = $1 AND user_id = $2",
            )
            .bind(contact_id)
            .bind(user_id)
            .bind(company)
            .execute(&self.pool)
            .await
            .ok();
        }

        Ok(enrichment)
    }

    async fn lookup_company(&self, domain: &str) -> Option<String> {
        // Would use external API - for now, extract from domain
        let company = domain.split('.').next()?;
        if company.len() > 2 {
            Some(company.chars().next()?.to_uppercase().to_string() + &company[1..])
        } else {
            None
        }
    }
}

// ============================================================================
// Contact Sync
// ============================================================================

pub struct ContactSync {
    pool: PgPool,
}

impl ContactSync {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import contacts from CardDAV
    pub async fn import_carddav(
        &self,
        user_id: Uuid,
        carddav_url: &str,
        username: &str,
        password: &str,
    ) -> Result<ImportResult, ContactError> {
        // Would make CardDAV requests here
        // Parse vCard format and import contacts

        Ok(ImportResult {
            imported: 0,
            updated: 0,
            failed: 0,
            errors: Vec::new(),
        })
    }

    /// Export contacts to vCard format
    pub async fn export_vcard(
        &self,
        user_id: Uuid,
        contact_ids: Option<Vec<Uuid>>,
    ) -> Result<String, ContactError> {
        let contacts: Vec<Contact> = if let Some(ids) = contact_ids {
            sqlx::query_as(
                r#"
                SELECT id, user_id, email, name, company, title, phone, avatar_url,
                       notes, tags, custom_fields, relationship_strength, last_interaction,
                       interaction_count, created_at, updated_at
                FROM contacts
                WHERE user_id = $1 AND id = ANY($2)
                "#,
            )
            .bind(user_id)
            .bind(&ids)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ContactError::Database(e.to_string()))?
        } else {
            sqlx::query_as(
                r#"
                SELECT id, user_id, email, name, company, title, phone, avatar_url,
                       notes, tags, custom_fields, relationship_strength, last_interaction,
                       interaction_count, created_at, updated_at
                FROM contacts
                WHERE user_id = $1
                "#,
            )
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ContactError::Database(e.to_string()))?
        };

        let mut vcards = String::new();
        for contact in contacts {
            vcards.push_str(&self.contact_to_vcard(&contact));
            vcards.push_str("\n");
        }

        Ok(vcards)
    }

    fn contact_to_vcard(&self, contact: &Contact) -> String {
        let mut vcard = String::from("BEGIN:VCARD\nVERSION:3.0\n");

        if let Some(ref name) = contact.name {
            vcard.push_str(&format!("FN:{}\n", name));
            vcard.push_str(&format!("N:{}\n", name));
        }

        vcard.push_str(&format!("EMAIL:{}\n", contact.email));

        if let Some(ref phone) = contact.phone {
            vcard.push_str(&format!("TEL:{}\n", phone));
        }

        if let Some(ref company) = contact.company {
            vcard.push_str(&format!("ORG:{}\n", company));
        }

        if let Some(ref title) = contact.title {
            vcard.push_str(&format!("TITLE:{}\n", title));
        }

        vcard.push_str("END:VCARD");
        vcard
    }
}

// ============================================================================
// Contact Groups
// ============================================================================

pub struct ContactGroups {
    pool: PgPool,
}

impl ContactGroups {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a contact group
    pub async fn create_group(
        &self,
        user_id: Uuid,
        name: &str,
        description: Option<&str>,
    ) -> Result<ContactGroup, ContactError> {
        let group_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO contact_groups (id, user_id, name, description, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#,
        )
        .bind(group_id)
        .bind(user_id)
        .bind(name)
        .bind(description)
        .execute(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(ContactGroup {
            id: group_id,
            user_id,
            name: name.to_string(),
            description: description.map(String::from),
            member_count: 0,
            created_at: Utc::now(),
        })
    }

    /// Add contacts to a group
    pub async fn add_to_group(
        &self,
        user_id: Uuid,
        group_id: Uuid,
        contact_ids: &[Uuid],
    ) -> Result<(), ContactError> {
        for contact_id in contact_ids {
            sqlx::query(
                r#"
                INSERT INTO contact_group_members (group_id, contact_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                "#,
            )
            .bind(group_id)
            .bind(contact_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ContactError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Get group members
    pub async fn get_group_members(
        &self,
        user_id: Uuid,
        group_id: Uuid,
    ) -> Result<Vec<Contact>, ContactError> {
        let contacts: Vec<Contact> = sqlx::query_as(
            r#"
            SELECT c.id, c.user_id, c.email, c.name, c.company, c.title, c.phone, c.avatar_url,
                   c.notes, c.tags, c.custom_fields, c.relationship_strength, c.last_interaction,
                   c.interaction_count, c.created_at, c.updated_at
            FROM contacts c
            JOIN contact_group_members cgm ON cgm.contact_id = c.id
            WHERE c.user_id = $1 AND cgm.group_id = $2
            ORDER BY c.name ASC
            "#,
        )
        .bind(user_id)
        .bind(group_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ContactError::Database(e.to_string()))?;

        Ok(contacts)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInput {
    pub id: Option<Uuid>,
    pub email: String,
    pub name: Option<String>,
    pub company: Option<String>,
    pub title: Option<String>,
    pub phone: Option<String>,
    pub avatar_url: Option<String>,
    pub notes: Option<String>,
    pub tags: Vec<String>,
    pub custom_fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Contact {
    pub id: Uuid,
    pub user_id: Uuid,
    pub email: String,
    pub name: Option<String>,
    pub company: Option<String>,
    pub title: Option<String>,
    pub phone: Option<String>,
    pub avatar_url: Option<String>,
    pub notes: Option<String>,
    pub tags: Vec<String>,
    pub custom_fields: serde_json::Value,
    pub relationship_strength: f64,
    pub last_interaction: Option<DateTime<Utc>>,
    pub interaction_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TagWithCount {
    pub tag: String,
    pub count: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InteractionType {
    EmailSent,
    EmailReceived,
    MeetingScheduled,
    MeetingHeld,
    PhoneCall,
    Note,
}

impl ToString for InteractionType {
    fn to_string(&self) -> String {
        match self {
            InteractionType::EmailSent => "email_sent".to_string(),
            InteractionType::EmailReceived => "email_received".to_string(),
            InteractionType::MeetingScheduled => "meeting_scheduled".to_string(),
            InteractionType::MeetingHeld => "meeting_held".to_string(),
            InteractionType::PhoneCall => "phone_call".to_string(),
            InteractionType::Note => "note".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Interaction {
    pub id: Uuid,
    pub interaction_type: String,
    pub created_at: DateTime<Utc>,
    pub email_id: Option<Uuid>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct RelationshipStats {
    total_interactions: i64,
    emails_sent: i64,
    emails_received: i64,
    recent_interactions: i64,
    last_interaction: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentResult {
    pub company_name: Option<String>,
    pub company_domain: Option<String>,
    pub linkedin_url: Option<String>,
    pub twitter_handle: Option<String>,
    pub location: Option<String>,
    pub industry: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub imported: i32,
    pub updated: i32,
    pub failed: i32,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactGroup {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub member_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum ContactError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Contact not found")]
    NotFound,
    #[error("Sync error: {0}")]
    SyncError(String),
}
