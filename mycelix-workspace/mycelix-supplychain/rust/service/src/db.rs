// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database layer for persistent storage
//!
//! Provides SQLite-backed storage for claims and lineage with automatic migrations.

use anyhow::Result;
use claim_model::DkgClaim;
use sqlx::{sqlite::SqlitePool, Row};
use tracing::{debug, info};

/// Database connection pool and operations
#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Create a new database connection with automatic migrations
    pub async fn new(database_url: &str) -> Result<Self> {
        info!("Connecting to database: {}", database_url);

        let pool = SqlitePool::connect(database_url).await?;

        info!("Running database migrations...");
        sqlx::migrate!("./migrations").run(&pool).await?;

        info!("Database ready");

        Ok(Self { pool })
    }

    /// Store a claim in the database
    pub async fn store_claim(&self, claim: &DkgClaim) -> Result<()> {
        let claim_json = serde_json::to_string(claim)?;

        // Start a transaction for atomic operations
        let mut tx = self.pool.begin().await?;

        // Insert the claim
        sqlx::query(
            r#"
            INSERT INTO claims (id, issuer, batch_id, product_id, event_type, vc_jwt, lineage_hash, timestamp, claim_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&claim.id)
        .bind(&claim.issuer)
        .bind(&claim.subject.batch_id)
        .bind(&claim.subject.product_id)
        .bind(format!("{:?}", claim.assertion.event_type))
        .bind(&claim.evidence.vc_jwt)
        .bind(&claim.lineage.hash)
        .bind(claim.timestamp.to_rfc3339())
        .bind(&claim_json)
        .execute(&mut *tx)
        .await?;

        // Insert lineage relationships if any
        if let Some(prev_claims) = &claim.lineage.previous_claims {
            for parent_id in prev_claims {
                sqlx::query(
                    r#"
                    INSERT OR IGNORE INTO lineage (claim_id, parent_claim_id)
                    VALUES (?, ?)
                    "#,
                )
                .bind(&claim.id)
                .bind(parent_id)
                .execute(&mut *tx)
                .await?;
            }
        }

        // Commit transaction
        tx.commit().await?;

        debug!("Stored claim: {}", claim.id);
        Ok(())
    }

    /// Retrieve a claim by ID
    pub async fn get_claim(&self, id: &str) -> Result<Option<DkgClaim>> {
        let row = sqlx::query(
            r#"
            SELECT claim_json FROM claims WHERE id = ?
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let claim_json: String = row.try_get("claim_json")?;
            let claim: DkgClaim = serde_json::from_str(&claim_json)?;
            Ok(Some(claim))
        } else {
            Ok(None)
        }
    }

    /// Get all claims for a specific batch
    pub async fn get_batch_claims(&self, batch_id: &str) -> Result<Vec<DkgClaim>> {
        let rows = sqlx::query(
            r#"
            SELECT claim_json FROM claims
            WHERE batch_id = ?
            ORDER BY timestamp ASC
            "#,
        )
        .bind(batch_id)
        .fetch_all(&self.pool)
        .await?;

        let mut claims = Vec::new();
        for row in rows {
            let claim_json: String = row.try_get("claim_json")?;
            let claim: DkgClaim = serde_json::from_str(&claim_json)?;
            claims.push(claim);
        }

        Ok(claims)
    }

    /// Alias for get_batch_claims for API consistency
    pub async fn get_claims_by_batch(&self, batch_id: &str) -> Result<Vec<DkgClaim>> {
        self.get_batch_claims(batch_id).await
    }

    /// Get all claims (for search/filter operations)
    pub async fn get_all_claims(&self) -> Result<Vec<DkgClaim>> {
        let rows = sqlx::query(
            r#"
            SELECT claim_json FROM claims
            ORDER BY timestamp DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut claims = Vec::new();
        for row in rows {
            let claim_json: String = row.try_get("claim_json")?;
            let claim: DkgClaim = serde_json::from_str(&claim_json)?;
            claims.push(claim);
        }

        Ok(claims)
    }

    /// Get parent claims for a given claim ID
    pub async fn get_parent_claims(&self, claim_id: &str) -> Result<Vec<DkgClaim>> {
        let rows = sqlx::query(
            r#"
            SELECT c.claim_json
            FROM claims c
            INNER JOIN lineage l ON c.id = l.parent_claim_id
            WHERE l.claim_id = ?
            "#,
        )
        .bind(claim_id)
        .fetch_all(&self.pool)
        .await?;

        let mut claims = Vec::new();
        for row in rows {
            let claim_json: String = row.try_get("claim_json")?;
            let claim: DkgClaim = serde_json::from_str(&claim_json)?;
            claims.push(claim);
        }

        Ok(claims)
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> Result<DatabaseStats> {
        let total_claims: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM claims")
            .fetch_one(&self.pool)
            .await?;

        let total_lineage_links: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lineage")
            .fetch_one(&self.pool)
            .await?;

        Ok(DatabaseStats {
            total_claims: total_claims as u64,
            total_lineage_links: total_lineage_links as u64,
        })
    }

    /// Health check - verify database connection
    pub async fn health_check(&self) -> Result<bool> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map(|_| true)
            .or(Ok(false))
    }
}

/// Database statistics
#[derive(Debug)]
pub struct DatabaseStats {
    pub total_claims: u64,
    pub total_lineage_links: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use claim_model::{
        Assertion, CredentialSubject, EventType, Evidence, Facility, Lineage, Subject,
        SupplyEventVC,
    };
    use chrono::Utc;

    async fn create_test_db() -> Result<Database> {
        // Use in-memory SQLite for tests
        Database::new("sqlite::memory:").await
    }

    fn create_test_claim(id: &str, batch_id: &str) -> DkgClaim {
        let vc = SupplyEventVC {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            vc_type: vec!["VerifiableCredential".to_string()],
            issuer: "did:mycelix:org:test".to_string(),
            issuance_date: Utc::now(),
            expiration_date: None,
            credential_subject: CredentialSubject {
                event_type: EventType::Produced,
                product_id: "SKU-001".to_string(),
                batch_id: batch_id.to_string(),
                prev_batch_ids: None,
                quantity: 100.0,
                unit: "kg".to_string(),
                facility: Facility {
                    id: "FAC-001".to_string(),
                    name: "Test Facility".to_string(),
                    location: None,
                },
                timestamp: Utc::now(),
                shipment: None,
                certification: None,
                metadata: None,
            },
            proof: None,
        };

        DkgClaim::from_vc(&vc, "mock.jwt.token".to_string(), None)
    }

    #[tokio::test]
    async fn test_store_and_retrieve_claim() {
        let db = create_test_db().await.unwrap();
        let claim = create_test_claim("claim-1", "BATCH-001");

        // Store claim
        db.store_claim(&claim).await.unwrap();

        // Retrieve claim
        let retrieved = db.get_claim(&claim.id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, claim.id);
        assert_eq!(retrieved.subject.batch_id, "BATCH-001");
    }

    #[tokio::test]
    async fn test_get_batch_claims() {
        let db = create_test_db().await.unwrap();

        // Store multiple claims for the same batch
        let claim1 = create_test_claim("claim-1", "BATCH-001");
        let claim2 = create_test_claim("claim-2", "BATCH-001");
        let claim3 = create_test_claim("claim-3", "BATCH-002");

        db.store_claim(&claim1).await.unwrap();
        db.store_claim(&claim2).await.unwrap();
        db.store_claim(&claim3).await.unwrap();

        // Get claims for BATCH-001
        let batch_claims = db.get_batch_claims("BATCH-001").await.unwrap();
        assert_eq!(batch_claims.len(), 2);

        // Get claims for BATCH-002
        let batch_claims = db.get_batch_claims("BATCH-002").await.unwrap();
        assert_eq!(batch_claims.len(), 1);
    }

    #[tokio::test]
    async fn test_lineage_relationships() {
        let db = create_test_db().await.unwrap();

        // Create parent and child claims
        let parent = create_test_claim("parent-1", "BATCH-PARENT");
        let parent_id = parent.id.clone();
        db.store_claim(&parent).await.unwrap();

        // Create child with parent reference
        let mut child = create_test_claim("child-1", "BATCH-CHILD");
        child.lineage.previous_claims = Some(vec![parent_id.clone()]);
        let child_id = child.id.clone();
        db.store_claim(&child).await.unwrap();

        // Get parent claims
        let parents = db.get_parent_claims(&child_id).await.unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].id, parent_id);
    }

    #[tokio::test]
    async fn test_database_stats() {
        let db = create_test_db().await.unwrap();

        // Store some claims
        let claim1 = create_test_claim("claim-1", "BATCH-001");
        let claim1_id = claim1.id.clone();
        db.store_claim(&claim1).await.unwrap();

        let mut claim2 = create_test_claim("claim-2", "BATCH-002");
        claim2.lineage.previous_claims = Some(vec![claim1_id.clone()]);
        db.store_claim(&claim2).await.unwrap();

        // Get stats
        let stats = db.get_stats().await.unwrap();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.total_lineage_links, 1);
    }

    #[tokio::test]
    async fn test_health_check() {
        let db = create_test_db().await.unwrap();
        let healthy = db.health_check().await.unwrap();
        assert!(healthy);
    }
}
