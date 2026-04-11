// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database migrations module
//!
//! SQLx migration runner and schema versioning

use sqlx::PgPool;
use tracing::info;

/// Run all pending migrations
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::migrate::MigrateError> {
    info!("Running database migrations...");
    sqlx::migrate!("./migrations").run(pool).await?;
    info!("Migrations completed successfully");
    Ok(())
}

/// Check if migrations are up to date
pub async fn check_migrations(pool: &PgPool) -> Result<bool, sqlx::Error> {
    let result = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT COUNT(*) FROM _sqlx_migrations
        WHERE success = true
        "#,
    )
    .fetch_one(pool)
    .await;

    match result {
        Ok(count) => {
            info!("Found {} completed migrations", count);
            Ok(true)
        }
        Err(_) => {
            info!("Migration table not found - migrations needed");
            Ok(false)
        }
    }
}
