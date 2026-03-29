// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Tenant Architecture
//!
//! Workspace isolation and organization management

use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

/// Tenant (Organization/Workspace)
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Tenant {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub domain: Option<String>,
    pub settings: serde_json::Value,
    pub plan: TenantPlan,
    pub status: TenantStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "tenant_plan", rename_all = "lowercase")]
pub enum TenantPlan {
    Free,
    Pro,
    Enterprise,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "tenant_status", rename_all = "lowercase")]
pub enum TenantStatus {
    Active,
    Suspended,
    Deleted,
}

/// Tenant member with role
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TenantMember {
    pub tenant_id: Uuid,
    pub user_id: Uuid,
    pub role: TenantRole,
    pub permissions: Vec<String>,
    pub joined_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "tenant_role", rename_all = "lowercase")]
pub enum TenantRole {
    Owner,
    Admin,
    Member,
    Guest,
}

/// Tenant settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantSettings {
    pub default_encryption: bool,
    pub require_2fa: bool,
    pub allowed_domains: Vec<String>,
    pub retention_days: u32,
    pub max_attachment_size_mb: u32,
    pub custom_branding: Option<CustomBranding>,
    pub sso_config: Option<SsoConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomBranding {
    pub logo_url: Option<String>,
    pub primary_color: String,
    pub company_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsoConfig {
    pub provider: SsoProvider,
    pub client_id: String,
    pub issuer_url: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SsoProvider {
    Okta,
    AzureAD,
    Google,
    OneLogin,
    Custom,
}

impl Default for TenantSettings {
    fn default() -> Self {
        Self {
            default_encryption: true,
            require_2fa: false,
            allowed_domains: vec![],
            retention_days: 365,
            max_attachment_size_mb: 25,
            custom_branding: None,
            sso_config: None,
        }
    }
}

/// Tenant service for multi-tenant operations
pub struct TenantService {
    pool: PgPool,
}

impl TenantService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new tenant
    pub async fn create_tenant(
        &self,
        name: &str,
        slug: &str,
        owner_id: Uuid,
        plan: TenantPlan,
    ) -> Result<Tenant, sqlx::Error> {
        let tenant_id = Uuid::new_v4();
        let settings = serde_json::to_value(TenantSettings::default()).unwrap();

        // Create tenant
        let tenant = sqlx::query_as::<_, Tenant>(
            r#"
            INSERT INTO tenants (id, name, slug, settings, plan, status, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, 'active', NOW(), NOW())
            RETURNING *
            "#,
        )
        .bind(tenant_id)
        .bind(name)
        .bind(slug)
        .bind(&settings)
        .bind(plan)
        .fetch_one(&self.pool)
        .await?;

        // Add owner as member
        sqlx::query(
            r#"
            INSERT INTO tenant_members (tenant_id, user_id, role, permissions, joined_at)
            VALUES ($1, $2, 'owner', ARRAY['*'], NOW())
            "#,
        )
        .bind(tenant_id)
        .bind(owner_id)
        .execute(&self.pool)
        .await?;

        Ok(tenant)
    }

    /// Get tenant by ID
    pub async fn get_tenant(&self, id: Uuid) -> Result<Option<Tenant>, sqlx::Error> {
        sqlx::query_as::<_, Tenant>("SELECT * FROM tenants WHERE id = $1 AND status != 'deleted'")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
    }

    /// Get tenant by slug
    pub async fn get_tenant_by_slug(&self, slug: &str) -> Result<Option<Tenant>, sqlx::Error> {
        sqlx::query_as::<_, Tenant>(
            "SELECT * FROM tenants WHERE slug = $1 AND status != 'deleted'",
        )
        .bind(slug)
        .fetch_optional(&self.pool)
        .await
    }

    /// Get tenant by domain
    pub async fn get_tenant_by_domain(&self, domain: &str) -> Result<Option<Tenant>, sqlx::Error> {
        sqlx::query_as::<_, Tenant>(
            "SELECT * FROM tenants WHERE domain = $1 AND status != 'deleted'",
        )
        .bind(domain)
        .fetch_optional(&self.pool)
        .await
    }

    /// Update tenant settings
    pub async fn update_settings(
        &self,
        tenant_id: Uuid,
        settings: TenantSettings,
    ) -> Result<(), sqlx::Error> {
        let settings_json = serde_json::to_value(settings).unwrap();

        sqlx::query(
            r#"
            UPDATE tenants
            SET settings = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(tenant_id)
        .bind(&settings_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Add member to tenant
    pub async fn add_member(
        &self,
        tenant_id: Uuid,
        user_id: Uuid,
        role: TenantRole,
        permissions: Vec<String>,
    ) -> Result<TenantMember, sqlx::Error> {
        sqlx::query_as::<_, TenantMember>(
            r#"
            INSERT INTO tenant_members (tenant_id, user_id, role, permissions, joined_at)
            VALUES ($1, $2, $3, $4, NOW())
            RETURNING *
            "#,
        )
        .bind(tenant_id)
        .bind(user_id)
        .bind(role)
        .bind(&permissions)
        .fetch_one(&self.pool)
        .await
    }

    /// Remove member from tenant
    pub async fn remove_member(&self, tenant_id: Uuid, user_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM tenant_members WHERE tenant_id = $1 AND user_id = $2")
            .bind(tenant_id)
            .bind(user_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Get user's tenants
    pub async fn get_user_tenants(&self, user_id: Uuid) -> Result<Vec<Tenant>, sqlx::Error> {
        sqlx::query_as::<_, Tenant>(
            r#"
            SELECT t.*
            FROM tenants t
            JOIN tenant_members tm ON t.id = tm.tenant_id
            WHERE tm.user_id = $1 AND t.status != 'deleted'
            ORDER BY t.name
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
    }

    /// Get tenant members
    pub async fn get_members(&self, tenant_id: Uuid) -> Result<Vec<TenantMember>, sqlx::Error> {
        sqlx::query_as::<_, TenantMember>(
            "SELECT * FROM tenant_members WHERE tenant_id = $1 ORDER BY joined_at",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
    }

    /// Check if user is member of tenant
    pub async fn is_member(&self, tenant_id: Uuid, user_id: Uuid) -> Result<bool, sqlx::Error> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tenant_members WHERE tenant_id = $1 AND user_id = $2",
        )
        .bind(tenant_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(count > 0)
    }

    /// Get user's role in tenant
    pub async fn get_user_role(
        &self,
        tenant_id: Uuid,
        user_id: Uuid,
    ) -> Result<Option<TenantRole>, sqlx::Error> {
        sqlx::query_scalar("SELECT role FROM tenant_members WHERE tenant_id = $1 AND user_id = $2")
            .bind(tenant_id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
    }

    /// Check if user has permission
    pub async fn has_permission(
        &self,
        tenant_id: Uuid,
        user_id: Uuid,
        permission: &str,
    ) -> Result<bool, sqlx::Error> {
        let member = sqlx::query_as::<_, TenantMember>(
            "SELECT * FROM tenant_members WHERE tenant_id = $1 AND user_id = $2",
        )
        .bind(tenant_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;

        match member {
            Some(m) => {
                // Owner and admin have all permissions
                if m.role == TenantRole::Owner || m.role == TenantRole::Admin {
                    return Ok(true);
                }
                // Check specific permission
                Ok(m.permissions.contains(&"*".to_string())
                    || m.permissions.contains(&permission.to_string()))
            }
            None => Ok(false),
        }
    }

    /// Suspend tenant
    pub async fn suspend_tenant(&self, tenant_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE tenants SET status = 'suspended', updated_at = NOW() WHERE id = $1")
            .bind(tenant_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Reactivate tenant
    pub async fn reactivate_tenant(&self, tenant_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE tenants SET status = 'active', updated_at = NOW() WHERE id = $1")
            .bind(tenant_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Delete tenant (soft delete)
    pub async fn delete_tenant(&self, tenant_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE tenants SET status = 'deleted', updated_at = NOW() WHERE id = $1")
            .bind(tenant_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

/// Tenant context for request handling
#[derive(Debug, Clone)]
pub struct TenantContext {
    pub tenant: Tenant,
    pub user_id: Uuid,
    pub role: TenantRole,
    pub permissions: Vec<String>,
}

impl TenantContext {
    pub fn has_permission(&self, permission: &str) -> bool {
        self.role == TenantRole::Owner
            || self.role == TenantRole::Admin
            || self.permissions.contains(&"*".to_string())
            || self.permissions.contains(&permission.to_string())
    }

    pub fn is_admin(&self) -> bool {
        self.role == TenantRole::Owner || self.role == TenantRole::Admin
    }
}

/// Migration for multi-tenant tables
pub const TENANT_MIGRATION: &str = r#"
-- Tenant plans enum
DO $$ BEGIN
    CREATE TYPE tenant_plan AS ENUM ('free', 'pro', 'enterprise');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Tenant status enum
DO $$ BEGIN
    CREATE TYPE tenant_status AS ENUM ('active', 'suspended', 'deleted');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Tenant role enum
DO $$ BEGIN
    CREATE TYPE tenant_role AS ENUM ('owner', 'admin', 'member', 'guest');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    domain VARCHAR(255) UNIQUE,
    settings JSONB NOT NULL DEFAULT '{}',
    plan tenant_plan NOT NULL DEFAULT 'free',
    status tenant_status NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tenants_slug ON tenants(slug);
CREATE INDEX IF NOT EXISTS idx_tenants_domain ON tenants(domain);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);

-- Tenant members table
CREATE TABLE IF NOT EXISTS tenant_members (
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role tenant_role NOT NULL DEFAULT 'member',
    permissions TEXT[] NOT NULL DEFAULT '{}',
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_tenant_members_user ON tenant_members(user_id);

-- Add tenant_id to emails table for isolation
ALTER TABLE emails ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_emails_tenant ON emails(tenant_id);

-- Add tenant_id to contacts table
ALTER TABLE contacts ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_contacts_tenant ON contacts(tenant_id);

-- Row Level Security policies
ALTER TABLE emails ENABLE ROW LEVEL SECURITY;
ALTER TABLE contacts ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see emails in their tenant
CREATE POLICY IF NOT EXISTS tenant_isolation_emails ON emails
    USING (tenant_id IN (
        SELECT tenant_id FROM tenant_members WHERE user_id = current_user_id()
    ));

CREATE POLICY IF NOT EXISTS tenant_isolation_contacts ON contacts
    USING (tenant_id IN (
        SELECT tenant_id FROM tenant_members WHERE user_id = current_user_id()
    ));
"#;
