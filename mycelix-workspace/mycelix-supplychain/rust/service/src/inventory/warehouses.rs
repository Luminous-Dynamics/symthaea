// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Warehouse and Location Management
//!
//! Handles warehouse definitions, storage locations, and zones.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::error::ServiceError;

/// Warehouse type
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "warehouse_type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WarehouseType {
    /// Main distribution warehouse
    Distribution,
    /// Manufacturing facility
    Manufacturing,
    /// Retail store location
    Retail,
    /// Third-party logistics
    ThirdParty,
    /// Virtual/dropship location
    Virtual,
}

/// Location type within warehouse
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "location_type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LocationType {
    /// Standard storage rack
    Rack,
    /// Bulk storage area
    Bulk,
    /// Receiving dock
    Receiving,
    /// Shipping dock
    Shipping,
    /// Quality control area
    QualityControl,
    /// Returns processing
    Returns,
    /// Picking location
    Picking,
}

/// Warehouse definition
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Warehouse {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub code: String,
    pub name: String,
    pub warehouse_type: String,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub contact_name: Option<String>,
    pub contact_email: Option<String>,
    pub contact_phone: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Storage location within warehouse
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StorageLocation {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub warehouse_id: Uuid,
    pub code: String,
    pub name: String,
    pub location_type: String,
    pub zone: Option<String>,
    pub aisle: Option<String>,
    pub rack: Option<String>,
    pub shelf: Option<String>,
    pub bin: Option<String>,
    pub max_weight: Option<rust_decimal::Decimal>,
    pub max_volume: Option<rust_decimal::Decimal>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Create warehouse request
#[derive(Debug, Clone, Deserialize)]
pub struct CreateWarehouseRequest {
    pub code: String,
    pub name: String,
    pub warehouse_type: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub contact_name: Option<String>,
    pub contact_email: Option<String>,
    pub contact_phone: Option<String>,
}

/// Update warehouse request
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateWarehouseRequest {
    pub code: Option<String>,
    pub name: Option<String>,
    pub warehouse_type: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub contact_name: Option<String>,
    pub contact_email: Option<String>,
    pub contact_phone: Option<String>,
}

/// Create location request
#[derive(Debug, Clone, Deserialize)]
pub struct CreateLocationRequest {
    pub warehouse_id: Uuid,
    pub code: String,
    pub name: String,
    pub location_type: Option<String>,
    pub zone: Option<String>,
    pub aisle: Option<String>,
    pub rack: Option<String>,
    pub shelf: Option<String>,
    pub bin: Option<String>,
}

/// Warehouse service
#[derive(Clone)]
pub struct WarehouseService {
    pool: PgPool,
}

impl WarehouseService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// List all warehouses for a tenant
    pub async fn list_warehouses(&self, tenant_id: Uuid) -> Result<Vec<Warehouse>, ServiceError> {
        let warehouses = sqlx::query_as!(
            Warehouse,
            r#"
            SELECT
                id, tenant_id, code, name, warehouse_type,
                address_line1, address_line2, city, state, postal_code, country,
                contact_name, contact_email, contact_phone,
                is_active, created_at, updated_at
            FROM inv_warehouses
            WHERE tenant_id = $1 AND is_active = true
            ORDER BY name
            "#,
            tenant_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(warehouses)
    }

    /// Get a warehouse by ID
    pub async fn get_warehouse(&self, tenant_id: Uuid, id: Uuid) -> Result<Warehouse, ServiceError> {
        let warehouse = sqlx::query_as!(
            Warehouse,
            r#"
            SELECT
                id, tenant_id, code, name, warehouse_type,
                address_line1, address_line2, city, state, postal_code, country,
                contact_name, contact_email, contact_phone,
                is_active, created_at, updated_at
            FROM inv_warehouses
            WHERE tenant_id = $1 AND id = $2
            "#,
            tenant_id,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?
        .ok_or(ServiceError::NotFound("Warehouse not found".into()))?;

        Ok(warehouse)
    }

    /// Create a new warehouse
    pub async fn create_warehouse(&self, tenant_id: Uuid, req: CreateWarehouseRequest) -> Result<Warehouse, ServiceError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let warehouse_type = req.warehouse_type.unwrap_or_else(|| "DISTRIBUTION".to_string());

        let warehouse = sqlx::query_as!(
            Warehouse,
            r#"
            INSERT INTO inv_warehouses (
                id, tenant_id, code, name, warehouse_type,
                address_line1, address_line2, city, state, postal_code, country,
                contact_name, contact_email, contact_phone,
                is_active, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10, $11,
                $12, $13, $14,
                true, $15, $15
            )
            RETURNING
                id, tenant_id, code, name, warehouse_type,
                address_line1, address_line2, city, state, postal_code, country,
                contact_name, contact_email, contact_phone,
                is_active, created_at, updated_at
            "#,
            id,
            tenant_id,
            req.code,
            req.name,
            warehouse_type,
            req.address_line1,
            req.address_line2,
            req.city,
            req.state,
            req.postal_code,
            req.country,
            req.contact_name,
            req.contact_email,
            req.contact_phone,
            now
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(warehouse)
    }

    /// Update a warehouse
    pub async fn update_warehouse(&self, tenant_id: Uuid, id: Uuid, req: UpdateWarehouseRequest) -> Result<Warehouse, ServiceError> {
        let existing = self.get_warehouse(tenant_id, id).await?;
        let now = Utc::now();

        let warehouse = sqlx::query_as!(
            Warehouse,
            r#"
            UPDATE inv_warehouses SET
                code = COALESCE($3, code),
                name = COALESCE($4, name),
                warehouse_type = COALESCE($5, warehouse_type),
                address_line1 = COALESCE($6, address_line1),
                address_line2 = COALESCE($7, address_line2),
                city = COALESCE($8, city),
                state = COALESCE($9, state),
                postal_code = COALESCE($10, postal_code),
                country = COALESCE($11, country),
                contact_name = COALESCE($12, contact_name),
                contact_email = COALESCE($13, contact_email),
                contact_phone = COALESCE($14, contact_phone),
                updated_at = $15
            WHERE tenant_id = $1 AND id = $2
            RETURNING
                id, tenant_id, code, name, warehouse_type,
                address_line1, address_line2, city, state, postal_code, country,
                contact_name, contact_email, contact_phone,
                is_active, created_at, updated_at
            "#,
            tenant_id,
            id,
            req.code.unwrap_or(existing.code),
            req.name.unwrap_or(existing.name),
            req.warehouse_type.unwrap_or(existing.warehouse_type),
            req.address_line1.or(existing.address_line1),
            req.address_line2.or(existing.address_line2),
            req.city.or(existing.city),
            req.state.or(existing.state),
            req.postal_code.or(existing.postal_code),
            req.country.or(existing.country),
            req.contact_name.or(existing.contact_name),
            req.contact_email.or(existing.contact_email),
            req.contact_phone.or(existing.contact_phone),
            now
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(warehouse)
    }

    /// List locations in a warehouse
    pub async fn list_locations(&self, tenant_id: Uuid, warehouse_id: Uuid) -> Result<Vec<StorageLocation>, ServiceError> {
        let locations = sqlx::query_as!(
            StorageLocation,
            r#"
            SELECT
                id, tenant_id, warehouse_id, code, name, location_type,
                zone, aisle, rack, shelf, bin,
                max_weight, max_volume,
                is_active, created_at, updated_at
            FROM inv_locations
            WHERE tenant_id = $1 AND warehouse_id = $2 AND is_active = true
            ORDER BY zone, aisle, rack, shelf, bin
            "#,
            tenant_id,
            warehouse_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(locations)
    }

    /// Get a location by ID
    pub async fn get_location(&self, tenant_id: Uuid, id: Uuid) -> Result<StorageLocation, ServiceError> {
        let location = sqlx::query_as!(
            StorageLocation,
            r#"
            SELECT
                id, tenant_id, warehouse_id, code, name, location_type,
                zone, aisle, rack, shelf, bin,
                max_weight, max_volume,
                is_active, created_at, updated_at
            FROM inv_locations
            WHERE tenant_id = $1 AND id = $2
            "#,
            tenant_id,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?
        .ok_or(ServiceError::NotFound("Location not found".into()))?;

        Ok(location)
    }

    /// Create a new location
    pub async fn create_location(&self, tenant_id: Uuid, req: CreateLocationRequest) -> Result<StorageLocation, ServiceError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let location_type = req.location_type.unwrap_or_else(|| "RACK".to_string());

        let location = sqlx::query_as!(
            StorageLocation,
            r#"
            INSERT INTO inv_locations (
                id, tenant_id, warehouse_id, code, name, location_type,
                zone, aisle, rack, shelf, bin,
                is_active, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10, $11,
                true, $12, $12
            )
            RETURNING
                id, tenant_id, warehouse_id, code, name, location_type,
                zone, aisle, rack, shelf, bin,
                max_weight, max_volume,
                is_active, created_at, updated_at
            "#,
            id,
            tenant_id,
            req.warehouse_id,
            req.code,
            req.name,
            location_type,
            req.zone,
            req.aisle,
            req.rack,
            req.shelf,
            req.bin,
            now
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(location)
    }

    /// Get warehouse summary with location counts
    pub async fn get_warehouse_summary(&self, tenant_id: Uuid, warehouse_id: Uuid) -> Result<WarehouseSummary, ServiceError> {
        let warehouse = self.get_warehouse(tenant_id, warehouse_id).await?;

        let stats = sqlx::query!(
            r#"
            SELECT
                COUNT(*) as total_locations,
                COUNT(DISTINCT zone) as zone_count
            FROM inv_locations
            WHERE tenant_id = $1 AND warehouse_id = $2 AND is_active = true
            "#,
            tenant_id,
            warehouse_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        let stock_summary = sqlx::query!(
            r#"
            SELECT
                COUNT(DISTINCT product_id) as unique_products,
                COALESCE(SUM(quantity_on_hand), 0) as total_quantity
            FROM inv_stock_levels
            WHERE tenant_id = $1 AND warehouse_id = $2
            "#,
            tenant_id,
            warehouse_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(WarehouseSummary {
            warehouse,
            total_locations: stats.total_locations.unwrap_or(0) as i32,
            zone_count: stats.zone_count.unwrap_or(0) as i32,
            unique_products: stock_summary.unique_products.unwrap_or(0) as i32,
            total_quantity: stock_summary.total_quantity.unwrap_or(rust_decimal::Decimal::ZERO),
        })
    }
}

/// Warehouse summary with statistics
#[derive(Debug, Clone, Serialize)]
pub struct WarehouseSummary {
    pub warehouse: Warehouse,
    pub total_locations: i32,
    pub zone_count: i32,
    pub unique_products: i32,
    pub total_quantity: rust_decimal::Decimal,
}
