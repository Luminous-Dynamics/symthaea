// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stock Levels and Inventory Movements
//!
//! Handles stock tracking, movements, adjustments, and valuation.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::error::ServiceError;

/// Stock movement type
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "movement_type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MovementType {
    /// Purchase receipt
    Receipt,
    /// Sales shipment
    Shipment,
    /// Transfer between locations
    Transfer,
    /// Inventory adjustment
    Adjustment,
    /// Return from customer
    Return,
    /// Scrap/damage write-off
    Scrap,
    /// Production consumption
    Consumption,
    /// Production output
    Production,
}

/// Stock valuation method
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "valuation_method", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValuationMethod {
    /// First In, First Out
    Fifo,
    /// Last In, First Out
    Lifo,
    /// Weighted Average Cost
    AverageCost,
    /// Standard Cost
    StandardCost,
}

/// Current stock level for a product at a location
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StockLevel {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub location_id: Option<Uuid>,
    pub quantity_on_hand: Decimal,
    pub quantity_reserved: Decimal,
    pub quantity_available: Decimal,
    pub quantity_on_order: Decimal,
    pub unit_cost: Option<Decimal>,
    pub total_value: Option<Decimal>,
    pub last_counted_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Stock movement record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StockMovement {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub location_id: Option<Uuid>,
    pub movement_type: String,
    pub quantity: Decimal,
    pub unit_cost: Option<Decimal>,
    pub reference_type: Option<String>,
    pub reference_id: Option<Uuid>,
    pub notes: Option<String>,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Create stock movement request
#[derive(Debug, Clone, Deserialize)]
pub struct CreateMovementRequest {
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub location_id: Option<Uuid>,
    pub movement_type: String,
    pub quantity: Decimal,
    pub unit_cost: Option<Decimal>,
    pub reference_type: Option<String>,
    pub reference_id: Option<Uuid>,
    pub notes: Option<String>,
}

/// Stock transfer request
#[derive(Debug, Clone, Deserialize)]
pub struct TransferRequest {
    pub product_id: Uuid,
    pub from_warehouse_id: Uuid,
    pub from_location_id: Option<Uuid>,
    pub to_warehouse_id: Uuid,
    pub to_location_id: Option<Uuid>,
    pub quantity: Decimal,
    pub notes: Option<String>,
}

/// Stock adjustment request
#[derive(Debug, Clone, Deserialize)]
pub struct AdjustmentRequest {
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub location_id: Option<Uuid>,
    pub new_quantity: Decimal,
    pub reason: String,
    pub notes: Option<String>,
}

/// Inventory count item
#[derive(Debug, Clone, Deserialize)]
pub struct CountItem {
    pub product_id: Uuid,
    pub counted_quantity: Decimal,
}

/// Stock service
#[derive(Clone)]
pub struct StockService {
    pool: PgPool,
}

impl StockService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get stock level for a product at a warehouse
    pub async fn get_stock_level(
        &self,
        tenant_id: Uuid,
        product_id: Uuid,
        warehouse_id: Uuid,
    ) -> Result<StockLevel, ServiceError> {
        let stock = sqlx::query_as!(
            StockLevel,
            r#"
            SELECT
                id, tenant_id, product_id, warehouse_id, location_id,
                quantity_on_hand, quantity_reserved, quantity_available, quantity_on_order,
                unit_cost, total_value, last_counted_at,
                created_at, updated_at
            FROM inv_stock_levels
            WHERE tenant_id = $1 AND product_id = $2 AND warehouse_id = $3
            "#,
            tenant_id,
            product_id,
            warehouse_id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?
        .ok_or(ServiceError::NotFound("Stock level not found".into()))?;

        Ok(stock)
    }

    /// List all stock levels for a warehouse
    pub async fn list_stock_by_warehouse(
        &self,
        tenant_id: Uuid,
        warehouse_id: Uuid,
    ) -> Result<Vec<StockLevelWithProduct>, ServiceError> {
        let stock = sqlx::query_as!(
            StockLevelWithProduct,
            r#"
            SELECT
                sl.id, sl.tenant_id, sl.product_id, sl.warehouse_id, sl.location_id,
                sl.quantity_on_hand, sl.quantity_reserved, sl.quantity_available, sl.quantity_on_order,
                sl.unit_cost, sl.total_value, sl.last_counted_at,
                sl.created_at, sl.updated_at,
                p.sku as product_sku, p.name as product_name
            FROM inv_stock_levels sl
            JOIN inv_products p ON sl.product_id = p.id
            WHERE sl.tenant_id = $1 AND sl.warehouse_id = $2
            ORDER BY p.name
            "#,
            tenant_id,
            warehouse_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(stock)
    }

    /// List all stock levels for a product across warehouses
    pub async fn list_stock_by_product(
        &self,
        tenant_id: Uuid,
        product_id: Uuid,
    ) -> Result<Vec<StockLevelWithWarehouse>, ServiceError> {
        let stock = sqlx::query_as!(
            StockLevelWithWarehouse,
            r#"
            SELECT
                sl.id, sl.tenant_id, sl.product_id, sl.warehouse_id, sl.location_id,
                sl.quantity_on_hand, sl.quantity_reserved, sl.quantity_available, sl.quantity_on_order,
                sl.unit_cost, sl.total_value, sl.last_counted_at,
                sl.created_at, sl.updated_at,
                w.code as warehouse_code, w.name as warehouse_name
            FROM inv_stock_levels sl
            JOIN inv_warehouses w ON sl.warehouse_id = w.id
            WHERE sl.tenant_id = $1 AND sl.product_id = $2
            ORDER BY w.name
            "#,
            tenant_id,
            product_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(stock)
    }

    /// Record a stock movement
    pub async fn record_movement(
        &self,
        tenant_id: Uuid,
        req: CreateMovementRequest,
        user_id: Option<Uuid>,
    ) -> Result<StockMovement, ServiceError> {
        let id = Uuid::new_v4();
        let now = Utc::now();

        // Start transaction
        let mut tx = self.pool.begin().await
            .map_err(|e| ServiceError::Database(e.to_string()))?;

        // Record the movement
        let movement = sqlx::query_as!(
            StockMovement,
            r#"
            INSERT INTO inv_stock_movements (
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, unit_cost,
                reference_type, reference_id, notes,
                created_by, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8,
                $9, $10, $11,
                $12, $13
            )
            RETURNING
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, unit_cost,
                reference_type, reference_id, notes,
                created_by, created_at
            "#,
            id,
            tenant_id,
            req.product_id,
            req.warehouse_id,
            req.location_id,
            req.movement_type,
            req.quantity,
            req.unit_cost,
            req.reference_type,
            req.reference_id,
            req.notes,
            user_id,
            now
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        // Update stock level
        let quantity_change = match req.movement_type.as_str() {
            "RECEIPT" | "RETURN" | "PRODUCTION" => req.quantity,
            "SHIPMENT" | "SCRAP" | "CONSUMPTION" => -req.quantity,
            _ => req.quantity,
        };

        sqlx::query!(
            r#"
            INSERT INTO inv_stock_levels (
                id, tenant_id, product_id, warehouse_id, location_id,
                quantity_on_hand, quantity_reserved, quantity_available, quantity_on_order,
                unit_cost, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, 0, $6, 0,
                $7, $8, $8
            )
            ON CONFLICT (tenant_id, product_id, warehouse_id, COALESCE(location_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET
                quantity_on_hand = inv_stock_levels.quantity_on_hand + $6,
                quantity_available = inv_stock_levels.quantity_available + $6,
                unit_cost = COALESCE($7, inv_stock_levels.unit_cost),
                total_value = (inv_stock_levels.quantity_on_hand + $6) * COALESCE($7, inv_stock_levels.unit_cost),
                updated_at = $8
            "#,
            Uuid::new_v4(),
            tenant_id,
            req.product_id,
            req.warehouse_id,
            req.location_id,
            quantity_change,
            req.unit_cost,
            now
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        tx.commit().await
            .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(movement)
    }

    /// Transfer stock between locations
    pub async fn transfer(
        &self,
        tenant_id: Uuid,
        req: TransferRequest,
        user_id: Option<Uuid>,
    ) -> Result<(StockMovement, StockMovement), ServiceError> {
        let now = Utc::now();

        // Start transaction
        let mut tx = self.pool.begin().await
            .map_err(|e| ServiceError::Database(e.to_string()))?;

        // Record outgoing movement
        let out_id = Uuid::new_v4();
        let out_movement = sqlx::query_as!(
            StockMovement,
            r#"
            INSERT INTO inv_stock_movements (
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, notes,
                created_by, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                'TRANSFER', -($6::DECIMAL), $7,
                $8, $9
            )
            RETURNING
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, unit_cost,
                reference_type, reference_id, notes,
                created_by, created_at
            "#,
            out_id,
            tenant_id,
            req.product_id,
            req.from_warehouse_id,
            req.from_location_id,
            req.quantity,
            req.notes,
            user_id,
            now
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        // Record incoming movement
        let in_id = Uuid::new_v4();
        let in_movement = sqlx::query_as!(
            StockMovement,
            r#"
            INSERT INTO inv_stock_movements (
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, notes,
                reference_id, created_by, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                'TRANSFER', $6, $7,
                $8, $9, $10
            )
            RETURNING
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, unit_cost,
                reference_type, reference_id, notes,
                created_by, created_at
            "#,
            in_id,
            tenant_id,
            req.product_id,
            req.to_warehouse_id,
            req.to_location_id,
            req.quantity,
            req.notes,
            out_id,
            user_id,
            now
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        // Update source stock level (decrease)
        sqlx::query!(
            r#"
            UPDATE inv_stock_levels SET
                quantity_on_hand = quantity_on_hand - $4,
                quantity_available = quantity_available - $4,
                total_value = (quantity_on_hand - $4) * COALESCE(unit_cost, 0),
                updated_at = $5
            WHERE tenant_id = $1 AND product_id = $2 AND warehouse_id = $3
            "#,
            tenant_id,
            req.product_id,
            req.from_warehouse_id,
            req.quantity,
            now
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        // Update destination stock level (increase)
        sqlx::query!(
            r#"
            INSERT INTO inv_stock_levels (
                id, tenant_id, product_id, warehouse_id, location_id,
                quantity_on_hand, quantity_reserved, quantity_available, quantity_on_order,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, 0, $6, 0,
                $7, $7
            )
            ON CONFLICT (tenant_id, product_id, warehouse_id, COALESCE(location_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET
                quantity_on_hand = inv_stock_levels.quantity_on_hand + $6,
                quantity_available = inv_stock_levels.quantity_available + $6,
                total_value = (inv_stock_levels.quantity_on_hand + $6) * COALESCE(inv_stock_levels.unit_cost, 0),
                updated_at = $7
            "#,
            Uuid::new_v4(),
            tenant_id,
            req.product_id,
            req.to_warehouse_id,
            req.to_location_id,
            req.quantity,
            now
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        tx.commit().await
            .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok((out_movement, in_movement))
    }

    /// Adjust stock quantity
    pub async fn adjust(
        &self,
        tenant_id: Uuid,
        req: AdjustmentRequest,
        user_id: Option<Uuid>,
    ) -> Result<StockMovement, ServiceError> {
        // Get current stock level
        let current = self.get_stock_level(tenant_id, req.product_id, req.warehouse_id).await?;
        let adjustment = req.new_quantity - current.quantity_on_hand;

        let movement_req = CreateMovementRequest {
            product_id: req.product_id,
            warehouse_id: req.warehouse_id,
            location_id: req.location_id,
            movement_type: "ADJUSTMENT".to_string(),
            quantity: adjustment,
            unit_cost: current.unit_cost,
            reference_type: Some("ADJUSTMENT".to_string()),
            reference_id: None,
            notes: Some(format!("{}: {}", req.reason, req.notes.unwrap_or_default())),
        };

        self.record_movement(tenant_id, movement_req, user_id).await
    }

    /// Get movement history for a product
    pub async fn get_movement_history(
        &self,
        tenant_id: Uuid,
        product_id: Uuid,
        limit: i64,
    ) -> Result<Vec<StockMovement>, ServiceError> {
        let movements = sqlx::query_as!(
            StockMovement,
            r#"
            SELECT
                id, tenant_id, product_id, warehouse_id, location_id,
                movement_type, quantity, unit_cost,
                reference_type, reference_id, notes,
                created_by, created_at
            FROM inv_stock_movements
            WHERE tenant_id = $1 AND product_id = $2
            ORDER BY created_at DESC
            LIMIT $3
            "#,
            tenant_id,
            product_id,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(movements)
    }

    /// Get inventory valuation summary
    pub async fn get_valuation_summary(&self, tenant_id: Uuid) -> Result<ValuationSummary, ServiceError> {
        let summary = sqlx::query!(
            r#"
            SELECT
                COUNT(DISTINCT product_id) as total_products,
                COALESCE(SUM(quantity_on_hand), 0) as total_quantity,
                COALESCE(SUM(total_value), 0) as total_value
            FROM inv_stock_levels
            WHERE tenant_id = $1
            "#,
            tenant_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        let by_warehouse = sqlx::query!(
            r#"
            SELECT
                w.id as warehouse_id,
                w.name as warehouse_name,
                COUNT(DISTINCT sl.product_id) as product_count,
                COALESCE(SUM(sl.quantity_on_hand), 0) as quantity,
                COALESCE(SUM(sl.total_value), 0) as value
            FROM inv_warehouses w
            LEFT JOIN inv_stock_levels sl ON w.id = sl.warehouse_id
            WHERE w.tenant_id = $1 AND w.is_active = true
            GROUP BY w.id, w.name
            ORDER BY w.name
            "#,
            tenant_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(ValuationSummary {
            total_products: summary.total_products.unwrap_or(0) as i32,
            total_quantity: summary.total_quantity.unwrap_or(Decimal::ZERO),
            total_value: summary.total_value.unwrap_or(Decimal::ZERO),
            by_warehouse: by_warehouse.into_iter().map(|w| WarehouseValuation {
                warehouse_id: w.warehouse_id,
                warehouse_name: w.warehouse_name,
                product_count: w.product_count.unwrap_or(0) as i32,
                quantity: w.quantity.unwrap_or(Decimal::ZERO),
                value: w.value.unwrap_or(Decimal::ZERO),
            }).collect(),
        })
    }

    /// Reserve stock for an order
    pub async fn reserve(
        &self,
        tenant_id: Uuid,
        product_id: Uuid,
        warehouse_id: Uuid,
        quantity: Decimal,
    ) -> Result<(), ServiceError> {
        let now = Utc::now();

        let result = sqlx::query!(
            r#"
            UPDATE inv_stock_levels SET
                quantity_reserved = quantity_reserved + $4,
                quantity_available = quantity_available - $4,
                updated_at = $5
            WHERE tenant_id = $1 AND product_id = $2 AND warehouse_id = $3
              AND quantity_available >= $4
            "#,
            tenant_id,
            product_id,
            warehouse_id,
            quantity,
            now
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(ServiceError::Validation("Insufficient available stock".into()));
        }

        Ok(())
    }

    /// Release reserved stock
    pub async fn release_reservation(
        &self,
        tenant_id: Uuid,
        product_id: Uuid,
        warehouse_id: Uuid,
        quantity: Decimal,
    ) -> Result<(), ServiceError> {
        let now = Utc::now();

        sqlx::query!(
            r#"
            UPDATE inv_stock_levels SET
                quantity_reserved = GREATEST(quantity_reserved - $4, 0),
                quantity_available = quantity_available + $4,
                updated_at = $5
            WHERE tenant_id = $1 AND product_id = $2 AND warehouse_id = $3
            "#,
            tenant_id,
            product_id,
            warehouse_id,
            quantity,
            now
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(())
    }
}

/// Stock level with product info
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StockLevelWithProduct {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub location_id: Option<Uuid>,
    pub quantity_on_hand: Decimal,
    pub quantity_reserved: Decimal,
    pub quantity_available: Decimal,
    pub quantity_on_order: Decimal,
    pub unit_cost: Option<Decimal>,
    pub total_value: Option<Decimal>,
    pub last_counted_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub product_sku: String,
    pub product_name: String,
}

/// Stock level with warehouse info
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StockLevelWithWarehouse {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub location_id: Option<Uuid>,
    pub quantity_on_hand: Decimal,
    pub quantity_reserved: Decimal,
    pub quantity_available: Decimal,
    pub quantity_on_order: Decimal,
    pub unit_cost: Option<Decimal>,
    pub total_value: Option<Decimal>,
    pub last_counted_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub warehouse_code: String,
    pub warehouse_name: String,
}

/// Inventory valuation summary
#[derive(Debug, Clone, Serialize)]
pub struct ValuationSummary {
    pub total_products: i32,
    pub total_quantity: Decimal,
    pub total_value: Decimal,
    pub by_warehouse: Vec<WarehouseValuation>,
}

/// Warehouse valuation
#[derive(Debug, Clone, Serialize)]
pub struct WarehouseValuation {
    pub warehouse_id: Uuid,
    pub warehouse_name: String,
    pub product_count: i32,
    pub quantity: Decimal,
    pub value: Decimal,
}
