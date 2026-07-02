// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Product Catalog Management
//!
//! Handles product definitions, SKUs, categories, and variants.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::error::ServiceError;

/// Product type classification
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "product_type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProductType {
    /// Physical goods that need inventory tracking
    Stockable,
    /// Services that don't need inventory
    Service,
    /// Consumables tracked as expense
    Consumable,
}

/// Product status
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "product_status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProductStatus {
    Active,
    Inactive,
    Discontinued,
    Draft,
}

/// Unit of measure
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "unit_of_measure", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UnitOfMeasure {
    Each,
    Box,
    Case,
    Pallet,
    Kg,
    Lb,
    Liter,
    Gallon,
    Meter,
    Foot,
    SquareMeter,
    SquareFoot,
}

/// Product category
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Category {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub name: String,
    pub code: String,
    pub description: Option<String>,
    pub parent_category_id: Option<Uuid>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Product definition
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Product {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub sku: String,
    pub name: String,
    pub description: Option<String>,
    pub category_id: Option<Uuid>,
    pub product_type: String,
    pub status: String,
    pub unit_of_measure: String,
    pub cost_price: Option<Decimal>,
    pub sale_price: Option<Decimal>,
    pub currency: String,
    pub barcode: Option<String>,
    pub weight: Option<Decimal>,
    pub weight_unit: Option<String>,
    pub dimensions_length: Option<Decimal>,
    pub dimensions_width: Option<Decimal>,
    pub dimensions_height: Option<Decimal>,
    pub dimensions_unit: Option<String>,
    pub min_stock_level: Option<Decimal>,
    pub max_stock_level: Option<Decimal>,
    pub reorder_point: Option<Decimal>,
    pub reorder_quantity: Option<Decimal>,
    pub lead_time_days: Option<i32>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Create product request
#[derive(Debug, Clone, Deserialize)]
pub struct CreateProductRequest {
    pub sku: String,
    pub name: String,
    pub description: Option<String>,
    pub category_id: Option<Uuid>,
    pub product_type: Option<String>,
    pub unit_of_measure: Option<String>,
    pub cost_price: Option<Decimal>,
    pub sale_price: Option<Decimal>,
    pub currency: Option<String>,
    pub barcode: Option<String>,
    pub min_stock_level: Option<Decimal>,
    pub reorder_point: Option<Decimal>,
    pub reorder_quantity: Option<Decimal>,
}

/// Update product request
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateProductRequest {
    pub sku: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub category_id: Option<Uuid>,
    pub product_type: Option<String>,
    pub status: Option<String>,
    pub unit_of_measure: Option<String>,
    pub cost_price: Option<Decimal>,
    pub sale_price: Option<Decimal>,
    pub barcode: Option<String>,
    pub min_stock_level: Option<Decimal>,
    pub reorder_point: Option<Decimal>,
    pub reorder_quantity: Option<Decimal>,
}

/// Product service
#[derive(Clone)]
pub struct ProductService {
    pool: PgPool,
}

impl ProductService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// List all products for a tenant
    pub async fn list(&self, tenant_id: Uuid) -> Result<Vec<Product>, ServiceError> {
        let products = sqlx::query_as!(
            Product,
            r#"
            SELECT
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                weight, weight_unit,
                dimensions_length, dimensions_width, dimensions_height, dimensions_unit,
                min_stock_level, max_stock_level, reorder_point, reorder_quantity, lead_time_days,
                is_active, created_at, updated_at
            FROM inv_products
            WHERE tenant_id = $1 AND is_active = true
            ORDER BY name
            "#,
            tenant_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(products)
    }

    /// Get a product by ID
    pub async fn get(&self, tenant_id: Uuid, id: Uuid) -> Result<Product, ServiceError> {
        let product = sqlx::query_as!(
            Product,
            r#"
            SELECT
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                weight, weight_unit,
                dimensions_length, dimensions_width, dimensions_height, dimensions_unit,
                min_stock_level, max_stock_level, reorder_point, reorder_quantity, lead_time_days,
                is_active, created_at, updated_at
            FROM inv_products
            WHERE tenant_id = $1 AND id = $2
            "#,
            tenant_id,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?
        .ok_or(ServiceError::NotFound("Product not found".into()))?;

        Ok(product)
    }

    /// Get a product by SKU
    pub async fn get_by_sku(&self, tenant_id: Uuid, sku: &str) -> Result<Product, ServiceError> {
        let product = sqlx::query_as!(
            Product,
            r#"
            SELECT
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                weight, weight_unit,
                dimensions_length, dimensions_width, dimensions_height, dimensions_unit,
                min_stock_level, max_stock_level, reorder_point, reorder_quantity, lead_time_days,
                is_active, created_at, updated_at
            FROM inv_products
            WHERE tenant_id = $1 AND sku = $2
            "#,
            tenant_id,
            sku
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?
        .ok_or(ServiceError::NotFound("Product not found".into()))?;

        Ok(product)
    }

    /// Create a new product
    pub async fn create(&self, tenant_id: Uuid, req: CreateProductRequest) -> Result<Product, ServiceError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let product_type = req.product_type.unwrap_or_else(|| "STOCKABLE".to_string());
        let unit_of_measure = req.unit_of_measure.unwrap_or_else(|| "EACH".to_string());
        let currency = req.currency.unwrap_or_else(|| "USD".to_string());

        let product = sqlx::query_as!(
            Product,
            r#"
            INSERT INTO inv_products (
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                min_stock_level, reorder_point, reorder_quantity,
                is_active, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, 'ACTIVE', $8,
                $9, $10, $11, $12,
                $13, $14, $15,
                true, $16, $16
            )
            RETURNING
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                weight, weight_unit,
                dimensions_length, dimensions_width, dimensions_height, dimensions_unit,
                min_stock_level, max_stock_level, reorder_point, reorder_quantity, lead_time_days,
                is_active, created_at, updated_at
            "#,
            id,
            tenant_id,
            req.sku,
            req.name,
            req.description,
            req.category_id,
            product_type,
            unit_of_measure,
            req.cost_price,
            req.sale_price,
            currency,
            req.barcode,
            req.min_stock_level,
            req.reorder_point,
            req.reorder_quantity,
            now
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(product)
    }

    /// Update a product
    pub async fn update(&self, tenant_id: Uuid, id: Uuid, req: UpdateProductRequest) -> Result<Product, ServiceError> {
        let existing = self.get(tenant_id, id).await?;
        let now = Utc::now();

        let product = sqlx::query_as!(
            Product,
            r#"
            UPDATE inv_products SET
                sku = COALESCE($3, sku),
                name = COALESCE($4, name),
                description = COALESCE($5, description),
                category_id = COALESCE($6, category_id),
                product_type = COALESCE($7, product_type),
                status = COALESCE($8, status),
                unit_of_measure = COALESCE($9, unit_of_measure),
                cost_price = COALESCE($10, cost_price),
                sale_price = COALESCE($11, sale_price),
                barcode = COALESCE($12, barcode),
                min_stock_level = COALESCE($13, min_stock_level),
                reorder_point = COALESCE($14, reorder_point),
                reorder_quantity = COALESCE($15, reorder_quantity),
                updated_at = $16
            WHERE tenant_id = $1 AND id = $2
            RETURNING
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                weight, weight_unit,
                dimensions_length, dimensions_width, dimensions_height, dimensions_unit,
                min_stock_level, max_stock_level, reorder_point, reorder_quantity, lead_time_days,
                is_active, created_at, updated_at
            "#,
            tenant_id,
            id,
            req.sku.unwrap_or(existing.sku),
            req.name.unwrap_or(existing.name),
            req.description.or(existing.description),
            req.category_id.or(existing.category_id),
            req.product_type.unwrap_or(existing.product_type),
            req.status.unwrap_or(existing.status),
            req.unit_of_measure.unwrap_or(existing.unit_of_measure),
            req.cost_price.or(existing.cost_price),
            req.sale_price.or(existing.sale_price),
            req.barcode.or(existing.barcode),
            req.min_stock_level.or(existing.min_stock_level),
            req.reorder_point.or(existing.reorder_point),
            req.reorder_quantity.or(existing.reorder_quantity),
            now
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(product)
    }

    /// Search products
    pub async fn search(&self, tenant_id: Uuid, query: &str) -> Result<Vec<Product>, ServiceError> {
        let search_pattern = format!("%{}%", query);

        let products = sqlx::query_as!(
            Product,
            r#"
            SELECT
                id, tenant_id, sku, name, description, category_id,
                product_type, status, unit_of_measure,
                cost_price, sale_price, currency, barcode,
                weight, weight_unit,
                dimensions_length, dimensions_width, dimensions_height, dimensions_unit,
                min_stock_level, max_stock_level, reorder_point, reorder_quantity, lead_time_days,
                is_active, created_at, updated_at
            FROM inv_products
            WHERE tenant_id = $1
              AND is_active = true
              AND (
                  name ILIKE $2
                  OR sku ILIKE $2
                  OR barcode ILIKE $2
                  OR description ILIKE $2
              )
            ORDER BY name
            LIMIT 50
            "#,
            tenant_id,
            search_pattern
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(products)
    }

    /// Get products below reorder point
    pub async fn get_low_stock(&self, tenant_id: Uuid) -> Result<Vec<Product>, ServiceError> {
        let products = sqlx::query_as!(
            Product,
            r#"
            SELECT
                p.id, p.tenant_id, p.sku, p.name, p.description, p.category_id,
                p.product_type, p.status, p.unit_of_measure,
                p.cost_price, p.sale_price, p.currency, p.barcode,
                p.weight, p.weight_unit,
                p.dimensions_length, p.dimensions_width, p.dimensions_height, p.dimensions_unit,
                p.min_stock_level, p.max_stock_level, p.reorder_point, p.reorder_quantity, p.lead_time_days,
                p.is_active, p.created_at, p.updated_at
            FROM inv_products p
            LEFT JOIN inv_stock_levels sl ON p.id = sl.product_id
            WHERE p.tenant_id = $1
              AND p.is_active = true
              AND p.reorder_point IS NOT NULL
              AND (sl.quantity_on_hand IS NULL OR sl.quantity_on_hand <= p.reorder_point)
            ORDER BY p.name
            "#,
            tenant_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(products)
    }

    /// List categories
    pub async fn list_categories(&self, tenant_id: Uuid) -> Result<Vec<Category>, ServiceError> {
        let categories = sqlx::query_as!(
            Category,
            r#"
            SELECT id, tenant_id, name, code, description, parent_category_id, is_active, created_at, updated_at
            FROM inv_categories
            WHERE tenant_id = $1 AND is_active = true
            ORDER BY name
            "#,
            tenant_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ServiceError::Database(e.to_string()))?;

        Ok(categories)
    }
}
