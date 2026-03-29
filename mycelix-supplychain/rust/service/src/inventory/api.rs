// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inventory API Endpoints
//!
//! REST API for inventory management operations.

use axum::{
    extract::{Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::auth::Claims;
use crate::error::ServiceError;
use crate::PoolState;

use super::products::{CreateProductRequest, Product, ProductService, UpdateProductRequest, Category};
use super::warehouses::{CreateWarehouseRequest, CreateLocationRequest, Warehouse, WarehouseService, StorageLocation, UpdateWarehouseRequest, WarehouseSummary};
use super::stock::{CreateMovementRequest, StockService, StockMovement, TransferRequest, AdjustmentRequest, StockLevelWithProduct, StockLevelWithWarehouse, ValuationSummary};

/// Query parameters for product search
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
}

/// Query parameters for movement history
#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    #[serde(default = "default_limit")]
    pub limit: i64,
}

fn default_limit() -> i64 {
    50
}

/// Create inventory routes
pub fn routes() -> Router<PoolState> {
    Router::new()
        // Product endpoints
        .route("/products", get(list_products).post(create_product))
        .route("/products/search", get(search_products))
        .route("/products/low-stock", get(get_low_stock))
        .route("/products/:id", get(get_product).put(update_product))
        .route("/products/:id/stock", get(get_product_stock))
        .route("/products/:id/movements", get(get_product_movements))
        // Category endpoints
        .route("/categories", get(list_categories))
        // Warehouse endpoints
        .route("/warehouses", get(list_warehouses).post(create_warehouse))
        .route("/warehouses/:id", get(get_warehouse).put(update_warehouse))
        .route("/warehouses/:id/summary", get(get_warehouse_summary))
        .route("/warehouses/:id/locations", get(list_locations).post(create_location))
        .route("/warehouses/:id/stock", get(get_warehouse_stock))
        // Stock operations
        .route("/stock/movements", post(record_movement))
        .route("/stock/transfer", post(transfer_stock))
        .route("/stock/adjust", post(adjust_stock))
        .route("/stock/reserve", post(reserve_stock))
        .route("/stock/release", post(release_stock))
        .route("/stock/valuation", get(get_valuation))
}

// ============================================================================
// Product Endpoints
// ============================================================================

/// Get tenant_id from claims or return error
fn get_tenant_id(claims: &Claims) -> Result<uuid::Uuid, ServiceError> {
    claims.tenant_id.ok_or_else(|| ServiceError::BadRequest("Tenant ID required".into()))
}

/// List all products
async fn list_products(
    State(state): State<PoolState>,
    claims: Claims,
) -> Result<Json<Vec<Product>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let products = service.list(tenant_id).await?;
    Ok(Json(products))
}

/// Create a new product
async fn create_product(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<CreateProductRequest>,
) -> Result<Json<Product>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let product = service.create(tenant_id, req).await?;
    Ok(Json(product))
}

/// Get a product by ID
async fn get_product(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
) -> Result<Json<Product>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let product = service.get(tenant_id, id).await?;
    Ok(Json(product))
}

/// Update a product
async fn update_product(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateProductRequest>,
) -> Result<Json<Product>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let product = service.update(tenant_id, id, req).await?;
    Ok(Json(product))
}

/// Search products
async fn search_products(
    State(state): State<PoolState>,
    claims: Claims,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<Product>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let products = service.search(tenant_id, &query.q).await?;
    Ok(Json(products))
}

/// Get products below reorder point
async fn get_low_stock(
    State(state): State<PoolState>,
    claims: Claims,
) -> Result<Json<Vec<Product>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let products = service.get_low_stock(tenant_id).await?;
    Ok(Json(products))
}

/// Get stock levels for a product
async fn get_product_stock(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<StockLevelWithWarehouse>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let stock = service.list_stock_by_product(tenant_id, id).await?;
    Ok(Json(stock))
}

/// Get movement history for a product
async fn get_product_movements(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Vec<StockMovement>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let movements = service.get_movement_history(tenant_id, id, query.limit).await?;
    Ok(Json(movements))
}

// ============================================================================
// Category Endpoints
// ============================================================================

/// List all categories
async fn list_categories(
    State(state): State<PoolState>,
    claims: Claims,
) -> Result<Json<Vec<Category>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = ProductService::new(state.pool.clone());
    let categories = service.list_categories(tenant_id).await?;
    Ok(Json(categories))
}

// ============================================================================
// Warehouse Endpoints
// ============================================================================

/// List all warehouses
async fn list_warehouses(
    State(state): State<PoolState>,
    claims: Claims,
) -> Result<Json<Vec<Warehouse>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = WarehouseService::new(state.pool.clone());
    let warehouses = service.list_warehouses(tenant_id).await?;
    Ok(Json(warehouses))
}

/// Create a new warehouse
async fn create_warehouse(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<CreateWarehouseRequest>,
) -> Result<Json<Warehouse>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = WarehouseService::new(state.pool.clone());
    let warehouse = service.create_warehouse(tenant_id, req).await?;
    Ok(Json(warehouse))
}

/// Get a warehouse by ID
async fn get_warehouse(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
) -> Result<Json<Warehouse>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = WarehouseService::new(state.pool.clone());
    let warehouse = service.get_warehouse(tenant_id, id).await?;
    Ok(Json(warehouse))
}

/// Update a warehouse
async fn update_warehouse(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateWarehouseRequest>,
) -> Result<Json<Warehouse>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = WarehouseService::new(state.pool.clone());
    let warehouse = service.update_warehouse(tenant_id, id, req).await?;
    Ok(Json(warehouse))
}

/// Get warehouse summary with statistics
async fn get_warehouse_summary(
    State(state): State<PoolState>,
    claims: Claims,
    Path(id): Path<Uuid>,
) -> Result<Json<WarehouseSummary>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = WarehouseService::new(state.pool.clone());
    let summary = service.get_warehouse_summary(tenant_id, id).await?;
    Ok(Json(summary))
}

/// List locations in a warehouse
async fn list_locations(
    State(state): State<PoolState>,
    claims: Claims,
    Path(warehouse_id): Path<Uuid>,
) -> Result<Json<Vec<StorageLocation>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = WarehouseService::new(state.pool.clone());
    let locations = service.list_locations(tenant_id, warehouse_id).await?;
    Ok(Json(locations))
}

/// Create a location in a warehouse
async fn create_location(
    State(state): State<PoolState>,
    claims: Claims,
    Path(warehouse_id): Path<Uuid>,
    Json(mut req): Json<CreateLocationRequest>,
) -> Result<Json<StorageLocation>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    req.warehouse_id = warehouse_id;
    let service = WarehouseService::new(state.pool.clone());
    let location = service.create_location(tenant_id, req).await?;
    Ok(Json(location))
}

/// Get stock levels in a warehouse
async fn get_warehouse_stock(
    State(state): State<PoolState>,
    claims: Claims,
    Path(warehouse_id): Path<Uuid>,
) -> Result<Json<Vec<StockLevelWithProduct>>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let stock = service.list_stock_by_warehouse(tenant_id, warehouse_id).await?;
    Ok(Json(stock))
}

// ============================================================================
// Stock Operation Endpoints
// ============================================================================

/// Record a stock movement
async fn record_movement(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<CreateMovementRequest>,
) -> Result<Json<StockMovement>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let movement = service.record_movement(tenant_id, req, Some(claims.sub)).await?;
    Ok(Json(movement))
}

/// Transfer stock between locations
async fn transfer_stock(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<TransferRequest>,
) -> Result<Json<TransferResponse>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let (out_movement, in_movement) = service.transfer(tenant_id, req, Some(claims.sub)).await?;
    Ok(Json(TransferResponse {
        outgoing: out_movement,
        incoming: in_movement,
    }))
}

/// Adjust stock quantity
async fn adjust_stock(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<AdjustmentRequest>,
) -> Result<Json<StockMovement>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let movement = service.adjust(tenant_id, req, Some(claims.sub)).await?;
    Ok(Json(movement))
}

/// Reserve request
#[derive(Debug, Deserialize)]
pub struct ReserveRequest {
    pub product_id: Uuid,
    pub warehouse_id: Uuid,
    pub quantity: rust_decimal::Decimal,
}

/// Reserve stock for an order
async fn reserve_stock(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<ReserveRequest>,
) -> Result<Json<SuccessResponse>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    service.reserve(tenant_id, req.product_id, req.warehouse_id, req.quantity).await?;
    Ok(Json(SuccessResponse { success: true, message: "Stock reserved successfully".to_string() }))
}

/// Release stock reservation
async fn release_stock(
    State(state): State<PoolState>,
    claims: Claims,
    Json(req): Json<ReserveRequest>,
) -> Result<Json<SuccessResponse>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    service.release_reservation(tenant_id, req.product_id, req.warehouse_id, req.quantity).await?;
    Ok(Json(SuccessResponse { success: true, message: "Reservation released successfully".to_string() }))
}

/// Get inventory valuation
async fn get_valuation(
    State(state): State<PoolState>,
    claims: Claims,
) -> Result<Json<ValuationSummary>, ServiceError> {
    let tenant_id = get_tenant_id(&claims)?;
    let service = StockService::new(state.pool.clone());
    let valuation = service.get_valuation_summary(tenant_id).await?;
    Ok(Json(valuation))
}

// ============================================================================
// Response Types
// ============================================================================

/// Transfer response with both movements
#[derive(Debug, serde::Serialize)]
pub struct TransferResponse {
    pub outgoing: StockMovement,
    pub incoming: StockMovement,
}

/// Generic success response
#[derive(Debug, serde::Serialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub message: String,
}
