// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Department Management
//!
//! Organization structure and department hierarchy.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use super::HrError;

// ============================================================================
// Types
// ============================================================================

/// A department
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Department {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub name: String,
    pub code: Option<String>,
    pub description: Option<String>,
    pub parent_department_id: Option<Uuid>,
    pub manager_id: Option<Uuid>,
    pub cost_center: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Create department request
#[derive(Debug, Deserialize)]
pub struct CreateDepartmentRequest {
    pub name: String,
    pub code: Option<String>,
    pub description: Option<String>,
    pub parent_department_id: Option<Uuid>,
    pub manager_id: Option<Uuid>,
    pub cost_center: Option<String>,
}

/// Department with employee count
#[derive(Debug, Serialize)]
pub struct DepartmentWithCount {
    #[serde(flatten)]
    pub department: Department,
    pub employee_count: i64,
}

// ============================================================================
// Service
// ============================================================================

/// Department Management Service
#[derive(Clone)]
pub struct DepartmentService {
    pool: PgPool,
}

impl DepartmentService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new department
    pub async fn create_department(
        &self,
        tenant_id: Uuid,
        req: CreateDepartmentRequest,
    ) -> Result<Department, HrError> {
        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO hr_departments (
                id, tenant_id, name, code, description, parent_department_id,
                manager_id, cost_center
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&req.name)
        .bind(&req.code)
        .bind(&req.description)
        .bind(req.parent_department_id)
        .bind(req.manager_id)
        .bind(&req.cost_center)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_department(id).await
    }

    /// Get a department by ID
    pub async fn get_department(&self, id: Uuid) -> Result<Department, HrError> {
        sqlx::query_as::<_, Department>("SELECT * FROM hr_departments WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(HrError::Database)?
            .ok_or_else(|| HrError::NotFound("Department not found".into()))
    }

    /// List departments for a tenant
    pub async fn list_departments(&self, tenant_id: Uuid) -> Result<Vec<Department>, HrError> {
        let departments = sqlx::query_as::<_, Department>(
            "SELECT * FROM hr_departments WHERE tenant_id = $1 AND is_active = true ORDER BY name",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(departments)
    }

    /// Get departments with employee counts
    pub async fn list_with_counts(
        &self,
        tenant_id: Uuid,
    ) -> Result<Vec<DepartmentWithCount>, HrError> {
        // Get departments
        let departments = self.list_departments(tenant_id).await?;

        // Get counts per department
        let counts: Vec<(Uuid, i64)> = sqlx::query_as(
            "SELECT department_id, COUNT(*) as count
             FROM hr_employees
             WHERE is_active = true AND department_id IS NOT NULL
             GROUP BY department_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        // Build a map of department_id -> count
        let count_map: std::collections::HashMap<Uuid, i64> = counts.into_iter().collect();

        Ok(departments
            .into_iter()
            .map(|dept| {
                let count = count_map.get(&dept.id).copied().unwrap_or(0);
                DepartmentWithCount {
                    department: dept,
                    employee_count: count,
                }
            })
            .collect())
    }

    /// Get child departments
    pub async fn get_children(&self, parent_id: Uuid) -> Result<Vec<Department>, HrError> {
        let departments = sqlx::query_as::<_, Department>(
            "SELECT * FROM hr_departments
             WHERE parent_department_id = $1 AND is_active = true
             ORDER BY name",
        )
        .bind(parent_id)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(departments)
    }

    /// Update department
    pub async fn update_department(
        &self,
        id: Uuid,
        name: Option<&str>,
        description: Option<&str>,
        manager_id: Option<Uuid>,
    ) -> Result<Department, HrError> {
        sqlx::query(
            "UPDATE hr_departments SET
                name = COALESCE($2, name),
                description = COALESCE($3, description),
                manager_id = COALESCE($4, manager_id),
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(name)
        .bind(description)
        .bind(manager_id)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_department(id).await
    }

    /// Get organization chart (hierarchical)
    pub async fn get_org_chart(&self, tenant_id: Uuid) -> Result<Vec<OrgChartNode>, HrError> {
        let departments = self.list_departments(tenant_id).await?;

        let nodes: Vec<OrgChartNode> = departments
            .into_iter()
            .map(|d| OrgChartNode {
                id: d.id,
                name: d.name,
                parent_id: d.parent_department_id,
                manager_id: d.manager_id,
                children: vec![],
            })
            .collect();

        // Build tree (simplified - returns flat with parent references)
        Ok(nodes)
    }
}

/// Organization chart node
#[derive(Debug, Serialize)]
pub struct OrgChartNode {
    pub id: Uuid,
    pub name: String,
    pub parent_id: Option<Uuid>,
    pub manager_id: Option<Uuid>,
    pub children: Vec<OrgChartNode>,
}
