// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Invoicing service
//!
//! Manages customer invoices and bills (accounts receivable/payable).

use chrono::Utc;
use serde_json::json;
use sha2::{Sha256, Digest};
use sqlx::PgPool;
use tracing::info;
use uuid::Uuid;

use super::models::*;

/// Service for managing invoices and bills
#[derive(Clone)]
pub struct InvoicingService {
    pool: PgPool,
}

impl InvoicingService {
    /// Create a new invoicing service
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new customer invoice
    pub async fn create_invoice(
        &self,
        req: CreateInvoiceRequest,
    ) -> Result<Invoice, Box<dyn std::error::Error>> {
        let mut tx = self.pool.begin().await?;
        let id = Uuid::new_v4();
        let now = Utc::now();
        let invoice_number = format!("INV-{}", id.to_string()[..8].to_uppercase());

        // Calculate totals from line items
        let mut subtotal = rust_decimal::Decimal::ZERO;
        let mut tax_amount = rust_decimal::Decimal::ZERO;

        for line in &req.lines {
            let line_total = line.quantity * line.unit_price;
            subtotal += line_total;

            if let Some(tax_rate) = line.tax_rate {
                tax_amount += line_total * tax_rate;
            }
        }

        let total_amount = subtotal + tax_amount;

        // Insert invoice
        let invoice = sqlx::query_as::<_, Invoice>(
            r#"
            INSERT INTO invoices (
                id, invoice_number, customer_id, invoice_date, due_date,
                currency, subtotal, tax_amount, total_amount, status,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&invoice_number)
        .bind(req.customer_id)
        .bind(req.invoice_date)
        .bind(req.due_date)
        .bind(&req.currency)
        .bind(subtotal)
        .bind(tax_amount)
        .bind(total_amount)
        .bind(InvoiceStatus::Draft)
        .bind(now)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;

        // Insert invoice lines
        for (line_num, line_req) in req.lines.iter().enumerate() {
            let line_total = line_req.quantity * line_req.unit_price;
            let line_tax = line_req.tax_rate.map(|rate| line_total * rate);

            sqlx::query(
                r#"
                INSERT INTO invoice_lines (
                    id, invoice_id, line_number, description, quantity,
                    unit_price, line_total, tax_rate, tax_amount, item_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                "#,
            )
            .bind(Uuid::new_v4())
            .bind(id)
            .bind(line_num as i32 + 1)
            .bind(&line_req.description)
            .bind(line_req.quantity)
            .bind(line_req.unit_price)
            .bind(line_total)
            .bind(line_req.tax_rate)
            .bind(line_tax)
            .bind(line_req.item_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(invoice)
    }

    /// Get an invoice by ID
    pub async fn get_invoice(&self, id: Uuid) -> Result<Option<Invoice>, sqlx::Error> {
        let invoice = sqlx::query_as::<_, Invoice>(
            "SELECT * FROM invoices WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(invoice)
    }

    /// List all invoices
    pub async fn list_invoices(&self) -> Result<Vec<Invoice>, sqlx::Error> {
        let invoices = sqlx::query_as::<_, Invoice>(
            "SELECT * FROM invoices ORDER BY invoice_date DESC, invoice_number DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(invoices)
    }

    /// Send an invoice to customer (change status to Sent)
    pub async fn send_invoice(
        &self,
        id: Uuid,
    ) -> Result<Invoice, Box<dyn std::error::Error>> {
        let now = Utc::now();

        // Update status
        let invoice = sqlx::query_as::<_, Invoice>(
            r#"
            UPDATE invoices
            SET status = 'SENT', updated_at = $1
            WHERE id = $2 AND status = 'DRAFT'
            RETURNING *
            "#,
        )
        .bind(now)
        .bind(id)
        .fetch_one(&self.pool)
        .await?;

        // Create DKG claim for audit trail
        let claim_id = self.create_invoice_claim(&invoice).await;

        // Update invoice with claim ID if created
        if let Some(ref cid) = claim_id {
            let _ = sqlx::query(
                "UPDATE invoices SET claim_id = $1 WHERE id = $2"
            )
            .bind(cid)
            .bind(id)
            .execute(&self.pool)
            .await;

            info!(
                invoice_id = %id,
                claim_id = %cid,
                "Created DKG claim for sent invoice"
            );
        }

        Ok(invoice)
    }

    /// Create a DKG claim for an invoice
    async fn create_invoice_claim(&self, invoice: &Invoice) -> Option<String> {
        // Build claim data structure for the invoice
        let claim_data = json!({
            "type": "FinancialInvoiceClaim",
            "invoice_number": invoice.invoice_number,
            "customer_id": invoice.customer_id.to_string(),
            "invoice_date": invoice.invoice_date.to_rfc3339(),
            "due_date": invoice.due_date.to_rfc3339(),
            "currency": invoice.currency,
            "subtotal": invoice.subtotal.to_string(),
            "tax_amount": invoice.tax_amount.to_string(),
            "total_amount": invoice.total_amount.to_string(),
            "status": format!("{:?}", invoice.status),
        });

        // Compute claim hash
        let mut hasher = Sha256::new();
        hasher.update(invoice.id.to_string().as_bytes());
        hasher.update(invoice.invoice_number.as_bytes());
        hasher.update(invoice.total_amount.to_string().as_bytes());
        hasher.update(invoice.invoice_date.to_rfc3339().as_bytes());
        let claim_hash = format!("{:x}", hasher.finalize());

        // Generate claim ID
        let claim_id = format!("fin:inv:{}", Uuid::new_v4());

        // Create the DKG claim structure
        let dkg_claim = claim_model::DkgClaim {
            id: claim_id.clone(),
            claim_type: "FinancialInvoiceClaim".to_string(),
            issuer: "did:mycelix:fin:invoices".to_string(),
            subject: claim_model::Subject {
                batch_id: invoice.invoice_number.clone(),
                product_id: invoice.id.to_string(),
            },
            assertion: claim_model::Assertion {
                event_type: claim_model::EventType::Certified,
                quantity: Some(invoice.total_amount.to_string().parse().unwrap_or(0.0)),
                unit: Some(invoice.currency.clone()),
                facility_id: None,
            },
            evidence: claim_model::Evidence {
                vc_jwt: serde_json::to_string(&claim_data).unwrap_or_default(),
                additional_documents: None,
            },
            lineage: claim_model::Lineage {
                hash: claim_hash,
                previous_claims: None,
            },
            timestamp: Utc::now(),
            confidence: Some(1.0),
            metadata: Some({
                let mut map = std::collections::HashMap::new();
                map.insert("invoice_id".to_string(), json!(invoice.id.to_string()));
                map.insert("invoice_number".to_string(), json!(invoice.invoice_number));
                map.insert("customer_id".to_string(), json!(invoice.customer_id.to_string()));
                map
            }),
        };

        // Publish to DKG
        match crate::dkg_client::publish_claim(&dkg_claim).await {
            Ok(_txid) => Some(claim_id),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to publish invoice claim to DKG");
                Some(claim_id)
            }
        }
    }

    /// Get invoice lines for an invoice
    pub async fn get_invoice_lines(&self, invoice_id: Uuid) -> Result<Vec<InvoiceLine>, sqlx::Error> {
        let lines = sqlx::query_as::<_, InvoiceLine>(
            "SELECT * FROM invoice_lines WHERE invoice_id = $1 ORDER BY line_number"
        )
        .bind(invoice_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(lines)
    }

    /// Create a new vendor bill
    pub async fn create_bill(
        &self,
        req: CreateBillRequest,
    ) -> Result<Bill, Box<dyn std::error::Error>> {
        let mut tx = self.pool.begin().await?;
        let id = Uuid::new_v4();
        let now = Utc::now();
        let bill_number = format!("BILL-{}", id.to_string()[..8].to_uppercase());

        // Calculate totals from line items
        let mut subtotal = rust_decimal::Decimal::ZERO;
        let mut tax_amount = rust_decimal::Decimal::ZERO;

        for line in &req.lines {
            let line_total = line.quantity * line.unit_price;
            subtotal += line_total;

            if let Some(tax_rate) = line.tax_rate {
                tax_amount += line_total * tax_rate;
            }
        }

        let total_amount = subtotal + tax_amount;

        // Insert bill
        let bill = sqlx::query_as::<_, Bill>(
            r#"
            INSERT INTO bills (
                id, bill_number, vendor_id, bill_date, due_date,
                currency, subtotal, tax_amount, total_amount, status,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&bill_number)
        .bind(req.vendor_id)
        .bind(req.bill_date)
        .bind(req.due_date)
        .bind(&req.currency)
        .bind(subtotal)
        .bind(tax_amount)
        .bind(total_amount)
        .bind(BillStatus::Draft)
        .bind(now)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;

        // Insert bill lines
        for (line_num, line_req) in req.lines.iter().enumerate() {
            let line_total = line_req.quantity * line_req.unit_price;
            let line_tax = line_req.tax_rate.map(|rate| line_total * rate);

            sqlx::query(
                r#"
                INSERT INTO bill_lines (
                    id, bill_id, line_number, description, quantity,
                    unit_price, line_total, tax_rate, tax_amount, expense_account_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                "#,
            )
            .bind(Uuid::new_v4())
            .bind(id)
            .bind(line_num as i32 + 1)
            .bind(&line_req.description)
            .bind(line_req.quantity)
            .bind(line_req.unit_price)
            .bind(line_total)
            .bind(line_req.tax_rate)
            .bind(line_tax)
            .bind(line_req.expense_account_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(bill)
    }

    /// Get a bill by ID
    pub async fn get_bill(&self, id: Uuid) -> Result<Option<Bill>, sqlx::Error> {
        let bill = sqlx::query_as::<_, Bill>(
            "SELECT * FROM bills WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(bill)
    }

    /// List all bills
    pub async fn list_bills(&self) -> Result<Vec<Bill>, sqlx::Error> {
        let bills = sqlx::query_as::<_, Bill>(
            "SELECT * FROM bills ORDER BY bill_date DESC, bill_number DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(bills)
    }

    /// Approve a bill for payment
    pub async fn approve_bill(
        &self,
        id: Uuid,
    ) -> Result<Bill, Box<dyn std::error::Error>> {
        let now = Utc::now();

        let bill = sqlx::query_as::<_, Bill>(
            r#"
            UPDATE bills
            SET status = 'APPROVED', updated_at = $1
            WHERE id = $2 AND status = 'DRAFT'
            RETURNING *
            "#,
        )
        .bind(now)
        .bind(id)
        .fetch_one(&self.pool)
        .await?;

        // Create DKG claim for audit trail
        let claim_id = self.create_bill_claim(&bill).await;

        // Update bill with claim ID if created
        if let Some(ref cid) = claim_id {
            let _ = sqlx::query(
                "UPDATE bills SET claim_id = $1 WHERE id = $2"
            )
            .bind(cid)
            .bind(id)
            .execute(&self.pool)
            .await;

            info!(
                bill_id = %id,
                claim_id = %cid,
                "Created DKG claim for approved bill"
            );
        }

        Ok(bill)
    }

    /// Create a DKG claim for a bill
    async fn create_bill_claim(&self, bill: &Bill) -> Option<String> {
        // Build claim data structure for the bill
        let claim_data = json!({
            "type": "FinancialBillClaim",
            "bill_number": bill.bill_number,
            "vendor_id": bill.vendor_id.to_string(),
            "bill_date": bill.bill_date.to_rfc3339(),
            "due_date": bill.due_date.to_rfc3339(),
            "currency": bill.currency,
            "subtotal": bill.subtotal.to_string(),
            "tax_amount": bill.tax_amount.to_string(),
            "total_amount": bill.total_amount.to_string(),
            "status": format!("{:?}", bill.status),
        });

        // Compute claim hash
        let mut hasher = Sha256::new();
        hasher.update(bill.id.to_string().as_bytes());
        hasher.update(bill.bill_number.as_bytes());
        hasher.update(bill.total_amount.to_string().as_bytes());
        hasher.update(bill.bill_date.to_rfc3339().as_bytes());
        let claim_hash = format!("{:x}", hasher.finalize());

        // Generate claim ID
        let claim_id = format!("fin:bill:{}", Uuid::new_v4());

        // Create the DKG claim structure
        let dkg_claim = claim_model::DkgClaim {
            id: claim_id.clone(),
            claim_type: "FinancialBillClaim".to_string(),
            issuer: "did:mycelix:fin:bills".to_string(),
            subject: claim_model::Subject {
                batch_id: bill.bill_number.clone(),
                product_id: bill.id.to_string(),
            },
            assertion: claim_model::Assertion {
                event_type: claim_model::EventType::Certified,
                quantity: Some(bill.total_amount.to_string().parse().unwrap_or(0.0)),
                unit: Some(bill.currency.clone()),
                facility_id: None,
            },
            evidence: claim_model::Evidence {
                vc_jwt: serde_json::to_string(&claim_data).unwrap_or_default(),
                additional_documents: None,
            },
            lineage: claim_model::Lineage {
                hash: claim_hash,
                previous_claims: None,
            },
            timestamp: Utc::now(),
            confidence: Some(1.0),
            metadata: Some({
                let mut map = std::collections::HashMap::new();
                map.insert("bill_id".to_string(), json!(bill.id.to_string()));
                map.insert("bill_number".to_string(), json!(bill.bill_number));
                map.insert("vendor_id".to_string(), json!(bill.vendor_id.to_string()));
                map
            }),
        };

        // Publish to DKG
        match crate::dkg_client::publish_claim(&dkg_claim).await {
            Ok(_txid) => Some(claim_id),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to publish bill claim to DKG");
                Some(claim_id)
            }
        }
    }
}
