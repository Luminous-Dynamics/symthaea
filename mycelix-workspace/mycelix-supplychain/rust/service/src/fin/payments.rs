// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Payment processing service
//!
//! Manages payments for invoices and bills.

use chrono::Utc;
use serde_json::json;
use sha2::{Sha256, Digest};
use sqlx::PgPool;
use tracing::info;
use uuid::Uuid;

use super::models::*;

/// Service for managing payments
#[derive(Clone)]
pub struct PaymentService {
    pool: PgPool,
}

impl PaymentService {
    /// Create a new payment service
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new payment
    pub async fn create_payment(
        &self,
        req: CreatePaymentRequest,
    ) -> Result<Payment, Box<dyn std::error::Error>> {
        let mut tx = self.pool.begin().await?;
        let id = Uuid::new_v4();
        let now = Utc::now();
        let payment_number = format!("PAY-{}", id.to_string()[..8].to_uppercase());

        // Insert payment
        let payment = sqlx::query_as::<_, Payment>(
            r#"
            INSERT INTO payments (
                id, payment_number, payment_type, payment_date, amount,
                currency, payment_method, reference, invoice_id, bill_id,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&payment_number)
        .bind(&req.payment_type)
        .bind(req.payment_date)
        .bind(req.amount)
        .bind(&req.currency)
        .bind(&req.payment_method)
        .bind(&req.reference)
        .bind(req.invoice_id)
        .bind(req.bill_id)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;

        // Update invoice or bill status based on payment
        if let Some(invoice_id) = req.invoice_id {
            self.update_invoice_status(&mut tx, invoice_id).await?;
        }

        if let Some(bill_id) = req.bill_id {
            self.update_bill_status(&mut tx, bill_id).await?;
        }

        tx.commit().await?;

        // Create DKG claim for audit trail
        let claim_id = self.create_payment_claim(&payment).await;

        // Update payment with claim ID if created
        if let Some(ref cid) = claim_id {
            let _ = sqlx::query(
                "UPDATE payments SET claim_id = $1 WHERE id = $2"
            )
            .bind(cid)
            .bind(id)
            .execute(&self.pool)
            .await;

            info!(
                payment_id = %id,
                claim_id = %cid,
                "Created DKG claim for payment"
            );
        }

        // Create journal entry for the payment
        if let Some(journal_entry_id) = self.create_payment_journal_entry(&payment).await? {
            // Update payment with journal entry ID
            let _ = sqlx::query(
                "UPDATE payments SET journal_entry_id = $1 WHERE id = $2"
            )
            .bind(journal_entry_id)
            .bind(id)
            .execute(&self.pool)
            .await;

            info!(
                payment_id = %id,
                journal_entry_id = %journal_entry_id,
                "Created journal entry for payment"
            );
        }

        Ok(payment)
    }

    /// Create a DKG claim for a payment
    async fn create_payment_claim(&self, payment: &Payment) -> Option<String> {
        // Build claim data structure for the payment
        let claim_data = json!({
            "type": "FinancialPaymentClaim",
            "payment_number": payment.payment_number,
            "payment_type": format!("{:?}", payment.payment_type),
            "payment_date": payment.payment_date.to_rfc3339(),
            "amount": payment.amount.to_string(),
            "currency": payment.currency,
            "payment_method": format!("{:?}", payment.payment_method),
            "invoice_id": payment.invoice_id.map(|id| id.to_string()),
            "bill_id": payment.bill_id.map(|id| id.to_string()),
        });

        // Compute claim hash
        let mut hasher = Sha256::new();
        hasher.update(payment.id.to_string().as_bytes());
        hasher.update(payment.payment_number.as_bytes());
        hasher.update(payment.amount.to_string().as_bytes());
        hasher.update(payment.payment_date.to_rfc3339().as_bytes());
        let claim_hash = format!("{:x}", hasher.finalize());

        // Generate claim ID
        let claim_id = format!("fin:pay:{}", Uuid::new_v4());

        // Create the DKG claim structure
        let dkg_claim = claim_model::DkgClaim {
            id: claim_id.clone(),
            claim_type: "FinancialPaymentClaim".to_string(),
            issuer: "did:mycelix:fin:payments".to_string(),
            subject: claim_model::Subject {
                batch_id: payment.payment_number.clone(),
                product_id: payment.id.to_string(),
            },
            assertion: claim_model::Assertion {
                event_type: claim_model::EventType::Certified,
                quantity: Some(payment.amount.to_string().parse().unwrap_or(0.0)),
                unit: Some(payment.currency.clone()),
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
                map.insert("payment_id".to_string(), json!(payment.id.to_string()));
                map.insert("payment_number".to_string(), json!(payment.payment_number));
                if let Some(inv_id) = payment.invoice_id {
                    map.insert("invoice_id".to_string(), json!(inv_id.to_string()));
                }
                if let Some(bill_id) = payment.bill_id {
                    map.insert("bill_id".to_string(), json!(bill_id.to_string()));
                }
                map
            }),
        };

        // Publish to DKG
        match crate::dkg_client::publish_claim(&dkg_claim).await {
            Ok(_txid) => Some(claim_id),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to publish payment claim to DKG");
                Some(claim_id)
            }
        }
    }

    /// Create a journal entry for a payment
    async fn create_payment_journal_entry(&self, payment: &Payment) -> Result<Option<Uuid>, Box<dyn std::error::Error>> {
        // Determine the GL accounts based on payment type
        // For receivable payments (customer paying invoice): DR Cash, CR Accounts Receivable
        // For payable payments (paying vendor bill): DR Accounts Payable, CR Cash
        let (debit_account, credit_account) = match payment.payment_type {
            PaymentType::Receivable => {
                // Customer payment received - DR Cash, CR A/R
                ("1100", "1200") // Cash / Accounts Receivable
            }
            PaymentType::Payable => {
                // Vendor payment made - DR A/P, CR Cash
                ("2100", "1100") // Accounts Payable / Cash
            }
        };

        // Look up GL account IDs by account number (with fallback)
        let debit_account_id: Option<Uuid> = sqlx::query_scalar(
            "SELECT id FROM gl_accounts WHERE account_number = $1 AND is_active = true LIMIT 1"
        )
        .bind(debit_account)
        .fetch_optional(&self.pool)
        .await?;

        let credit_account_id: Option<Uuid> = sqlx::query_scalar(
            "SELECT id FROM gl_accounts WHERE account_number = $1 AND is_active = true LIMIT 1"
        )
        .bind(credit_account)
        .fetch_optional(&self.pool)
        .await?;

        // If we don't have the required GL accounts set up, skip journal entry creation
        let (debit_id, credit_id) = match (debit_account_id, credit_account_id) {
            (Some(d), Some(c)) => (d, c),
            _ => {
                tracing::warn!(
                    payment_id = %payment.id,
                    "Cannot create journal entry - required GL accounts not found"
                );
                return Ok(None);
            }
        };

        // Create journal entry
        let entry_id = Uuid::new_v4();
        let now = Utc::now();
        let entry_number = format!("JE-PAY-{}", entry_id.to_string()[..8].to_uppercase());

        let description = match payment.payment_type {
            PaymentType::Receivable => format!("Payment received - {}", payment.payment_number),
            PaymentType::Payable => format!("Payment made - {}", payment.payment_number),
        };

        // Calculate lines hash
        let lines_data = format!("{}:{}:{}:{}", debit_id, credit_id, payment.amount, payment.amount);
        let mut hasher = Sha256::new();
        hasher.update(lines_data.as_bytes());
        let lines_hash = format!("{:x}", hasher.finalize());

        let mut tx = self.pool.begin().await?;

        // Insert journal entry - auto-posted for payment entries
        sqlx::query(
            r#"
            INSERT INTO journal_entries (
                id, entry_number, entry_date, description, reference,
                status, lines_hash, created_by, created_at, posted_at
            )
            VALUES ($1, $2, $3, $4, $5, 'POSTED', $6, $7, $8, $9)
            "#,
        )
        .bind(entry_id)
        .bind(&entry_number)
        .bind(payment.payment_date)
        .bind(&description)
        .bind(&payment.payment_number)
        .bind(&lines_hash)
        .bind(Uuid::nil()) // System-generated
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        // Insert debit line
        sqlx::query(
            r#"
            INSERT INTO journal_lines (
                id, entry_id, line_number, account_id,
                debit_amount, credit_amount, description
            )
            VALUES ($1, $2, 1, $3, $4, NULL, $5)
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(entry_id)
        .bind(debit_id)
        .bind(payment.amount)
        .bind(&description)
        .execute(&mut *tx)
        .await?;

        // Insert credit line
        sqlx::query(
            r#"
            INSERT INTO journal_lines (
                id, entry_id, line_number, account_id,
                debit_amount, credit_amount, description
            )
            VALUES ($1, $2, 2, $3, NULL, $4, $5)
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(entry_id)
        .bind(credit_id)
        .bind(payment.amount)
        .bind(&description)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok(Some(entry_id))
    }

    /// Get a payment by ID
    pub async fn get_payment(&self, id: Uuid) -> Result<Option<Payment>, sqlx::Error> {
        let payment = sqlx::query_as::<_, Payment>(
            "SELECT * FROM payments WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(payment)
    }

    /// List all payments
    pub async fn list_payments(&self) -> Result<Vec<Payment>, sqlx::Error> {
        let payments = sqlx::query_as::<_, Payment>(
            "SELECT * FROM payments ORDER BY payment_date DESC, payment_number DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(payments)
    }

    /// Get payments for a specific invoice
    pub async fn get_invoice_payments(&self, invoice_id: Uuid) -> Result<Vec<Payment>, sqlx::Error> {
        let payments = sqlx::query_as::<_, Payment>(
            "SELECT * FROM payments WHERE invoice_id = $1 ORDER BY payment_date"
        )
        .bind(invoice_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(payments)
    }

    /// Get payments for a specific bill
    pub async fn get_bill_payments(&self, bill_id: Uuid) -> Result<Vec<Payment>, sqlx::Error> {
        let payments = sqlx::query_as::<_, Payment>(
            "SELECT * FROM payments WHERE bill_id = $1 ORDER BY payment_date"
        )
        .bind(bill_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(payments)
    }

    /// Update invoice status based on payment totals
    async fn update_invoice_status(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        invoice_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get invoice total
        let invoice_total: rust_decimal::Decimal = sqlx::query_scalar(
            "SELECT total_amount FROM invoices WHERE id = $1"
        )
        .bind(invoice_id)
        .fetch_one(&mut **tx)
        .await?;

        // Get sum of all payments for this invoice
        let paid_amount: Option<rust_decimal::Decimal> = sqlx::query_scalar(
            "SELECT SUM(amount) FROM payments WHERE invoice_id = $1"
        )
        .bind(invoice_id)
        .fetch_one(&mut **tx)
        .await?;

        let paid = paid_amount.unwrap_or(rust_decimal::Decimal::ZERO);

        // Determine new status
        let new_status = if paid >= invoice_total {
            InvoiceStatus::Paid
        } else if paid > rust_decimal::Decimal::ZERO {
            InvoiceStatus::PartiallyPaid
        } else {
            InvoiceStatus::Sent // Keep as Sent if no payments
        };

        // Update invoice status
        sqlx::query(
            "UPDATE invoices SET status = $1, updated_at = $2 WHERE id = $3"
        )
        .bind(&new_status)
        .bind(Utc::now())
        .bind(invoice_id)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    /// Update bill status based on payment totals
    async fn update_bill_status(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        bill_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get bill total
        let bill_total: rust_decimal::Decimal = sqlx::query_scalar(
            "SELECT total_amount FROM bills WHERE id = $1"
        )
        .bind(bill_id)
        .fetch_one(&mut **tx)
        .await?;

        // Get sum of all payments for this bill
        let paid_amount: Option<rust_decimal::Decimal> = sqlx::query_scalar(
            "SELECT SUM(amount) FROM payments WHERE bill_id = $1"
        )
        .bind(bill_id)
        .fetch_one(&mut **tx)
        .await?;

        let paid = paid_amount.unwrap_or(rust_decimal::Decimal::ZERO);

        // Determine new status
        let new_status = if paid >= bill_total {
            BillStatus::Paid
        } else if paid > rust_decimal::Decimal::ZERO {
            BillStatus::PartiallyPaid
        } else {
            BillStatus::Approved // Keep as Approved if no payments
        };

        // Update bill status
        sqlx::query(
            "UPDATE bills SET status = $1, updated_at = $2 WHERE id = $3"
        )
        .bind(&new_status)
        .bind(Utc::now())
        .bind(bill_id)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }
}
