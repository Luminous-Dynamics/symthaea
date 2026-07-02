// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AI Invoice Processing Module
//!
//! Provides OCR, field extraction, and intelligent categorization for invoices.
//! Supports learning from user corrections to improve accuracy over time.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

/// Processing queue item
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ProcessingQueueItem {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub source_type: String,
    pub file_name: Option<String>,
    pub file_path: Option<String>,
    pub file_size: Option<i32>,
    pub mime_type: Option<String>,
    pub status: String,
    pub processing_started_at: Option<DateTime<Utc>>,
    pub processing_completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub created_invoice_id: Option<Uuid>,
    pub created_bill_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Processing status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessingStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Review,
}

impl ProcessingStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessingStatus::Pending => "PENDING",
            ProcessingStatus::Processing => "PROCESSING",
            ProcessingStatus::Completed => "COMPLETED",
            ProcessingStatus::Failed => "FAILED",
            ProcessingStatus::Review => "REVIEW",
        }
    }
}

/// Extracted invoice data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedInvoiceData {
    pub id: Uuid,
    pub queue_id: Uuid,
    pub document_type: Option<DocumentType>,
    pub confidence: f64,

    // Header fields
    pub vendor_name: Option<String>,
    pub vendor_address: Option<String>,
    pub vendor_tax_id: Option<String>,
    pub invoice_number: Option<String>,
    pub invoice_date: Option<NaiveDate>,
    pub due_date: Option<NaiveDate>,

    // Amounts
    pub subtotal: Option<Decimal>,
    pub tax_amount: Option<Decimal>,
    pub total_amount: Option<Decimal>,
    pub currency: String,

    // Line items
    pub line_items: Vec<ExtractedLineItem>,

    // Suggestions
    pub suggested_vendor_id: Option<Uuid>,
    pub suggested_gl_account_id: Option<Uuid>,
    pub suggested_category: Option<String>,
}

/// Document type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    Invoice,
    Bill,
    Receipt,
    CreditNote,
}

impl DocumentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentType::Invoice => "INVOICE",
            DocumentType::Bill => "BILL",
            DocumentType::Receipt => "RECEIPT",
            DocumentType::CreditNote => "CREDIT_NOTE",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "INVOICE" => Some(DocumentType::Invoice),
            "BILL" => Some(DocumentType::Bill),
            "RECEIPT" => Some(DocumentType::Receipt),
            "CREDIT_NOTE" => Some(DocumentType::CreditNote),
            _ => None,
        }
    }
}

/// Extracted line item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLineItem {
    pub line_number: i32,
    pub description: Option<String>,
    pub quantity: Option<Decimal>,
    pub unit_price: Option<Decimal>,
    pub amount: Option<Decimal>,
    pub tax_rate: Option<Decimal>,
    pub suggested_item_id: Option<Uuid>,
    pub suggested_gl_account_id: Option<Uuid>,
    pub confidence: f64,
}

/// OCR result from external service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub raw_text: String,
    pub blocks: Vec<TextBlock>,
    pub confidence: f64,
}

/// Text block from OCR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
    pub confidence: f64,
    pub bounding_box: Option<BoundingBox>,
    pub block_type: Option<String>,
}

/// Bounding box for text location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Field extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub extract_line_items: bool,
    pub extract_tax_details: bool,
    pub match_vendors: bool,
    pub suggest_gl_accounts: bool,
    pub min_confidence: f64,
}

/// Result of approving extracted data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResult {
    pub queue_id: Uuid,
    pub created_type: String,
    pub created_id: Uuid,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            extract_line_items: true,
            extract_tax_details: true,
            match_vendors: true,
            suggest_gl_accounts: true,
            min_confidence: 0.7,
        }
    }
}

// ============================================================================
// Service
// ============================================================================

/// AI Invoice Processing Service
#[derive(Clone)]
pub struct AiInvoiceService {
    pool: PgPool,
    ocr_endpoint: Option<String>,
}

impl AiInvoiceService {
    pub fn new(pool: PgPool) -> Self {
        let ocr_endpoint = std::env::var("OCR_SERVICE_URL").ok();
        Self { pool, ocr_endpoint }
    }

    /// Queue a document for processing
    pub async fn queue_document(
        &self,
        tenant_id: Uuid,
        source_type: &str,
        file_name: Option<&str>,
        file_path: Option<&str>,
        file_size: Option<i32>,
        mime_type: Option<&str>,
    ) -> Result<ProcessingQueueItem, AiProcessingError> {
        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO fin_invoice_processing_queue
             (id, tenant_id, source_type, file_name, file_path, file_size, mime_type)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(source_type)
        .bind(file_name)
        .bind(file_path)
        .bind(file_size)
        .bind(mime_type)
        .execute(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        self.get_queue_item(id).await
    }

    /// Get a queue item
    pub async fn get_queue_item(&self, id: Uuid) -> Result<ProcessingQueueItem, AiProcessingError> {
        sqlx::query_as::<_, ProcessingQueueItem>(
            "SELECT id, tenant_id, source_type, file_name, file_path, file_size,
                    mime_type, status, processing_started_at, processing_completed_at,
                    error_message, created_invoice_id, created_bill_id, created_at
             FROM fin_invoice_processing_queue
             WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?
        .ok_or_else(|| AiProcessingError::NotFound("Queue item not found".into()))
    }

    /// Process a queued document
    pub async fn process_document(
        &self,
        queue_id: Uuid,
        config: ExtractionConfig,
    ) -> Result<ExtractedInvoiceData, AiProcessingError> {
        // Update status to processing
        self.update_status(queue_id, ProcessingStatus::Processing, None)
            .await?;

        let queue_item = self.get_queue_item(queue_id).await?;

        // Perform OCR (simulated if no external service)
        let ocr_result = match (&self.ocr_endpoint, &queue_item.file_path) {
            (Some(_endpoint), Some(_path)) => {
                // TODO: Call external OCR service
                self.simulate_ocr().await?
            }
            _ => self.simulate_ocr().await?,
        };

        // Extract fields from OCR result
        let mut extracted = self.extract_fields(&ocr_result, &config).await?;
        extracted.queue_id = queue_id;

        // Match vendor if configured
        if config.match_vendors {
            if let Some(vendor_name) = &extracted.vendor_name {
                extracted.suggested_vendor_id =
                    self.match_vendor(queue_item.tenant_id, vendor_name).await?;
            }
        }

        // Suggest GL account if configured
        if config.suggest_gl_accounts {
            let keywords: Vec<&str> = extracted
                .line_items
                .iter()
                .filter_map(|li| li.description.as_deref())
                .collect();
            extracted.suggested_gl_account_id =
                self.suggest_gl_account(queue_item.tenant_id, &keywords).await?;
        }

        // Save extracted data
        self.save_extracted_data(queue_item.tenant_id, &extracted)
            .await?;

        // Update status
        let status = if extracted.confidence >= config.min_confidence {
            ProcessingStatus::Completed
        } else {
            ProcessingStatus::Review
        };
        self.update_status(queue_id, status, None).await?;

        Ok(extracted)
    }

    /// Simulate OCR (for testing without external service)
    async fn simulate_ocr(&self) -> Result<OcrResult, AiProcessingError> {
        Ok(OcrResult {
            raw_text: "INVOICE\nInvoice #: INV-2024-001\nDate: 2024-01-15\nDue Date: 2024-02-15\n\nFrom: Acme Corp\n123 Business St\nNew York, NY 10001\n\nTo: Customer Inc\n\nDescription                  Qty    Unit Price    Amount\n-------------------------------------------------\nProfessional Services        10     $150.00       $1,500.00\nConsulting                   5      $200.00       $1,000.00\n\nSubtotal: $2,500.00\nTax (8%): $200.00\nTotal: $2,700.00\n\nPayment Terms: Net 30".to_string(),
            blocks: vec![
                TextBlock {
                    text: "INVOICE".to_string(),
                    confidence: 0.99,
                    bounding_box: None,
                    block_type: Some("HEADER".to_string()),
                },
                TextBlock {
                    text: "INV-2024-001".to_string(),
                    confidence: 0.95,
                    bounding_box: None,
                    block_type: Some("INVOICE_NUMBER".to_string()),
                },
            ],
            confidence: 0.92,
        })
    }

    /// Extract fields from OCR result
    async fn extract_fields(
        &self,
        ocr: &OcrResult,
        _config: &ExtractionConfig,
    ) -> Result<ExtractedInvoiceData, AiProcessingError> {
        let text = &ocr.raw_text;

        // Simple pattern-based extraction (in production, use ML models)
        let invoice_number = self.extract_invoice_number(text);
        let invoice_date = self.extract_date(text, "Date:");
        let due_date = self.extract_date(text, "Due Date:");
        let vendor_name = self.extract_vendor_name(text);
        let total_amount = self.extract_amount(text, "Total:");
        let subtotal = self.extract_amount(text, "Subtotal:");
        let tax_amount = self.extract_amount(text, "Tax");

        // Extract line items
        let line_items = self.extract_line_items(text);

        // Determine document type
        let document_type = if text.to_lowercase().contains("credit")
            || text.to_lowercase().contains("refund")
        {
            Some(DocumentType::CreditNote)
        } else if text.to_lowercase().contains("receipt") {
            Some(DocumentType::Receipt)
        } else if text.to_lowercase().contains("bill") {
            Some(DocumentType::Bill)
        } else {
            Some(DocumentType::Invoice)
        };

        Ok(ExtractedInvoiceData {
            id: Uuid::new_v4(),
            queue_id: Uuid::nil(), // Will be set by caller
            document_type,
            confidence: ocr.confidence,
            vendor_name,
            vendor_address: None,
            vendor_tax_id: None,
            invoice_number,
            invoice_date,
            due_date,
            subtotal,
            tax_amount,
            total_amount,
            currency: "USD".to_string(),
            line_items,
            suggested_vendor_id: None,
            suggested_gl_account_id: None,
            suggested_category: None,
        })
    }

    /// Extract invoice number
    fn extract_invoice_number(&self, text: &str) -> Option<String> {
        // Look for patterns like "Invoice #: XXX" or "Inv #XXX"
        for line in text.lines() {
            let lower = line.to_lowercase();
            if lower.contains("invoice") && (lower.contains('#') || lower.contains("number")) {
                // Extract the number part
                if let Some(pos) = line.find('#') {
                    let after = &line[pos + 1..];
                    let number: String = after
                        .chars()
                        .skip_while(|c| c.is_whitespace() || *c == ':')
                        .take_while(|c| !c.is_whitespace())
                        .collect();
                    if !number.is_empty() {
                        return Some(number);
                    }
                }
            }
        }
        None
    }

    /// Extract a date after a label
    fn extract_date(&self, text: &str, label: &str) -> Option<NaiveDate> {
        for line in text.lines() {
            if line.contains(label) {
                // Try to find a date pattern
                let parts: Vec<&str> = line.split(label).collect();
                if parts.len() > 1 {
                    let date_str = parts[1].trim();
                    // Try common formats
                    if let Ok(d) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                        return Some(d);
                    }
                    if let Ok(d) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                        return Some(d);
                    }
                    if let Ok(d) = NaiveDate::parse_from_str(date_str, "%d/%m/%Y") {
                        return Some(d);
                    }
                }
            }
        }
        None
    }

    /// Extract vendor name
    fn extract_vendor_name(&self, text: &str) -> Option<String> {
        // Look for "From:" section
        for (i, line) in text.lines().enumerate() {
            if line.to_lowercase().contains("from:") {
                // Next non-empty line is likely the vendor name
                let remaining: Vec<&str> = text.lines().skip(i + 1).collect();
                for next_line in remaining {
                    let trimmed = next_line.trim();
                    if !trimmed.is_empty() && !trimmed.contains(':') {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }
        None
    }

    /// Extract an amount after a label
    fn extract_amount(&self, text: &str, label: &str) -> Option<Decimal> {
        for line in text.lines() {
            if line.contains(label) {
                // Find dollar amount
                let cleaned = line.replace('$', "").replace(',', "");
                for word in cleaned.split_whitespace() {
                    if let Ok(amount) = word.parse::<Decimal>() {
                        if amount > Decimal::ZERO {
                            return Some(amount);
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract line items (simplified)
    fn extract_line_items(&self, text: &str) -> Vec<ExtractedLineItem> {
        let mut items = Vec::new();
        let mut in_items_section = false;
        let mut line_num = 0;

        for line in text.lines() {
            let lower = line.to_lowercase();

            // Detect start of items section
            if lower.contains("description") && lower.contains("qty") {
                in_items_section = true;
                continue;
            }

            // Detect end of items section
            if in_items_section
                && (lower.contains("subtotal") || lower.contains("total") || line.trim().is_empty())
            {
                in_items_section = false;
                continue;
            }

            // Parse line item
            if in_items_section && !line.trim().is_empty() && !line.contains("---") {
                line_num += 1;

                // Try to parse the line
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    // Last element is likely amount
                    let amount_str = parts.last().unwrap().replace('$', "").replace(',', "");
                    let amount = amount_str.parse::<Decimal>().ok();

                    // Try to find quantity and unit price
                    let mut qty = None;
                    let mut unit_price = None;
                    if parts.len() >= 4 {
                        let qty_str = parts[parts.len() - 3];
                        qty = qty_str.parse::<Decimal>().ok();
                        let price_str = parts[parts.len() - 2].replace('$', "").replace(',', "");
                        unit_price = price_str.parse::<Decimal>().ok();
                    }

                    // Description is everything before the numbers
                    let desc: Vec<&str> = parts
                        .iter()
                        .take(parts.len().saturating_sub(3))
                        .cloned()
                        .collect();

                    items.push(ExtractedLineItem {
                        line_number: line_num,
                        description: Some(desc.join(" ")),
                        quantity: qty,
                        unit_price,
                        amount,
                        tax_rate: None,
                        suggested_item_id: None,
                        suggested_gl_account_id: None,
                        confidence: 0.8,
                    });
                }
            }
        }

        items
    }

    /// Match vendor by name patterns
    async fn match_vendor(
        &self,
        tenant_id: Uuid,
        vendor_name: &str,
    ) -> Result<Option<Uuid>, AiProcessingError> {
        // Check exact match patterns
        let result: Option<(Uuid,)> = sqlx::query_as(
            "SELECT vendor_id FROM fin_vendor_patterns
             WHERE tenant_id = $1
               AND pattern_type = 'NAME'
               AND is_active = true
               AND LOWER(pattern_value) = LOWER($2)
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(vendor_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        if let Some((vendor_id,)) = result {
            // Update match count
            sqlx::query(
                "UPDATE fin_vendor_patterns
                 SET match_count = match_count + 1, last_matched_at = NOW()
                 WHERE tenant_id = $1 AND vendor_id = $2 AND pattern_type = 'NAME'",
            )
            .bind(tenant_id)
            .bind(vendor_id)
            .execute(&self.pool)
            .await
            .ok();

            return Ok(Some(vendor_id));
        }

        // Try partial match
        let result: Option<(Uuid,)> = sqlx::query_as(
            "SELECT vendor_id FROM fin_vendor_patterns
             WHERE tenant_id = $1
               AND pattern_type = 'NAME'
               AND is_active = true
               AND LOWER($2) LIKE '%' || LOWER(pattern_value) || '%'
             ORDER BY LENGTH(pattern_value) DESC
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(vendor_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        Ok(result.map(|(id,)| id))
    }

    /// Suggest GL account based on keywords
    async fn suggest_gl_account(
        &self,
        tenant_id: Uuid,
        keywords: &[&str],
    ) -> Result<Option<Uuid>, AiProcessingError> {
        if keywords.is_empty() {
            return Ok(None);
        }

        // Score GL accounts by matching keywords
        let keyword_list = keywords.join(" ").to_lowercase();

        let result: Option<(Uuid,)> = sqlx::query_as(
            "SELECT gl_account_id
             FROM fin_gl_account_patterns
             WHERE tenant_id = $1 AND is_active = true
               AND LOWER($2) LIKE '%' || LOWER(keyword) || '%'
             GROUP BY gl_account_id
             ORDER BY SUM(weight) DESC
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(&keyword_list)
        .fetch_optional(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        Ok(result.map(|(id,)| id))
    }

    /// Save extracted data
    async fn save_extracted_data(
        &self,
        tenant_id: Uuid,
        data: &ExtractedInvoiceData,
    ) -> Result<(), AiProcessingError> {
        sqlx::query(
            "INSERT INTO fin_extracted_invoice_data
             (id, queue_id, tenant_id, document_type, confidence,
              vendor_name, vendor_address, vendor_tax_id,
              invoice_number, invoice_date, due_date,
              subtotal, tax_amount, total_amount, currency,
              suggested_vendor_id, suggested_gl_account_id, suggested_category)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)",
        )
        .bind(data.id)
        .bind(data.queue_id)
        .bind(tenant_id)
        .bind(data.document_type.as_ref().map(|dt| dt.as_str()))
        .bind(Decimal::try_from(data.confidence).unwrap_or_default())
        .bind(&data.vendor_name)
        .bind(&data.vendor_address)
        .bind(&data.vendor_tax_id)
        .bind(&data.invoice_number)
        .bind(data.invoice_date)
        .bind(data.due_date)
        .bind(data.subtotal)
        .bind(data.tax_amount)
        .bind(data.total_amount)
        .bind(&data.currency)
        .bind(data.suggested_vendor_id)
        .bind(data.suggested_gl_account_id)
        .bind(&data.suggested_category)
        .execute(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        // Save line items
        for item in &data.line_items {
            sqlx::query(
                "INSERT INTO fin_extracted_line_items
                 (id, extracted_data_id, line_number, description, quantity,
                  unit_price, amount, tax_rate, suggested_item_id, suggested_gl_account_id, confidence)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
            )
            .bind(Uuid::new_v4())
            .bind(data.id)
            .bind(item.line_number)
            .bind(&item.description)
            .bind(item.quantity)
            .bind(item.unit_price)
            .bind(item.amount)
            .bind(item.tax_rate)
            .bind(item.suggested_item_id)
            .bind(item.suggested_gl_account_id)
            .bind(Decimal::try_from(item.confidence).unwrap_or_default())
            .execute(&self.pool)
            .await
            .map_err(AiProcessingError::Database)?;
        }

        Ok(())
    }

    /// Update processing status
    async fn update_status(
        &self,
        queue_id: Uuid,
        status: ProcessingStatus,
        error: Option<&str>,
    ) -> Result<(), AiProcessingError> {
        let now = Utc::now();

        match status {
            ProcessingStatus::Processing => {
                sqlx::query(
                    "UPDATE fin_invoice_processing_queue
                     SET status = $2, processing_started_at = $3, updated_at = NOW()
                     WHERE id = $1",
                )
                .bind(queue_id)
                .bind(status.as_str())
                .bind(now)
                .execute(&self.pool)
                .await
                .map_err(AiProcessingError::Database)?;
            }
            ProcessingStatus::Completed | ProcessingStatus::Review => {
                sqlx::query(
                    "UPDATE fin_invoice_processing_queue
                     SET status = $2, processing_completed_at = $3, updated_at = NOW()
                     WHERE id = $1",
                )
                .bind(queue_id)
                .bind(status.as_str())
                .bind(now)
                .execute(&self.pool)
                .await
                .map_err(AiProcessingError::Database)?;
            }
            ProcessingStatus::Failed => {
                sqlx::query(
                    "UPDATE fin_invoice_processing_queue
                     SET status = $2, processing_completed_at = $3, error_message = $4, updated_at = NOW()
                     WHERE id = $1",
                )
                .bind(queue_id)
                .bind(status.as_str())
                .bind(now)
                .bind(error)
                .execute(&self.pool)
                .await
                .map_err(AiProcessingError::Database)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Record a training sample from user correction
    pub async fn record_correction(
        &self,
        tenant_id: Uuid,
        queue_id: Uuid,
        sample_type: &str,
        input_text: &str,
        expected_output: &str,
        user_id: Uuid,
    ) -> Result<(), AiProcessingError> {
        sqlx::query(
            "INSERT INTO fin_ai_training_samples
             (id, tenant_id, sample_type, input_text, expected_output, source_queue_id, corrected_by)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(Uuid::new_v4())
        .bind(tenant_id)
        .bind(sample_type)
        .bind(input_text)
        .bind(expected_output)
        .bind(queue_id)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        // Mark queue item as having corrections
        sqlx::query(
            "UPDATE fin_invoice_processing_queue
             SET corrections_made = true, updated_at = NOW()
             WHERE id = $1",
        )
        .bind(queue_id)
        .execute(&self.pool)
        .await
        .ok();

        Ok(())
    }

    /// Add a vendor pattern for future matching
    pub async fn add_vendor_pattern(
        &self,
        tenant_id: Uuid,
        vendor_id: Uuid,
        pattern_type: &str,
        pattern_value: &str,
    ) -> Result<(), AiProcessingError> {
        sqlx::query(
            "INSERT INTO fin_vendor_patterns
             (id, tenant_id, vendor_id, pattern_type, pattern_value)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (tenant_id, vendor_id, pattern_type, pattern_value) DO NOTHING",
        )
        .bind(Uuid::new_v4())
        .bind(tenant_id)
        .bind(vendor_id)
        .bind(pattern_type)
        .bind(pattern_value)
        .execute(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        Ok(())
    }

    /// Get pending items in queue
    pub async fn get_pending_items(
        &self,
        tenant_id: Uuid,
        limit: i32,
    ) -> Result<Vec<ProcessingQueueItem>, AiProcessingError> {
        let items = sqlx::query_as::<_, ProcessingQueueItem>(
            "SELECT id, tenant_id, source_type, file_name, file_path, file_size,
                    mime_type, status, processing_started_at, processing_completed_at,
                    error_message, created_invoice_id, created_bill_id, created_at
             FROM fin_invoice_processing_queue
             WHERE tenant_id = $1 AND status = 'PENDING'
             ORDER BY created_at
             LIMIT $2",
        )
        .bind(tenant_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        Ok(items)
    }

    /// Get extracted data for a queue item
    pub async fn get_extracted_data(
        &self,
        queue_id: Uuid,
    ) -> Result<Option<ExtractedInvoiceData>, AiProcessingError> {
        // Query the extracted invoice data
        let result: Option<(
            Uuid,           // id
            Uuid,           // queue_id
            Option<String>, // document_type
            Decimal,        // confidence
            Option<String>, // vendor_name
            Option<String>, // vendor_address
            Option<String>, // vendor_tax_id
            Option<String>, // invoice_number
            Option<NaiveDate>, // invoice_date
            Option<NaiveDate>, // due_date
            Option<Decimal>, // subtotal
            Option<Decimal>, // tax_amount
            Option<Decimal>, // total_amount
            String,         // currency
            Option<Uuid>,   // suggested_vendor_id
            Option<Uuid>,   // suggested_gl_account_id
            Option<String>, // suggested_category
        )> = sqlx::query_as(
            "SELECT id, queue_id, document_type, confidence,
                    vendor_name, vendor_address, vendor_tax_id,
                    invoice_number, invoice_date, due_date,
                    subtotal, tax_amount, total_amount, currency,
                    suggested_vendor_id, suggested_gl_account_id, suggested_category
             FROM fin_extracted_invoice_data
             WHERE queue_id = $1",
        )
        .bind(queue_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        match result {
            Some((
                id, queue_id, document_type, confidence,
                vendor_name, vendor_address, vendor_tax_id,
                invoice_number, invoice_date, due_date,
                subtotal, tax_amount, total_amount, currency,
                suggested_vendor_id, suggested_gl_account_id, suggested_category,
            )) => {
                // Get line items
                let line_items = self.get_extracted_line_items(id).await?;

                Ok(Some(ExtractedInvoiceData {
                    id,
                    queue_id,
                    document_type: document_type.and_then(|dt| DocumentType::from_str(&dt)),
                    confidence: confidence.try_into().unwrap_or(0.0),
                    vendor_name,
                    vendor_address,
                    vendor_tax_id,
                    invoice_number,
                    invoice_date,
                    due_date,
                    subtotal,
                    tax_amount,
                    total_amount,
                    currency,
                    line_items,
                    suggested_vendor_id,
                    suggested_gl_account_id,
                    suggested_category,
                }))
            }
            None => Ok(None),
        }
    }

    /// Get extracted line items for an extraction
    async fn get_extracted_line_items(
        &self,
        extracted_data_id: Uuid,
    ) -> Result<Vec<ExtractedLineItem>, AiProcessingError> {
        let items: Vec<(
            i32,            // line_number
            Option<String>, // description
            Option<Decimal>, // quantity
            Option<Decimal>, // unit_price
            Option<Decimal>, // amount
            Option<Decimal>, // tax_rate
            Option<Uuid>,   // suggested_item_id
            Option<Uuid>,   // suggested_gl_account_id
            Decimal,        // confidence
        )> = sqlx::query_as(
            "SELECT line_number, description, quantity, unit_price, amount,
                    tax_rate, suggested_item_id, suggested_gl_account_id, confidence
             FROM fin_extracted_line_items
             WHERE extracted_data_id = $1
             ORDER BY line_number",
        )
        .bind(extracted_data_id)
        .fetch_all(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        Ok(items
            .into_iter()
            .map(|(line_number, description, quantity, unit_price, amount, tax_rate, suggested_item_id, suggested_gl_account_id, confidence)| {
                ExtractedLineItem {
                    line_number,
                    description,
                    quantity,
                    unit_price,
                    amount,
                    tax_rate,
                    suggested_item_id,
                    suggested_gl_account_id,
                    confidence: confidence.try_into().unwrap_or(0.0),
                }
            })
            .collect())
    }

    /// Approve extracted data and create invoice or bill
    pub async fn approve_extraction(
        &self,
        queue_id: Uuid,
        user_id: Uuid,
        create_as: &str,
    ) -> Result<ApprovalResult, AiProcessingError> {
        // Get the extracted data
        let extracted = self.get_extracted_data(queue_id).await?
            .ok_or_else(|| AiProcessingError::NotFound("No extracted data found for queue item".into()))?;

        // Get queue item to retrieve tenant_id
        let queue_item = self.get_queue_item(queue_id).await?;

        match create_as.to_uppercase().as_str() {
            "INVOICE" => {
                let invoice_id = self.create_invoice_from_extraction(&queue_item.tenant_id, &extracted, user_id).await?;

                // Update queue item with created invoice
                sqlx::query(
                    "UPDATE fin_invoice_processing_queue
                     SET status = 'APPROVED', created_invoice_id = $2, approved_by = $3, approved_at = NOW(), updated_at = NOW()
                     WHERE id = $1"
                )
                .bind(queue_id)
                .bind(invoice_id)
                .bind(user_id)
                .execute(&self.pool)
                .await
                .map_err(AiProcessingError::Database)?;

                Ok(ApprovalResult {
                    queue_id,
                    created_type: "INVOICE".to_string(),
                    created_id: invoice_id,
                })
            }
            "BILL" => {
                let bill_id = self.create_bill_from_extraction(&queue_item.tenant_id, &extracted, user_id).await?;

                // Update queue item with created bill
                sqlx::query(
                    "UPDATE fin_invoice_processing_queue
                     SET status = 'APPROVED', created_bill_id = $2, approved_by = $3, approved_at = NOW(), updated_at = NOW()
                     WHERE id = $1"
                )
                .bind(queue_id)
                .bind(bill_id)
                .bind(user_id)
                .execute(&self.pool)
                .await
                .map_err(AiProcessingError::Database)?;

                Ok(ApprovalResult {
                    queue_id,
                    created_type: "BILL".to_string(),
                    created_id: bill_id,
                })
            }
            _ => Err(AiProcessingError::InvalidInput(
                "create_as must be 'INVOICE' or 'BILL'".to_string(),
            )),
        }
    }

    /// Create an invoice from extracted data
    async fn create_invoice_from_extraction(
        &self,
        tenant_id: &Uuid,
        extracted: &ExtractedInvoiceData,
        _user_id: Uuid,
    ) -> Result<Uuid, AiProcessingError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let invoice_number = extracted.invoice_number.clone()
            .unwrap_or_else(|| format!("INV-{}", id.to_string()[..8].to_uppercase()));

        // Use suggested vendor or create a placeholder customer ID
        let customer_id = extracted.suggested_vendor_id.unwrap_or_else(Uuid::new_v4);

        let invoice_date = extracted.invoice_date
            .map(|d| chrono::DateTime::from_naive_utc_and_tz(d.and_hms_opt(0, 0, 0).unwrap(), Utc))
            .unwrap_or(now);

        let due_date = extracted.due_date
            .map(|d| chrono::DateTime::from_naive_utc_and_tz(d.and_hms_opt(0, 0, 0).unwrap(), Utc))
            .unwrap_or(now + chrono::Duration::days(30));

        let subtotal = extracted.subtotal.unwrap_or(Decimal::ZERO);
        let tax_amount = extracted.tax_amount.unwrap_or(Decimal::ZERO);
        let total_amount = extracted.total_amount.unwrap_or(subtotal + tax_amount);

        // Insert invoice
        sqlx::query(
            r#"
            INSERT INTO invoices (
                id, tenant_id, invoice_number, customer_id, invoice_date, due_date,
                currency, subtotal, tax_amount, total_amount, status,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'DRAFT', $11, $12)
            "#,
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&invoice_number)
        .bind(customer_id)
        .bind(invoice_date)
        .bind(due_date)
        .bind(&extracted.currency)
        .bind(subtotal)
        .bind(tax_amount)
        .bind(total_amount)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        // Insert invoice lines
        for item in &extracted.line_items {
            let line_total = item.amount.unwrap_or(
                item.quantity.unwrap_or(Decimal::ONE) * item.unit_price.unwrap_or(Decimal::ZERO)
            );
            let line_tax = item.tax_rate.map(|rate| line_total * rate);

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
            .bind(item.line_number)
            .bind(&item.description)
            .bind(item.quantity.unwrap_or(Decimal::ONE))
            .bind(item.unit_price.unwrap_or(Decimal::ZERO))
            .bind(line_total)
            .bind(item.tax_rate)
            .bind(line_tax)
            .bind(item.suggested_item_id)
            .execute(&self.pool)
            .await
            .map_err(AiProcessingError::Database)?;
        }

        Ok(id)
    }

    /// Create a bill from extracted data
    async fn create_bill_from_extraction(
        &self,
        tenant_id: &Uuid,
        extracted: &ExtractedInvoiceData,
        _user_id: Uuid,
    ) -> Result<Uuid, AiProcessingError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let bill_number = extracted.invoice_number.clone()
            .unwrap_or_else(|| format!("BILL-{}", id.to_string()[..8].to_uppercase()));

        // Use suggested vendor or create a placeholder vendor ID
        let vendor_id = extracted.suggested_vendor_id.unwrap_or_else(Uuid::new_v4);

        let bill_date = extracted.invoice_date
            .map(|d| chrono::DateTime::from_naive_utc_and_tz(d.and_hms_opt(0, 0, 0).unwrap(), Utc))
            .unwrap_or(now);

        let due_date = extracted.due_date
            .map(|d| chrono::DateTime::from_naive_utc_and_tz(d.and_hms_opt(0, 0, 0).unwrap(), Utc))
            .unwrap_or(now + chrono::Duration::days(30));

        let subtotal = extracted.subtotal.unwrap_or(Decimal::ZERO);
        let tax_amount = extracted.tax_amount.unwrap_or(Decimal::ZERO);
        let total_amount = extracted.total_amount.unwrap_or(subtotal + tax_amount);

        // Insert bill
        sqlx::query(
            r#"
            INSERT INTO bills (
                id, tenant_id, bill_number, vendor_id, bill_date, due_date,
                currency, subtotal, tax_amount, total_amount, status,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'DRAFT', $11, $12)
            "#,
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&bill_number)
        .bind(vendor_id)
        .bind(bill_date)
        .bind(due_date)
        .bind(&extracted.currency)
        .bind(subtotal)
        .bind(tax_amount)
        .bind(total_amount)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(AiProcessingError::Database)?;

        // Insert bill lines
        for item in &extracted.line_items {
            let line_total = item.amount.unwrap_or(
                item.quantity.unwrap_or(Decimal::ONE) * item.unit_price.unwrap_or(Decimal::ZERO)
            );
            let line_tax = item.tax_rate.map(|rate| line_total * rate);

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
            .bind(item.line_number)
            .bind(&item.description)
            .bind(item.quantity.unwrap_or(Decimal::ONE))
            .bind(item.unit_price.unwrap_or(Decimal::ZERO))
            .bind(line_total)
            .bind(item.tax_rate)
            .bind(line_tax)
            .bind(item.suggested_gl_account_id)
            .execute(&self.pool)
            .await
            .map_err(AiProcessingError::Database)?;
        }

        Ok(id)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// AI processing errors
#[derive(Debug, thiserror::Error)]
pub enum AiProcessingError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("OCR error: {0}")]
    OcrError(String),

    #[error("Extraction error: {0}")]
    ExtractionError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
