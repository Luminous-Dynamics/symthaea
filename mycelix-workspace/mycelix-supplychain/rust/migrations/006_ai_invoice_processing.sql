-- AI Invoice Processing Module
-- OCR, field extraction, and intelligent categorization

-- Invoice processing queue
CREATE TABLE IF NOT EXISTS fin_invoice_processing_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    -- Source document
    source_type TEXT NOT NULL,  -- UPLOAD, EMAIL, API
    file_name TEXT,
    file_path TEXT,
    file_size INT,
    mime_type TEXT,

    -- Processing status
    status TEXT DEFAULT 'PENDING',  -- PENDING, PROCESSING, COMPLETED, FAILED, REVIEW
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    error_message TEXT,

    -- Created records
    created_invoice_id UUID REFERENCES fin_invoices(id),
    created_bill_id UUID REFERENCES fin_bills(id),

    -- Review tracking
    reviewed_by UUID REFERENCES auth_users(id),
    reviewed_at TIMESTAMPTZ,
    corrections_made BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Extracted invoice data before review
CREATE TABLE IF NOT EXISTS fin_extracted_invoice_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_id UUID NOT NULL REFERENCES fin_invoice_processing_queue(id),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    -- Document classification
    document_type TEXT,  -- INVOICE, BILL, RECEIPT, CREDIT_NOTE
    confidence DECIMAL(5, 2),  -- Overall extraction confidence

    -- Extracted header fields
    vendor_name TEXT,
    vendor_name_confidence DECIMAL(5, 2),
    vendor_address TEXT,
    vendor_tax_id TEXT,

    invoice_number TEXT,
    invoice_number_confidence DECIMAL(5, 2),

    invoice_date DATE,
    invoice_date_confidence DECIMAL(5, 2),

    due_date DATE,
    due_date_confidence DECIMAL(5, 2),

    -- Extracted amounts
    subtotal DECIMAL(15, 2),
    tax_amount DECIMAL(15, 2),
    total_amount DECIMAL(15, 2),
    total_amount_confidence DECIMAL(5, 2),
    currency TEXT DEFAULT 'USD',

    -- Payment terms
    payment_terms TEXT,

    -- Raw OCR text
    raw_text TEXT,

    -- Suggested categorization
    suggested_vendor_id UUID REFERENCES crm_contacts(id),
    suggested_gl_account_id UUID REFERENCES fin_gl_accounts(id),
    suggested_category TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Extracted line items
CREATE TABLE IF NOT EXISTS fin_extracted_line_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    extracted_data_id UUID NOT NULL REFERENCES fin_extracted_invoice_data(id),

    line_number INT,
    description TEXT,
    quantity DECIMAL(15, 4),
    unit_price DECIMAL(15, 4),
    amount DECIMAL(15, 2),
    tax_rate DECIMAL(5, 2),

    -- AI suggestions
    suggested_item_id UUID REFERENCES inv_items(id),
    suggested_gl_account_id UUID REFERENCES fin_gl_accounts(id),
    confidence DECIMAL(5, 2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- AI training data from corrections
CREATE TABLE IF NOT EXISTS fin_ai_training_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    -- Sample type
    sample_type TEXT NOT NULL,  -- VENDOR_MATCH, CATEGORY, LINE_ITEM, DATE_FORMAT

    -- Input (what the AI saw)
    input_text TEXT NOT NULL,
    input_context JSONB,

    -- Expected output (what was correct)
    expected_output TEXT NOT NULL,
    expected_metadata JSONB,

    -- Correction context
    source_queue_id UUID REFERENCES fin_invoice_processing_queue(id),
    corrected_by UUID REFERENCES auth_users(id),

    -- Training status
    used_in_training BOOLEAN DEFAULT false,
    training_batch_id TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vendor patterns for matching
CREATE TABLE IF NOT EXISTS fin_vendor_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),
    vendor_id UUID NOT NULL REFERENCES crm_contacts(id),

    -- Pattern matching
    pattern_type TEXT NOT NULL,  -- NAME, TAX_ID, EMAIL, ADDRESS
    pattern_value TEXT NOT NULL,
    is_regex BOOLEAN DEFAULT false,

    -- Match statistics
    match_count INT DEFAULT 0,
    last_matched_at TIMESTAMPTZ,

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(tenant_id, vendor_id, pattern_type, pattern_value)
);

-- GL account prediction model
CREATE TABLE IF NOT EXISTS fin_gl_account_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),
    gl_account_id UUID NOT NULL REFERENCES fin_gl_accounts(id),

    -- Pattern matching
    keyword TEXT NOT NULL,
    weight DECIMAL(5, 2) DEFAULT 1.0,

    -- Statistics
    prediction_count INT DEFAULT 0,
    correct_count INT DEFAULT 0,

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_processing_queue_tenant ON fin_invoice_processing_queue(tenant_id);
CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON fin_invoice_processing_queue(status);
CREATE INDEX IF NOT EXISTS idx_extracted_data_queue ON fin_extracted_invoice_data(queue_id);
CREATE INDEX IF NOT EXISTS idx_vendor_patterns_tenant ON fin_vendor_patterns(tenant_id, pattern_type);
CREATE INDEX IF NOT EXISTS idx_gl_patterns_keyword ON fin_gl_account_patterns(tenant_id, keyword);
CREATE INDEX IF NOT EXISTS idx_training_samples_type ON fin_ai_training_samples(tenant_id, sample_type);
