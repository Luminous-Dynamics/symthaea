// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Type definitions for Mycelix ERP Finance Module
 */

// Account Types
export type AccountType =
  | 'ASSET'
  | 'LIABILITY'
  | 'EQUITY'
  | 'REVENUE'
  | 'EXPENSE';

export type AccountSubtype =
  | 'CURRENT_ASSET'
  | 'FIXED_ASSET'
  | 'CURRENT_LIABILITY'
  | 'LONG_TERM_LIABILITY'
  | 'RETAINED_EARNINGS'
  | 'COMMON_STOCK'
  | 'OPERATING_REVENUE'
  | 'OTHER_REVENUE'
  | 'OPERATING_EXPENSE'
  | 'OTHER_EXPENSE'
  | 'COGS';

export interface GLAccount {
  id: string;
  code: string;
  name: string;
  account_type: AccountType;
  account_subtype?: AccountSubtype;
  parent_id?: string;
  description?: string;
  is_active: boolean;
  balance: string; // Decimal as string for precision
  currency: string;
  created_at: string;
  updated_at: string;
}

export interface CreateAccountRequest {
  code: string;
  name: string;
  account_type: AccountType;
  account_subtype?: AccountSubtype;
  parent_id?: string;
  description?: string;
  currency?: string;
}

// Journal Entries
export type JournalEntryStatus = 'DRAFT' | 'POSTED' | 'REVERSED';

export interface JournalLine {
  id?: string;
  account_id: string;
  account_code?: string;
  account_name?: string;
  description?: string;
  debit: string;
  credit: string;
  line_hash?: string;
}

export interface JournalEntry {
  id: string;
  entry_number: string;
  entry_date: string;
  description: string;
  reference?: string;
  status: JournalEntryStatus;
  lines: JournalLine[];
  total_debit: string;
  total_credit: string;
  entry_hash: string;
  created_by?: string;
  created_at: string;
  posted_at?: string;
}

export interface CreateJournalEntryRequest {
  entry_date: string;
  description: string;
  reference?: string;
  lines: Omit<JournalLine, 'id' | 'line_hash'>[];
}

// Invoices (Accounts Receivable)
export type InvoiceStatus =
  | 'DRAFT'
  | 'SENT'
  | 'VIEWED'
  | 'PARTIAL'
  | 'PAID'
  | 'OVERDUE'
  | 'VOID';

export interface InvoiceLine {
  id?: string;
  description: string;
  quantity: string;
  unit_price: string;
  amount: string;
  tax_rate?: string;
  tax_amount?: string;
  account_id?: string;
}

export interface Invoice {
  id: string;
  invoice_number: string;
  customer_id: string;
  customer_name?: string;
  issue_date: string;
  due_date: string;
  status: InvoiceStatus;
  lines: InvoiceLine[];
  subtotal: string;
  tax_total: string;
  total: string;
  amount_paid: string;
  amount_due: string;
  currency: string;
  notes?: string;
  terms?: string;
  invoice_hash: string;
  created_at: string;
  updated_at: string;
  sent_at?: string;
  paid_at?: string;
}

export interface CreateInvoiceRequest {
  customer_id: string;
  issue_date?: string;
  due_date: string;
  lines: Omit<InvoiceLine, 'id' | 'amount'>[];
  currency?: string;
  notes?: string;
  terms?: string;
}

// Bills (Accounts Payable)
export type BillStatus =
  | 'DRAFT'
  | 'PENDING_APPROVAL'
  | 'APPROVED'
  | 'PARTIAL'
  | 'PAID'
  | 'VOID';

export interface BillLine {
  id?: string;
  description: string;
  quantity: string;
  unit_price: string;
  amount: string;
  tax_rate?: string;
  tax_amount?: string;
  account_id?: string;
}

export interface Bill {
  id: string;
  bill_number: string;
  vendor_id: string;
  vendor_name?: string;
  bill_date: string;
  due_date: string;
  status: BillStatus;
  lines: BillLine[];
  subtotal: string;
  tax_total: string;
  total: string;
  amount_paid: string;
  amount_due: string;
  currency: string;
  notes?: string;
  reference?: string;
  bill_hash: string;
  created_at: string;
  updated_at: string;
  approved_at?: string;
  approved_by?: string;
  paid_at?: string;
}

export interface CreateBillRequest {
  vendor_id: string;
  bill_number?: string;
  bill_date?: string;
  due_date: string;
  lines: Omit<BillLine, 'id' | 'amount'>[];
  currency?: string;
  notes?: string;
  reference?: string;
}

// Payments
export type PaymentType = 'RECEIVED' | 'SENT';
export type PaymentMethod =
  | 'CASH'
  | 'CHECK'
  | 'BANK_TRANSFER'
  | 'CREDIT_CARD'
  | 'ACH'
  | 'WIRE'
  | 'OTHER';

export interface PaymentAllocation {
  document_id: string;
  document_type: 'INVOICE' | 'BILL';
  amount: string;
}

export interface Payment {
  id: string;
  payment_number: string;
  payment_type: PaymentType;
  payment_date: string;
  amount: string;
  currency: string;
  payment_method: PaymentMethod;
  reference?: string;
  notes?: string;
  allocations: PaymentAllocation[];
  payment_hash: string;
  created_at: string;
}

export interface CreatePaymentRequest {
  payment_type: PaymentType;
  payment_date?: string;
  amount: string;
  currency?: string;
  payment_method: PaymentMethod;
  reference?: string;
  notes?: string;
  allocations: PaymentAllocation[];
}

// Financial Reports
export interface TrialBalanceRow {
  account_id: string;
  account_code: string;
  account_name: string;
  account_type: AccountType;
  debit: string;
  credit: string;
  balance: string;
}

export interface TrialBalanceReport {
  as_of_date: string;
  rows: TrialBalanceRow[];
  total_debits: string;
  total_credits: string;
  is_balanced: boolean;
  generated_at: string;
}

export interface IncomeStatementRow {
  account_id: string;
  account_code: string;
  account_name: string;
  amount: string;
}

export interface IncomeStatementReport {
  start_date: string;
  end_date: string;
  revenue: IncomeStatementRow[];
  total_revenue: string;
  expenses: IncomeStatementRow[];
  total_expenses: string;
  cogs: IncomeStatementRow[];
  total_cogs: string;
  gross_profit: string;
  operating_income: string;
  net_income: string;
  generated_at: string;
}

export interface BalanceSheetSection {
  rows: TrialBalanceRow[];
  total: string;
}

export interface BalanceSheetReport {
  as_of_date: string;
  assets: {
    current: BalanceSheetSection;
    fixed: BalanceSheetSection;
    total: string;
  };
  liabilities: {
    current: BalanceSheetSection;
    long_term: BalanceSheetSection;
    total: string;
  };
  equity: BalanceSheetSection;
  total_liabilities_and_equity: string;
  is_balanced: boolean;
  generated_at: string;
}

export interface AgingBucket {
  range: string;
  amount: string;
  count: number;
}

export interface AgingRow {
  entity_id: string;
  entity_name: string;
  current: string;
  days_1_30: string;
  days_31_60: string;
  days_61_90: string;
  over_90: string;
  total: string;
}

export interface AgingReport {
  report_type: 'AR' | 'AP';
  as_of_date: string;
  buckets: AgingBucket[];
  rows: AgingRow[];
  total_current: string;
  total_1_30: string;
  total_31_60: string;
  total_61_90: string;
  total_over_90: string;
  grand_total: string;
  generated_at: string;
}

// Customers & Vendors
export interface Customer {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  billing_address?: Address;
  shipping_address?: Address;
  tax_id?: string;
  payment_terms?: number;
  credit_limit?: string;
  balance: string;
  currency: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Vendor {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  address?: Address;
  tax_id?: string;
  payment_terms?: number;
  balance: string;
  currency: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Address {
  line1: string;
  line2?: string;
  city: string;
  state?: string;
  postal_code: string;
  country: string;
}

// List responses
export interface ListResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface ListFilters {
  limit?: number;
  offset?: number;
  status?: string;
  from_date?: string;
  to_date?: string;
}
