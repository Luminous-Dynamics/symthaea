// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix ERP Finance Module Client
 */

import { AxiosInstance } from 'axios';
import {
  GLAccount,
  CreateAccountRequest,
  JournalEntry,
  CreateJournalEntryRequest,
  Invoice,
  CreateInvoiceRequest,
  Bill,
  CreateBillRequest,
  Payment,
  CreatePaymentRequest,
  TrialBalanceReport,
  IncomeStatementReport,
  BalanceSheetReport,
  AgingReport,
  Customer,
  Vendor,
  ListResponse,
  ListFilters,
} from './finance-types';

export class FinanceClient {
  constructor(private client: AxiosInstance) {}

  // ============ GL Accounts ============

  /**
   * List all GL accounts
   */
  async listAccounts(filters?: ListFilters): Promise<ListResponse<GLAccount>> {
    const response = await this.client.get<ListResponse<GLAccount>>(
      '/v1/fin/accounts',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get a GL account by ID
   */
  async getAccount(accountId: string): Promise<GLAccount> {
    const response = await this.client.get<GLAccount>(
      `/v1/fin/accounts/${accountId}`
    );
    return response.data;
  }

  /**
   * Create a new GL account
   */
  async createAccount(account: CreateAccountRequest): Promise<GLAccount> {
    const response = await this.client.post<GLAccount>(
      '/v1/fin/accounts',
      account
    );
    return response.data;
  }

  /**
   * Update a GL account
   */
  async updateAccount(
    accountId: string,
    updates: Partial<CreateAccountRequest>
  ): Promise<GLAccount> {
    const response = await this.client.patch<GLAccount>(
      `/v1/fin/accounts/${accountId}`,
      updates
    );
    return response.data;
  }

  // ============ Journal Entries ============

  /**
   * List journal entries
   */
  async listJournalEntries(
    filters?: ListFilters
  ): Promise<ListResponse<JournalEntry>> {
    const response = await this.client.get<ListResponse<JournalEntry>>(
      '/v1/fin/journal-entries',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get a journal entry by ID
   */
  async getJournalEntry(entryId: string): Promise<JournalEntry> {
    const response = await this.client.get<JournalEntry>(
      `/v1/fin/journal-entries/${entryId}`
    );
    return response.data;
  }

  /**
   * Create a journal entry (draft)
   */
  async createJournalEntry(
    entry: CreateJournalEntryRequest
  ): Promise<JournalEntry> {
    const response = await this.client.post<JournalEntry>(
      '/v1/fin/journal-entries',
      entry
    );
    return response.data;
  }

  /**
   * Post a journal entry (finalize and record to GL)
   */
  async postJournalEntry(entryId: string): Promise<JournalEntry> {
    const response = await this.client.post<JournalEntry>(
      `/v1/fin/journal-entries/${entryId}/post`
    );
    return response.data;
  }

  /**
   * Reverse a posted journal entry
   */
  async reverseJournalEntry(
    entryId: string,
    reversalDate?: string
  ): Promise<JournalEntry> {
    const response = await this.client.post<JournalEntry>(
      `/v1/fin/journal-entries/${entryId}/reverse`,
      { reversal_date: reversalDate }
    );
    return response.data;
  }

  // ============ Invoices (AR) ============

  /**
   * List invoices
   */
  async listInvoices(filters?: ListFilters): Promise<ListResponse<Invoice>> {
    const response = await this.client.get<ListResponse<Invoice>>(
      '/v1/fin/invoices',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get an invoice by ID
   */
  async getInvoice(invoiceId: string): Promise<Invoice> {
    const response = await this.client.get<Invoice>(
      `/v1/fin/invoices/${invoiceId}`
    );
    return response.data;
  }

  /**
   * Create a new invoice
   */
  async createInvoice(invoice: CreateInvoiceRequest): Promise<Invoice> {
    const response = await this.client.post<Invoice>(
      '/v1/fin/invoices',
      invoice
    );
    return response.data;
  }

  /**
   * Update a draft invoice
   */
  async updateInvoice(
    invoiceId: string,
    updates: Partial<CreateInvoiceRequest>
  ): Promise<Invoice> {
    const response = await this.client.patch<Invoice>(
      `/v1/fin/invoices/${invoiceId}`,
      updates
    );
    return response.data;
  }

  /**
   * Send an invoice (mark as sent)
   */
  async sendInvoice(invoiceId: string): Promise<Invoice> {
    const response = await this.client.post<Invoice>(
      `/v1/fin/invoices/${invoiceId}/send`
    );
    return response.data;
  }

  /**
   * Void an invoice
   */
  async voidInvoice(invoiceId: string, reason?: string): Promise<Invoice> {
    const response = await this.client.post<Invoice>(
      `/v1/fin/invoices/${invoiceId}/void`,
      { reason }
    );
    return response.data;
  }

  // ============ Bills (AP) ============

  /**
   * List bills
   */
  async listBills(filters?: ListFilters): Promise<ListResponse<Bill>> {
    const response = await this.client.get<ListResponse<Bill>>(
      '/v1/fin/bills',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get a bill by ID
   */
  async getBill(billId: string): Promise<Bill> {
    const response = await this.client.get<Bill>(`/v1/fin/bills/${billId}`);
    return response.data;
  }

  /**
   * Create a new bill
   */
  async createBill(bill: CreateBillRequest): Promise<Bill> {
    const response = await this.client.post<Bill>('/v1/fin/bills', bill);
    return response.data;
  }

  /**
   * Approve a bill for payment
   */
  async approveBill(billId: string): Promise<Bill> {
    const response = await this.client.post<Bill>(
      `/v1/fin/bills/${billId}/approve`
    );
    return response.data;
  }

  /**
   * Void a bill
   */
  async voidBill(billId: string, reason?: string): Promise<Bill> {
    const response = await this.client.post<Bill>(
      `/v1/fin/bills/${billId}/void`,
      { reason }
    );
    return response.data;
  }

  // ============ Payments ============

  /**
   * List payments
   */
  async listPayments(filters?: ListFilters): Promise<ListResponse<Payment>> {
    const response = await this.client.get<ListResponse<Payment>>(
      '/v1/fin/payments',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get a payment by ID
   */
  async getPayment(paymentId: string): Promise<Payment> {
    const response = await this.client.get<Payment>(
      `/v1/fin/payments/${paymentId}`
    );
    return response.data;
  }

  /**
   * Record a payment
   */
  async createPayment(payment: CreatePaymentRequest): Promise<Payment> {
    const response = await this.client.post<Payment>(
      '/v1/fin/payments',
      payment
    );
    return response.data;
  }

  /**
   * Helper: Record payment for an invoice
   */
  async payInvoice(
    invoiceId: string,
    amount: string,
    method: CreatePaymentRequest['payment_method'],
    reference?: string
  ): Promise<Payment> {
    return this.createPayment({
      payment_type: 'RECEIVED',
      amount,
      payment_method: method,
      reference,
      allocations: [
        {
          document_id: invoiceId,
          document_type: 'INVOICE',
          amount,
        },
      ],
    });
  }

  /**
   * Helper: Record payment for a bill
   */
  async payBill(
    billId: string,
    amount: string,
    method: CreatePaymentRequest['payment_method'],
    reference?: string
  ): Promise<Payment> {
    return this.createPayment({
      payment_type: 'SENT',
      amount,
      payment_method: method,
      reference,
      allocations: [
        {
          document_id: billId,
          document_type: 'BILL',
          amount,
        },
      ],
    });
  }

  // ============ Reports ============

  /**
   * Get trial balance report
   */
  async getTrialBalance(asOfDate?: string): Promise<TrialBalanceReport> {
    const response = await this.client.get<TrialBalanceReport>(
      '/v1/fin/reports/trial-balance',
      { params: { as_of_date: asOfDate } }
    );
    return response.data;
  }

  /**
   * Get income statement (P&L)
   */
  async getIncomeStatement(
    startDate: string,
    endDate: string
  ): Promise<IncomeStatementReport> {
    const response = await this.client.get<IncomeStatementReport>(
      '/v1/fin/reports/income-statement',
      { params: { start_date: startDate, end_date: endDate } }
    );
    return response.data;
  }

  /**
   * Get balance sheet
   */
  async getBalanceSheet(asOfDate?: string): Promise<BalanceSheetReport> {
    const response = await this.client.get<BalanceSheetReport>(
      '/v1/fin/reports/balance-sheet',
      { params: { as_of_date: asOfDate } }
    );
    return response.data;
  }

  /**
   * Get accounts receivable aging report
   */
  async getARAgingReport(asOfDate?: string): Promise<AgingReport> {
    const response = await this.client.get<AgingReport>(
      '/v1/fin/reports/ar-aging',
      { params: { as_of_date: asOfDate } }
    );
    return response.data;
  }

  /**
   * Get accounts payable aging report
   */
  async getAPAgingReport(asOfDate?: string): Promise<AgingReport> {
    const response = await this.client.get<AgingReport>(
      '/v1/fin/reports/ap-aging',
      { params: { as_of_date: asOfDate } }
    );
    return response.data;
  }

  // ============ Customers ============

  /**
   * List customers
   */
  async listCustomers(filters?: ListFilters): Promise<ListResponse<Customer>> {
    const response = await this.client.get<ListResponse<Customer>>(
      '/v1/fin/customers',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get a customer by ID
   */
  async getCustomer(customerId: string): Promise<Customer> {
    const response = await this.client.get<Customer>(
      `/v1/fin/customers/${customerId}`
    );
    return response.data;
  }

  /**
   * Create a customer
   */
  async createCustomer(
    customer: Omit<Customer, 'id' | 'balance' | 'created_at' | 'updated_at'>
  ): Promise<Customer> {
    const response = await this.client.post<Customer>(
      '/v1/fin/customers',
      customer
    );
    return response.data;
  }

  // ============ Vendors ============

  /**
   * List vendors
   */
  async listVendors(filters?: ListFilters): Promise<ListResponse<Vendor>> {
    const response = await this.client.get<ListResponse<Vendor>>(
      '/v1/fin/vendors',
      { params: filters }
    );
    return response.data;
  }

  /**
   * Get a vendor by ID
   */
  async getVendor(vendorId: string): Promise<Vendor> {
    const response = await this.client.get<Vendor>(
      `/v1/fin/vendors/${vendorId}`
    );
    return response.data;
  }

  /**
   * Create a vendor
   */
  async createVendor(
    vendor: Omit<Vendor, 'id' | 'balance' | 'created_at' | 'updated_at'>
  ): Promise<Vendor> {
    const response = await this.client.post<Vendor>('/v1/fin/vendors', vendor);
    return response.data;
  }
}
