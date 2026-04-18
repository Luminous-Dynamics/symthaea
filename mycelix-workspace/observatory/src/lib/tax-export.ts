// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tax Export — generates SARS-compliant records of TEND exchanges.
 *
 * South African Revenue Service requires barter transactions to be reported
 * at their fair market value in Rands. This module takes TEND exchange history
 * and the price oracle's basket index to produce exportable records.
 *
 * Legal basis: Income Tax Act (Section 1, "gross income" includes barter);
 * SARS Practice Note on barter transactions.
 *
 * Output: CSV suitable for attachment to IT12 return.
 */

import {
  type ExchangeRecord,
} from './resilience-client';
import {
  getCommunityConfig,
  getTaxYearBounds,
  formatCurrency,
} from './community';

// ============================================================================
// Dual-DID guard — see MYCELIX_STATE_COEXISTENCE.md
// ============================================================================

/**
 * Raised when tax-export is called under the primary DID. State-facing
 * output must be generated under `did:mycelix:legal:<opaque>` to keep
 * it cryptographically unlinked from the user's pseudonymous identity.
 * See `mycelix-lawful-identity/docs/THREAT_MODEL.md`.
 */
export class PrimaryDidStateInteropError extends Error {
  constructor(did: string) {
    super(
      `tax-export refuses to run under primary DID "${did}". ` +
      `State-facing exports must use did:mycelix:legal:* to preserve the ` +
      `on-chain separation between consciousness-gated identity and ` +
      `state-facing identity. See MYCELIX_STATE_COEXISTENCE.md.`,
    );
    this.name = 'PrimaryDidStateInteropError';
  }
}

/**
 * Assert that a DID is NOT the primary consciousness-gated identity.
 *
 * Rules:
 * - `did:mycelix:primary:*` → REJECTED (hard error).
 * - `did:mycelix:legal:*`   → accepted.
 * - anything else           → accepted (legacy DIDs, external DIDs).
 *
 * The legacy-accept carveout exists because the existing tax-export
 * callers use unqualified `did:mycelix:<agent>` strings. A future
 * migration pass should tighten this to require the `legal:` prefix
 * explicitly.
 */
export function assertLegalDid(did: string): void {
  if (did.startsWith('did:mycelix:primary:') || did === 'did:mycelix:primary') {
    throw new PrimaryDidStateInteropError(did);
  }
}

// ============================================================================
// Types
// ============================================================================

export interface TaxExportRow {
  /** ISO 8601 date string */
  date: string;
  /** 'provided' (income) or 'received' (expense) */
  direction: 'provided' | 'received';
  /** Service category */
  category: string;
  /** Description of service */
  description: string;
  /** TEND hours */
  hours: number;
  /** Estimated local currency value (hours * hourly_rate) */
  estimated_local_value: number;
  /** Counterparty DID (truncated for privacy) */
  counterparty: string;
  /** Exchange status */
  status: string;
}

export interface TaxExportSummary {
  /** Member DID */
  member_did: string;
  /** DAO/cooperative name */
  dao_did: string;
  /** Tax year */
  tax_year: string;
  /** Date range */
  period_start: string;
  period_end: string;
  /** Total hours provided (taxable income) */
  total_hours_provided: number;
  /** Total hours received */
  total_hours_received: number;
  /** Net hours */
  net_hours: number;
  /** Estimated local currency value of hours provided */
  estimated_income: number;
  /** Estimated local currency value of hours received */
  estimated_expense: number;
  /** Hourly rate used for conversion */
  hourly_rate: number;
  /** Currency code (e.g., ZAR, GBP, USD) */
  currency_code: string;
  /** Currency symbol (e.g., R, £, $) */
  currency_symbol: string;
  /** Tax authority name */
  tax_authority: string;
  /** Tax form name */
  tax_form: string;
  /** Number of confirmed exchanges */
  exchange_count: number;
  /** Individual rows */
  rows: TaxExportRow[];
}

// ============================================================================
// Export Generation
// ============================================================================

/**
 * Generate a tax export summary from TEND exchange records.
 *
 * Hourly rate, currency, tax year bounds, and authority are all loaded
 * from community-config.json — making this work for any jurisdiction.
 *
 * @param exchanges - All exchanges for the member (from get_my_exchanges)
 * @param memberDid - The member's DID
 * @param daoDid - The DAO/cooperative
 * @param taxYear - Tax year string (e.g., "2026")
 */
export function generateTaxExport(
  exchanges: ExchangeRecord[],
  memberDid: string,
  daoDid: string,
  taxYear: string,
): TaxExportSummary {
  // State-facing output MUST NOT run under the primary DID — not as
  // the member (that would deanonymize the filer) and not as the
  // DAO id (that would embed a primary DID in the taxable-entity
  // reference, a subtler but equally bad leak).
  // See MYCELIX_STATE_COEXISTENCE.md for the dual-DID rule.
  assertLegalDid(memberDid);
  assertLegalDid(daoDid);

  const config = getCommunityConfig();
  const hourlyRate = config.labor_hour_value;
  const { start, end } = getTaxYearBounds(taxYear);

  const confirmed = exchanges.filter((ex) => {
    if (ex.status !== 'Confirmed') return false;
    const ts = typeof ex.timestamp === 'number' ? ex.timestamp : Date.now();
    return ts >= start.getTime() && ts <= end.getTime();
  });

  const rows: TaxExportRow[] = [];
  let totalProvided = 0;
  let totalReceived = 0;

  for (const ex of confirmed) {
    const isProvider = ex.provider_did === memberDid;
    const direction = isProvider ? 'provided' : 'received';
    const hours = ex.hours;
    const localValue = hours * hourlyRate;

    if (isProvider) {
      totalProvided += hours;
    } else {
      totalReceived += hours;
    }

    const ts = typeof ex.timestamp === 'number' ? ex.timestamp : Date.now();

    rows.push({
      date: new Date(ts).toISOString().split('T')[0],
      direction,
      category: ex.service_category,
      description: ex.service_description,
      hours,
      estimated_local_value: Math.round(localValue * 100) / 100,
      counterparty: truncateDid(isProvider ? ex.receiver_did : ex.provider_did),
      status: ex.status,
    });
  }

  // Sort by date
  rows.sort((a, b) => a.date.localeCompare(b.date));

  const startStr = `${start.getFullYear()}-${String(start.getMonth() + 1).padStart(2, '0')}-${String(start.getDate()).padStart(2, '0')}`;
  const endStr = `${end.getFullYear()}-${String(end.getMonth() + 1).padStart(2, '0')}-${String(end.getDate()).padStart(2, '0')}`;

  return {
    member_did: memberDid,
    dao_did: daoDid,
    tax_year: taxYear,
    period_start: startStr,
    period_end: endStr,
    total_hours_provided: Math.round(totalProvided * 100) / 100,
    total_hours_received: Math.round(totalReceived * 100) / 100,
    net_hours: Math.round((totalProvided - totalReceived) * 100) / 100,
    estimated_income: Math.round(totalProvided * hourlyRate * 100) / 100,
    estimated_expense: Math.round(totalReceived * hourlyRate * 100) / 100,
    hourly_rate: hourlyRate,
    currency_code: config.currency_code,
    currency_symbol: config.currency_symbol,
    tax_authority: config.tax_authority,
    tax_form: config.tax_form_name,
    exchange_count: confirmed.length,
    rows,
  };
}

// ============================================================================
// CSV Export
// ============================================================================

/**
 * Convert a tax export summary to CSV string.
 * Format adapts to the community's jurisdiction automatically.
 */
export function exportToCsv(summary: TaxExportSummary): string {
  const config = getCommunityConfig();
  const lines: string[] = [];

  // Header section
  lines.push(`# MYCELIX COOPERATIVE — TEND EXCHANGE TAX RECORD`);
  lines.push(`# Cooperative: ${summary.dao_did}`);
  lines.push(`# Member: ${summary.member_did}`);
  lines.push(`# Tax Year: ${summary.tax_year} (${summary.period_start} to ${summary.period_end})`);
  lines.push(`# Tax Authority: ${summary.tax_authority} (${summary.tax_form})`);
  lines.push(`# Generated: ${new Date().toISOString()}`);
  lines.push(`# Valuation: ${summary.currency_code} ${summary.hourly_rate}/hour (${config.labor_hour_source})`);
  lines.push('#');
  lines.push(`# SUMMARY`);
  lines.push(`# Hours provided (income): ${summary.total_hours_provided}`);
  lines.push(`# Hours received (expense): ${summary.total_hours_received}`);
  lines.push(`# Net hours: ${summary.net_hours}`);
  lines.push(`# Estimated income: ${summary.currency_symbol}${summary.estimated_income.toFixed(2)}`);
  lines.push(`# Estimated expense: ${summary.currency_symbol}${summary.estimated_expense.toFixed(2)}`);
  lines.push(`# Confirmed exchanges: ${summary.exchange_count}`);
  lines.push('#');

  // CSV header
  lines.push(`Date,Direction,Category,Description,Hours,Estimated_${summary.currency_code},Counterparty,Status`);

  // Data rows
  for (const row of summary.rows) {
    lines.push([
      row.date,
      row.direction,
      csvEscape(row.category),
      csvEscape(row.description),
      row.hours.toFixed(2),
      row.estimated_local_value.toFixed(2),
      row.counterparty,
      row.status,
    ].join(','));
  }

  return lines.join('\n');
}

/**
 * Convert a tax export summary to JSON string.
 */
export function exportToJson(summary: TaxExportSummary): string {
  return JSON.stringify(summary, null, 2);
}

/**
 * Trigger a browser download of the export file.
 */
export function downloadExport(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// Helpers
// ============================================================================

function csvEscape(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function truncateDid(did: string): string {
  if (did.length <= 20) return did;
  return `${did.slice(0, 12)}...${did.slice(-6)}`;
}
