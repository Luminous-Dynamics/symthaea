/**
 * Community Configuration — loads deployment-specific settings from community-config.json.
 *
 * To deploy Mycelix for a different community:
 * 1. Edit community-config.json with your local basket items, currency, and tax rules
 * 2. Everything else adapts automatically
 *
 * Example configs are in observatory/configs/
 */

import defaultConfig from './community-config.json';

// ============================================================================
// Types
// ============================================================================

export interface BasketItemConfig {
  key: string;
  name: string;
  unit: string;
  default_price: number;
  weight: number;
}

export interface CommunityConfig {
  /** Display name for the cooperative */
  community_name: string;
  /** Name for the price basket */
  basket_name: string;
  /** Default DAO identifier */
  dao_did: string;
  /** ISO 4217 currency code for local fiat */
  currency_code: string;
  /** Currency symbol for display (R, £, $, ₹, KSh) */
  currency_symbol: string;
  /** Value of 1 hour of labor in local currency (minimum wage or equivalent) */
  labor_hour_value: number;
  /** Source citation for the hourly rate */
  labor_hour_source: string;
  /** Month the tax year starts (1=Jan, 3=Mar, 4=Apr) */
  tax_year_start_month: number;
  /** Name of the tax form for reporting */
  tax_form_name: string;
  /** Name of the tax authority */
  tax_authority: string;
  /** Basket items with weights (weights must sum to ~1.0) */
  basket_items: BasketItemConfig[];
}

// ============================================================================
// Singleton Config
// ============================================================================

let _config: CommunityConfig = defaultConfig as CommunityConfig;

/** Get the current community configuration. */
export function getCommunityConfig(): CommunityConfig {
  return _config;
}

/** Override the community config (for testing or runtime switching). */
export function setCommunityConfig(config: CommunityConfig): void {
  _config = config;
}

// ============================================================================
// Derived Accessors
// ============================================================================

/** Canonical basket items (keys, names, units, default prices). */
export function getCanonicalItems(): { key: string; name: string; unit: string; default_price: number }[] {
  return _config.basket_items.map(({ key, name, unit, default_price }) => ({
    key, name, unit, default_price,
  }));
}

/** Basket weights for defining the on-chain basket. */
export function getBasketWeights(): { item: string; weight: number }[] {
  return _config.basket_items.map(({ key, weight }) => ({
    item: key,
    weight,
  }));
}

/** Community basket name. */
export function getBasketName(): string {
  return _config.basket_name;
}

/** Default DAO DID. */
export function getDefaultDao(): string {
  return _config.dao_did;
}

/**
 * Compute tax year bounds for a given year.
 * SA: Mar 1 → Feb 28. UK: Apr 6 → Apr 5. US: Jan 1 → Dec 31.
 */
export function getTaxYearBounds(year: string): { start: Date; end: Date } {
  const y = parseInt(year);
  const startMonth = _config.tax_year_start_month;

  if (startMonth === 1) {
    // Jan-Dec (US, Kenya)
    return {
      start: new Date(y, 0, 1),
      end: new Date(y, 11, 31, 23, 59, 59),
    };
  }

  // Mar-Feb (SA), Apr-Mar (UK, India), etc.
  return {
    start: new Date(y, startMonth - 1, 1),
    end: new Date(y + 1, startMonth - 1, 0, 23, 59, 59), // last day of previous month
  };
}

/** Format a monetary value in local currency. */
export function formatCurrency(value: number): string {
  return `${_config.currency_symbol}${value.toFixed(2)}`;
}
