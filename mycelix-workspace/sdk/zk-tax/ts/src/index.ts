/**
 * @mycelix/zk-tax - Zero-Knowledge Tax Bracket Proofs
 *
 * Privacy-preserving tax compliance using zero-knowledge proofs.
 *
 * @example
 * ```typescript
 * import { ZkTax } from '@mycelix/zk-tax';
 *
 * // Initialize the SDK
 * const zktax = await ZkTax.init();
 *
 * // Generate a proof (income is NEVER revealed)
 * const proof = await zktax.prove(85000, 'US', 'single', 2024);
 * console.log(`Bracket: ${proof.bracket_index}, Rate: ${proof.rate_bps / 100}%`);
 *
 * // Verify the proof
 * const isValid = await zktax.verify(proof);
 * console.log(`Valid: ${isValid}`);
 * ```
 *
 * @packageDocumentation
 */

import type {
  TaxBracketProof,
  ProofInfo,
  RangeProof,
  BatchProof,
  CompressedProof,
  BracketInfo,
  JurisdictionInfo,
  JurisdictionCode,
  FilingStatusCode,
  TaxYear,
  Region,
  ProveOptions,
  BatchProveOptions,
  RangeProveOptions,
  VerificationResult,
  InitStatus,
  ProgressCallback,
} from './types';

export * from './types';

// =============================================================================
// WASM Module Types
// =============================================================================

interface WasmModule {
  prove_bracket(
    income: number,
    jurisdiction: string,
    filing_status: string,
    tax_year: number
  ): TaxBracketProof;
  prove_range(
    income: number,
    tax_year: number,
    min_income: number | null,
    max_income: number | null
  ): RangeProof;
  prove_batch(
    years_json: string,
    jurisdiction: string,
    filing_status: string
  ): BatchProof;
  verify_proof(proof: TaxBracketProof): boolean;
  get_proof_info(proof: TaxBracketProof): ProofInfo;
  compress_proof(proof: TaxBracketProof): CompressedProof;
  decompress_proof(compressed: CompressedProof): TaxBracketProof;
  get_brackets(
    jurisdiction: string,
    tax_year: number,
    filing_status: string
  ): BracketInfo[];
  get_jurisdictions(): JurisdictionInfo[];
  get_supported_years(): number[];
  get_version(): string;
}

// =============================================================================
// SDK Class
// =============================================================================

/**
 * Main ZK-Tax SDK class.
 *
 * Use `ZkTax.init()` to create an instance.
 */
export class ZkTax {
  private readonly wasm: WasmModule;
  private readonly cache: Map<string, TaxBracketProof> = new Map();

  private constructor(wasm: WasmModule) {
    this.wasm = wasm;
  }

  /**
   * Initialize the ZK-Tax SDK.
   *
   * @param wasmPath - Optional path to WASM file
   * @returns Initialized ZkTax instance
   *
   * @example
   * ```typescript
   * const zktax = await ZkTax.init();
   * ```
   */
  static async init(wasmPath?: string): Promise<ZkTax> {
    // Dynamic import of WASM module
    const wasmModule = await import(wasmPath ?? '../wasm/mycelix_zk_tax.js');

    // Initialize WASM
    if (typeof wasmModule.default === 'function') {
      await wasmModule.default();
    }

    return new ZkTax(wasmModule as WasmModule);
  }

  /**
   * Get SDK initialization status.
   */
  status(): InitStatus {
    return {
      initialized: true,
      version: this.wasm.get_version(),
      wasmLoaded: true,
      jurisdictions: this.wasm.get_jurisdictions().length,
    };
  }

  // ===========================================================================
  // Proof Generation
  // ===========================================================================

  /**
   * Generate a tax bracket proof.
   *
   * The proof demonstrates that income falls within a specific tax bracket
   * WITHOUT revealing the actual income amount.
   *
   * @param income - Annual gross income (private, never leaves device)
   * @param jurisdiction - Country code (e.g., "US", "UK", "DE")
   * @param filingStatus - Filing status ("single", "mfj", "mfs", "hoh")
   * @param taxYear - Tax year (2020-2025)
   * @param options - Optional proof generation options
   * @returns Zero-knowledge tax bracket proof
   *
   * @example
   * ```typescript
   * const proof = await zktax.prove(85000, 'US', 'single', 2024);
   * console.log(`Bracket ${proof.bracket_index}: ${proof.rate_bps / 100}%`);
   * // Output: "Bracket 2: 22%"
   * ```
   */
  async prove(
    income: number,
    jurisdiction: JurisdictionCode,
    filingStatus: FilingStatusCode,
    taxYear: TaxYear,
    options: ProveOptions = {}
  ): Promise<TaxBracketProof> {
    // Check cache first
    if (options.cache !== false) {
      const cacheKey = this.makeCacheKey(jurisdiction, filingStatus, taxYear, income);
      const cached = this.cache.get(cacheKey);
      if (cached) {
        return cached;
      }
    }

    // Generate proof
    const proof = this.wasm.prove_bracket(
      income,
      jurisdiction,
      filingStatus,
      taxYear
    );

    // Cache the result
    if (options.cache !== false) {
      const cacheKey = this.makeCacheKey(jurisdiction, filingStatus, taxYear, income);
      this.cache.set(cacheKey, proof);
    }

    return proof;
  }

  /**
   * Generate a range proof.
   *
   * Proves income is within specified bounds without revealing exact value.
   *
   * @param income - Actual income (private)
   * @param taxYear - Tax year
   * @param options - Range bounds and options
   *
   * @example
   * ```typescript
   * const proof = await zktax.proveRange(85000, 2024, {
   *   minIncome: 50000,
   *   maxIncome: 100000
   * });
   * // Proves: 50,000 <= income <= 100,000
   * ```
   */
  async proveRange(
    income: number,
    taxYear: TaxYear,
    options: RangeProveOptions = {}
  ): Promise<RangeProof> {
    return this.wasm.prove_range(
      income,
      taxYear,
      options.minIncome ?? null,
      options.maxIncome ?? null
    );
  }

  /**
   * Generate a batch proof for multiple years.
   *
   * @param jurisdiction - Country code
   * @param filingStatus - Filing status
   * @param options - Years and incomes to prove
   *
   * @example
   * ```typescript
   * const proof = await zktax.proveBatch('US', 'single', {
   *   years: [
   *     { year: 2022, income: 75000 },
   *     { year: 2023, income: 82000 },
   *     { year: 2024, income: 85000 }
   *   ]
   * });
   * ```
   */
  async proveBatch(
    jurisdiction: JurisdictionCode,
    filingStatus: FilingStatusCode,
    options: BatchProveOptions
  ): Promise<BatchProof> {
    const yearsJson = JSON.stringify(options.years);
    return this.wasm.prove_batch(yearsJson, jurisdiction, filingStatus);
  }

  // ===========================================================================
  // Verification
  // ===========================================================================

  /**
   * Verify a tax bracket proof.
   *
   * @param proof - The proof to verify
   * @returns true if valid
   * @throws ZkTaxError if invalid
   *
   * @example
   * ```typescript
   * const isValid = await zktax.verify(proof);
   * ```
   */
  async verify(proof: TaxBracketProof): Promise<boolean> {
    return this.wasm.verify_proof(proof);
  }

  /**
   * Verify a proof and get detailed information.
   *
   * @param proof - The proof to verify
   * @returns Verification result with proof info
   */
  async verifyWithInfo(proof: TaxBracketProof): Promise<VerificationResult> {
    try {
      const valid = this.wasm.verify_proof(proof);
      const info = this.wasm.get_proof_info(proof);
      return { valid, info };
    } catch (error) {
      return {
        valid: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get human-readable information about a proof.
   */
  getProofInfo(proof: TaxBracketProof): ProofInfo {
    return this.wasm.get_proof_info(proof);
  }

  // ===========================================================================
  // Compression
  // ===========================================================================

  /**
   * Compress a proof for efficient storage.
   *
   * @param proof - Proof to compress
   * @returns Compressed proof
   *
   * @example
   * ```typescript
   * const compressed = zktax.compress(proof);
   * console.log(`Compressed: ${compressed.compressed_data.length} bytes`);
   * ```
   */
  compress(proof: TaxBracketProof): CompressedProof {
    return this.wasm.compress_proof(proof);
  }

  /**
   * Decompress a proof.
   */
  decompress(compressed: CompressedProof): TaxBracketProof {
    return this.wasm.decompress_proof(compressed);
  }

  // ===========================================================================
  // Bracket Information
  // ===========================================================================

  /**
   * Get tax brackets for a jurisdiction.
   *
   * @example
   * ```typescript
   * const brackets = zktax.getBrackets('US', 2024, 'single');
   * brackets.forEach(b => {
   *   console.log(`${b.rate_percent}%: $${b.lower} - $${b.upper}`);
   * });
   * ```
   */
  getBrackets(
    jurisdiction: JurisdictionCode,
    taxYear: TaxYear,
    filingStatus: FilingStatusCode
  ): BracketInfo[] {
    return this.wasm.get_brackets(jurisdiction, taxYear, filingStatus);
  }

  /**
   * Find the bracket for a specific income.
   */
  findBracket(
    income: number,
    jurisdiction: JurisdictionCode,
    taxYear: TaxYear,
    filingStatus: FilingStatusCode
  ): BracketInfo | undefined {
    const brackets = this.getBrackets(jurisdiction, taxYear, filingStatus);
    return brackets.find((b) => income >= b.lower && income <= b.upper);
  }

  /**
   * Get all supported jurisdictions.
   *
   * @example
   * ```typescript
   * const jurisdictions = zktax.getJurisdictions();
   * console.log(`${jurisdictions.length} countries supported`);
   * // Output: "58 countries supported"
   * ```
   */
  getJurisdictions(): JurisdictionInfo[] {
    return this.wasm.get_jurisdictions();
  }

  /**
   * Get jurisdictions by region.
   */
  getJurisdictionsByRegion(region: Region): JurisdictionInfo[] {
    return this.wasm.get_jurisdictions().filter((j) => j.region === region);
  }

  /**
   * Get a specific jurisdiction by code.
   */
  getJurisdiction(code: JurisdictionCode): JurisdictionInfo | undefined {
    return this.wasm.get_jurisdictions().find((j) => j.code === code);
  }

  /**
   * Get supported tax years.
   */
  getSupportedYears(): TaxYear[] {
    return this.wasm.get_supported_years() as TaxYear[];
  }

  /**
   * Get SDK version.
   */
  getVersion(): string {
    return this.wasm.get_version();
  }

  // ===========================================================================
  // Cache Management
  // ===========================================================================

  /**
   * Clear the proof cache.
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics.
   */
  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }

  private makeCacheKey(
    jurisdiction: string,
    filingStatus: string,
    taxYear: number,
    income: number
  ): string {
    return `${jurisdiction}:${filingStatus}:${taxYear}:${income}`;
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format a tax rate for display.
 *
 * @example
 * ```typescript
 * formatRate(2200); // "22%"
 * formatRate(3700); // "37%"
 * ```
 */
export function formatRate(rateBps: number): string {
  return `${rateBps / 100}%`;
}

/**
 * Format currency amount.
 *
 * @example
 * ```typescript
 * formatCurrency(85000, 'USD'); // "$85,000"
 * formatCurrency(85000, 'EUR'); // "85.000 €"
 * ```
 */
export function formatCurrency(
  amount: number,
  currency: string,
  locale?: string
): string {
  return new Intl.NumberFormat(locale ?? 'en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(amount);
}

/**
 * Check if a proof is in dev mode.
 */
export function isDevMode(proof: TaxBracketProof): boolean {
  // Dev mode uses a specific marker in receipt_bytes
  return (
    proof.receipt_bytes.length === 4 &&
    proof.receipt_bytes[0] === 0xde &&
    proof.receipt_bytes[1] === 0xad &&
    proof.receipt_bytes[2] === 0xbe &&
    proof.receipt_bytes[3] === 0xef
  );
}

/**
 * Validate a jurisdiction code.
 */
export function isValidJurisdiction(code: string): code is JurisdictionCode {
  const validCodes = [
    'US', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'EC', 'UY',
    'UK', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'PT', 'IE',
    'PL', 'SE', 'DK', 'FI', 'NO', 'CH', 'CZ', 'GR', 'HU', 'RO',
    'RU', 'TR', 'UA',
    'JP', 'CN', 'IN', 'KR', 'ID', 'AU', 'NZ', 'SG', 'HK', 'TW',
    'MY', 'TH', 'VN', 'PH', 'PK',
    'SA', 'AE', 'IL', 'EG', 'QA',
    'ZA', 'NG', 'KE', 'MA', 'GH',
  ];
  return validCodes.includes(code);
}

/**
 * Validate a tax year.
 */
export function isValidTaxYear(year: number): year is TaxYear {
  return year >= 2020 && year <= 2025;
}

/**
 * Get region for a jurisdiction.
 */
export function getRegion(code: JurisdictionCode): Region {
  const americas = ['US', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'EC', 'UY'];
  const asiaPacific = ['JP', 'CN', 'IN', 'KR', 'ID', 'AU', 'NZ', 'SG', 'HK', 'TW', 'MY', 'TH', 'VN', 'PH', 'PK'];
  const middleEast = ['SA', 'AE', 'IL', 'EG', 'QA'];
  const africa = ['ZA', 'NG', 'KE', 'MA', 'GH'];

  if (americas.includes(code)) return 'Americas';
  if (asiaPacific.includes(code)) return 'Asia-Pacific';
  if (middleEast.includes(code)) return 'Middle East';
  if (africa.includes(code)) return 'Africa';
  return 'Europe';
}

// =============================================================================
// Default Export
// =============================================================================

export default ZkTax;
