/**
 * UESS E4 Reproducibility Bundle Verification
 *
 * Verification of E4-level claims requiring public reproducibility.
 * @see docs/architecture/uess/UESS-04-EMPIRICAL-VERIFICATION.md
 */

import { hash, secureUUID, type HashAlgorithm } from '../security/index.js';
import { computeCID } from './backends/memory.js';

// Map algorithm names to Web Crypto format
const ALGO_MAP: Record<string, HashAlgorithm> = {
  'sha256': 'SHA-256',
  'sha512': 'SHA-512',
  'blake3': 'SHA-256', // Fallback to SHA-256 for blake3
};

// =============================================================================
// Types
// =============================================================================

/**
 * Reproducibility Bundle - Contains everything needed to reproduce a result
 */
export interface ReproBundle {
  /** Unique bundle identifier */
  id: string;

  /** Version of the bundle format */
  version: '1.0.0';

  /** Content identifier for the bundle */
  cid: string;

  /** When the bundle was created */
  createdAt: number;

  /** Who created the bundle */
  creator: string;

  /** Description of what this bundle reproduces */
  description: string;

  /** The claim being verified */
  claim: ReproClaim;

  /** Input data specification */
  inputs: ReproInput[];

  /** Execution environment specification */
  environment: ReproEnvironment;

  /** Execution steps */
  steps: ReproStep[];

  /** Expected outputs */
  expectedOutputs: ReproOutput[];

  /** Verification configuration */
  verification: VerificationConfig;

  /** Signatures from reproducers */
  reproductions: ReproductionRecord[];
}

/**
 * The claim being verified for reproducibility
 */
export interface ReproClaim {
  /** Claim statement */
  statement: string;

  /** Domain/field of the claim */
  domain: string;

  /** Original source of the claim */
  source: string;

  /** Original publication date */
  publishedAt?: number;

  /** DOI or other identifier */
  identifier?: string;
}

/**
 * Input data for reproduction
 */
export interface ReproInput {
  /** Input identifier */
  id: string;

  /** Human-readable name */
  name: string;

  /** Input type */
  type: 'data' | 'code' | 'model' | 'config' | 'environment';

  /** Content hash (for verification) */
  hash: string;

  /** Hash algorithm used */
  hashAlgorithm: 'sha256' | 'sha512' | 'blake3';

  /** Size in bytes */
  sizeBytes: number;

  /** Content-addressed location */
  location: {
    type: 'ipfs' | 'http' | 'git' | 'inline';
    uri: string;
  };

  /** Optional inline content (for small inputs) */
  inlineContent?: string;

  /** Schema or format specification */
  format?: string;
}

/**
 * Execution environment specification
 */
export interface ReproEnvironment {
  /** OS requirements */
  os?: {
    type: 'linux' | 'darwin' | 'windows' | 'any';
    minVersion?: string;
  };

  /** Hardware requirements */
  hardware?: {
    minMemoryGB?: number;
    minCPUCores?: number;
    gpu?: boolean;
    gpuType?: string;
  };

  /** Software dependencies */
  dependencies: Array<{
    name: string;
    version: string;
    source?: string;
  }>;

  /** Container specification (preferred for reproducibility) */
  container?: {
    type: 'docker' | 'podman' | 'nix';
    image?: string;
    dockerfile?: string;
    nixFlake?: string;
  };

  /** Environment variables */
  envVars?: Record<string, string>;

  /** Random seed for reproducibility */
  randomSeed?: number;
}

/**
 * Execution step
 */
export interface ReproStep {
  /** Step order (1-indexed) */
  order: number;

  /** Step name */
  name: string;

  /** Step description */
  description: string;

  /** Command to execute */
  command: string;

  /** Working directory */
  workDir?: string;

  /** Expected exit code */
  expectedExitCode: number;

  /** Expected execution time (for timeout) */
  timeoutSeconds: number;

  /** Inputs used by this step */
  inputRefs: string[];

  /** Outputs produced by this step */
  outputRefs: string[];
}

/**
 * Expected output
 */
export interface ReproOutput {
  /** Output identifier */
  id: string;

  /** Output name */
  name: string;

  /** Output type */
  type: 'file' | 'directory' | 'stdout' | 'metric' | 'assertion';

  /** Path where output is produced */
  path?: string;

  /** Expected content hash (for exact match) */
  expectedHash?: string;

  /** Tolerance for numeric outputs */
  tolerance?: {
    type: 'absolute' | 'relative' | 'statistical';
    value: number;
    confidence?: number;
  };

  /** Validation schema */
  schema?: string;

  /** Human-readable description of expected result */
  expectedDescription: string;
}

/**
 * Verification configuration
 */
export interface VerificationConfig {
  /** Minimum number of independent reproductions required */
  minReproductions: number;

  /** Time limit for reproduction in seconds */
  maxReproductionTimeSeconds: number;

  /** Whether reproductions must be from different organizations */
  requireDifferentOrgs: boolean;

  /** Whether reproductions must use different hardware */
  requireDifferentHardware: boolean;

  /** Allowed deviation from expected results */
  allowedDeviation: 'exact' | 'statistical' | 'semantic';

  /** Confidence level for statistical verification */
  confidenceLevel?: number;
}

/**
 * Record of a reproduction attempt
 */
export interface ReproductionRecord {
  /** Reproduction ID */
  id: string;

  /** Bundle ID being reproduced */
  bundleId: string;

  /** Who performed the reproduction */
  reproducer: string;

  /** Organization of reproducer */
  organization?: string;

  /** When reproduction was performed */
  performedAt: number;

  /** Whether reproduction succeeded */
  success: boolean;

  /** Environment used */
  environmentUsed: Partial<ReproEnvironment>;

  /** Output results */
  results: Array<{
    outputId: string;
    success: boolean;
    actualHash?: string;
    deviation?: number;
    notes?: string;
  }>;

  /** Total execution time */
  executionTimeSeconds: number;

  /** Digital signature of the record */
  signature: string;

  /** Notes from reproducer */
  notes?: string;
}

/**
 * Bundle verification status
 */
export interface VerificationStatus {
  bundleId: string;
  isVerified: boolean;
  totalReproductions: number;
  successfulReproductions: number;
  failedReproductions: number;
  uniqueReproducers: number;
  uniqueOrganizations: number;
  meetsMinReproductions: boolean;
  meetsDiversityRequirements: boolean;
  confidenceScore: number;
  lastReproductionAt?: number;
}

// =============================================================================
// ReproBundle Manager
// =============================================================================

/**
 * Manager for E4 Reproducibility Bundles
 */
export class ReproBundleManager {
  private readonly bundles: Map<string, ReproBundle> = new Map();
  private readonly reproductions: Map<string, ReproductionRecord[]> = new Map();

  // ===========================================================================
  // Bundle Creation
  // ===========================================================================

  /**
   * Create a new reproducibility bundle
   */
  async createBundle(params: {
    creator: string;
    description: string;
    claim: ReproClaim;
    inputs: ReproInput[];
    environment: ReproEnvironment;
    steps: ReproStep[];
    expectedOutputs: ReproOutput[];
    verification?: Partial<VerificationConfig>;
  }): Promise<ReproBundle> {
    const id = `repro_${secureUUID()}`;

    // Validate inputs have hashes
    for (const input of params.inputs) {
      if (!input.hash) {
        throw new Error(`Input '${input.id}' must have a content hash`);
      }
    }

    // Validate steps reference valid inputs/outputs
    const inputIds = new Set(params.inputs.map(i => i.id));
    const outputIds = new Set(params.expectedOutputs.map(o => o.id));

    for (const step of params.steps) {
      for (const ref of step.inputRefs) {
        if (!inputIds.has(ref)) {
          throw new Error(`Step '${step.name}' references unknown input '${ref}'`);
        }
      }
      for (const ref of step.outputRefs) {
        if (!outputIds.has(ref)) {
          throw new Error(`Step '${step.name}' references unknown output '${ref}'`);
        }
      }
    }

    const verification: VerificationConfig = {
      minReproductions: params.verification?.minReproductions ?? 3,
      maxReproductionTimeSeconds: params.verification?.maxReproductionTimeSeconds ?? 3600,
      requireDifferentOrgs: params.verification?.requireDifferentOrgs ?? true,
      requireDifferentHardware: params.verification?.requireDifferentHardware ?? false,
      allowedDeviation: params.verification?.allowedDeviation ?? 'exact',
      confidenceLevel: params.verification?.confidenceLevel ?? 0.95,
    };

    const bundleContent = {
      id,
      version: '1.0.0' as const,
      createdAt: Date.now(),
      creator: params.creator,
      description: params.description,
      claim: params.claim,
      inputs: params.inputs,
      environment: params.environment,
      steps: params.steps,
      expectedOutputs: params.expectedOutputs,
      verification,
      reproductions: [],
    };

    const cid = computeCID(bundleContent);
    const bundle: ReproBundle = { ...bundleContent, cid };

    this.bundles.set(id, bundle);
    this.reproductions.set(id, []);

    return bundle;
  }

  /**
   * Get a bundle by ID
   */
  getBundle(bundleId: string): ReproBundle | null {
    return this.bundles.get(bundleId) ?? null;
  }

  /**
   * Get a bundle by CID
   */
  getBundleByCID(cid: string): ReproBundle | null {
    for (const bundle of this.bundles.values()) {
      if (bundle.cid === cid) {
        return bundle;
      }
    }
    return null;
  }

  // ===========================================================================
  // Reproduction Recording
  // ===========================================================================

  /**
   * Record a reproduction attempt
   */
  async recordReproduction(record: Omit<ReproductionRecord, 'id'>): Promise<ReproductionRecord> {
    const bundle = this.bundles.get(record.bundleId);
    if (!bundle) {
      throw new Error(`Bundle not found: ${record.bundleId}`);
    }

    const fullRecord: ReproductionRecord = {
      ...record,
      id: `repro_rec_${secureUUID()}`,
    };

    // Validate all outputs are accounted for
    const expectedOutputIds = new Set(bundle.expectedOutputs.map(o => o.id));
    for (const result of fullRecord.results) {
      if (!expectedOutputIds.has(result.outputId)) {
        throw new Error(`Unknown output '${result.outputId}' in reproduction record`);
      }
    }

    // Add to reproductions
    const bundleRepros = this.reproductions.get(record.bundleId) ?? [];
    bundleRepros.push(fullRecord);
    this.reproductions.set(record.bundleId, bundleRepros);

    // Update bundle
    bundle.reproductions.push(fullRecord);

    return fullRecord;
  }

  /**
   * Get all reproductions for a bundle
   */
  getReproductions(bundleId: string): ReproductionRecord[] {
    return this.reproductions.get(bundleId) ?? [];
  }

  // ===========================================================================
  // Verification
  // ===========================================================================

  /**
   * Get verification status for a bundle
   */
  getVerificationStatus(bundleId: string): VerificationStatus {
    const bundle = this.bundles.get(bundleId);
    if (!bundle) {
      throw new Error(`Bundle not found: ${bundleId}`);
    }

    const repros = this.reproductions.get(bundleId) ?? [];
    const successfulRepros = repros.filter(r => r.success);
    const uniqueReproducers = new Set(repros.map(r => r.reproducer)).size;
    const uniqueOrgs = new Set(repros.filter(r => r.organization).map(r => r.organization)).size;

    const meetsMinRepros = successfulRepros.length >= bundle.verification.minReproductions;
    const meetsDiversity = !bundle.verification.requireDifferentOrgs || uniqueOrgs >= 2;

    // Calculate confidence score based on reproductions
    const confidenceScore = this.calculateConfidenceScore(bundle, successfulRepros);

    const lastRepro = repros.length > 0
      ? Math.max(...repros.map(r => r.performedAt))
      : undefined;

    return {
      bundleId,
      isVerified: meetsMinRepros && meetsDiversity && confidenceScore >= (bundle.verification.confidenceLevel ?? 0.95),
      totalReproductions: repros.length,
      successfulReproductions: successfulRepros.length,
      failedReproductions: repros.length - successfulRepros.length,
      uniqueReproducers,
      uniqueOrganizations: uniqueOrgs,
      meetsMinReproductions: meetsMinRepros,
      meetsDiversityRequirements: meetsDiversity,
      confidenceScore,
      lastReproductionAt: lastRepro,
    };
  }

  /**
   * Check if a bundle is fully verified (E4 level)
   */
  isVerified(bundleId: string): boolean {
    try {
      const status = this.getVerificationStatus(bundleId);
      return status.isVerified;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // Validation
  // ===========================================================================

  /**
   * Validate bundle structure
   */
  validateBundle(bundle: ReproBundle): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check required fields
    if (!bundle.id) errors.push('Missing bundle ID');
    if (!bundle.claim?.statement) errors.push('Missing claim statement');
    if (!bundle.inputs?.length) errors.push('No inputs specified');
    if (!bundle.steps?.length) errors.push('No execution steps');
    if (!bundle.expectedOutputs?.length) errors.push('No expected outputs');

    // Validate inputs
    for (const input of bundle.inputs) {
      if (!input.hash) {
        errors.push(`Input '${input.id}' missing content hash`);
      }
      if (!input.location?.uri && !input.inlineContent) {
        errors.push(`Input '${input.id}' has no location or inline content`);
      }
    }

    // Validate steps are ordered
    const stepOrders = bundle.steps.map(s => s.order);
    for (let i = 1; i <= bundle.steps.length; i++) {
      if (!stepOrders.includes(i)) {
        errors.push(`Missing step order ${i}`);
      }
    }

    // Validate outputs
    for (const output of bundle.expectedOutputs) {
      if (output.type !== 'assertion' && !output.expectedHash && !output.tolerance) {
        errors.push(`Output '${output.id}' needs expectedHash or tolerance`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate an input against its hash
   */
  async validateInputHash(
    input: ReproInput,
    content: Uint8Array
  ): Promise<{ valid: boolean; actualHash: string }> {
    const algorithm = ALGO_MAP[input.hashAlgorithm] ?? 'SHA-256';
    const hashBytes = await hash(content, algorithm);
    const actualHash = Buffer.from(hashBytes).toString('hex');

    return {
      valid: actualHash === input.hash,
      actualHash,
    };
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private calculateConfidenceScore(
    bundle: ReproBundle,
    successfulRepros: ReproductionRecord[]
  ): number {
    if (successfulRepros.length === 0) return 0;

    const targetRepros = bundle.verification.minReproductions;
    const reproRatio = Math.min(successfulRepros.length / targetRepros, 1);

    // Check diversity
    const uniqueReproducers = new Set(successfulRepros.map(r => r.reproducer)).size;
    const diversityBonus = Math.min(uniqueReproducers / targetRepros, 1) * 0.2;

    // Check output consistency
    let consistencyScore = 1;
    if (successfulRepros.length >= 2) {
      const outputConsistency = this.calculateOutputConsistency(bundle, successfulRepros);
      consistencyScore = outputConsistency;
    }

    return Math.min(reproRatio * 0.6 + consistencyScore * 0.2 + diversityBonus, 1);
  }

  private calculateOutputConsistency(
    bundle: ReproBundle,
    repros: ReproductionRecord[]
  ): number {
    if (repros.length < 2) return 1;

    let totalChecks = 0;
    let consistentChecks = 0;

    for (const output of bundle.expectedOutputs) {
      const hashes = repros
        .flatMap(r => r.results)
        .filter(r => r.outputId === output.id && r.actualHash)
        .map(r => r.actualHash);

      if (hashes.length >= 2) {
        totalChecks++;
        // Check if all hashes match
        if (hashes.every(h => h === hashes[0])) {
          consistentChecks++;
        }
      }
    }

    return totalChecks === 0 ? 1 : consistentChecks / totalChecks;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new ReproBundle manager
 */
export function createReproBundleManager(): ReproBundleManager {
  return new ReproBundleManager();
}

/**
 * Create a simple reproducibility bundle for a computation
 */
export async function createComputationBundle(
  manager: ReproBundleManager,
  params: {
    creator: string;
    claim: string;
    code: { content: string; language: string };
    data: Array<{ name: string; hash: string; location: string }>;
    expectedResult: { hash: string; description: string };
  }
): Promise<ReproBundle> {
  const codeHash = computeCID(params.code.content);

  return manager.createBundle({
    creator: params.creator,
    description: `Reproducibility bundle for: ${params.claim}`,
    claim: {
      statement: params.claim,
      domain: 'computation',
      source: params.creator,
    },
    inputs: [
      {
        id: 'code',
        name: 'Execution Code',
        type: 'code',
        hash: codeHash,
        hashAlgorithm: 'sha256',
        sizeBytes: new TextEncoder().encode(params.code.content).length,
        location: { type: 'inline', uri: 'inline' },
        inlineContent: params.code.content,
        format: params.code.language,
      },
      ...params.data.map((d, i) => ({
        id: `data_${i}`,
        name: d.name,
        type: 'data' as const,
        hash: d.hash,
        hashAlgorithm: 'sha256' as const,
        sizeBytes: 0, // Unknown until fetched
        location: { type: 'ipfs' as const, uri: d.location },
      })),
    ],
    environment: {
      dependencies: [],
      container: { type: 'nix' },
    },
    steps: [
      {
        order: 1,
        name: 'Execute computation',
        description: 'Run the computation code with provided data',
        command: `run ${params.code.language}`,
        expectedExitCode: 0,
        timeoutSeconds: 3600,
        inputRefs: ['code', ...params.data.map((_, i) => `data_${i}`)],
        outputRefs: ['result'],
      },
    ],
    expectedOutputs: [
      {
        id: 'result',
        name: 'Computation Result',
        type: 'file',
        path: 'output/result',
        expectedHash: params.expectedResult.hash,
        expectedDescription: params.expectedResult.description,
      },
    ],
  });
}
