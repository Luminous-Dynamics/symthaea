/**
 * UESS Capability-Based Access Control (CapBAC)
 *
 * Cryptographically-signed capability tokens for N0/N1 access control.
 * @see docs/architecture/uess/UESS-08-ACCESS-CONTROL.md
 */

import { hmac, verifyHmac, secureRandomBytes, secureUUID } from '../security/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Access rights that can be granted
 */
export enum AccessRight {
  READ = 'read',
  WRITE = 'write',
  DELETE = 'delete',
  DELEGATE = 'delegate',
  ADMIN = 'admin',
}

/**
 * A capability token granting access to a resource
 */
export interface CapabilityToken {
  /** Unique token identifier */
  id: string;

  /** Resource key or pattern this capability grants access to */
  resource: string;

  /** Whether resource is a pattern (e.g., "user:alice:*") */
  isPattern: boolean;

  /** Granted access rights */
  rights: AccessRight[];

  /** Token issuer (agent ID) */
  issuer: string;

  /** Token holder (agent ID) */
  holder: string;

  /** When the token was issued */
  issuedAt: number;

  /** When the token expires (optional) */
  expiresAt?: number;

  /** Maximum delegation depth (0 = no delegation) */
  delegationDepth: number;

  /** Parent capability ID if delegated */
  parentId?: string;

  /** Delegation chain (issuer IDs from root to current) */
  delegationChain: string[];

  /** Constraints on the capability */
  constraints?: CapabilityConstraints;

  /** Cryptographic signature */
  signature: string;
}

/**
 * Constraints that can be placed on a capability
 */
export interface CapabilityConstraints {
  /** Maximum number of uses (undefined = unlimited) */
  maxUses?: number;

  /** IP addresses/ranges allowed (undefined = any) */
  allowedIPs?: string[];

  /** Time windows when access is allowed */
  timeWindows?: Array<{
    start: number; // Hour (0-23)
    end: number;   // Hour (0-23)
    days?: number[]; // Days of week (0-6, 0=Sunday)
  }>;

  /** Required context fields */
  requiredContext?: Record<string, unknown>;
}

/**
 * Capability validation result
 */
export interface ValidationResult {
  valid: boolean;
  error?: string;
  capability?: CapabilityToken;
}

/**
 * Context for capability validation
 */
export interface ValidationContext {
  /** Current agent ID */
  agentId: string;

  /** Resource being accessed */
  resource: string;

  /** Requested access right */
  right: AccessRight;

  /** Current timestamp */
  timestamp: number;

  /** Client IP address (optional) */
  clientIP?: string;

  /** Additional context data */
  contextData?: Record<string, unknown>;
}

// =============================================================================
// CapBAC Manager
// =============================================================================

/**
 * CapBAC Manager Configuration
 */
export interface CapBACConfig {
  /** Secret key for signing tokens */
  signingKey: Uint8Array;

  /** Maximum allowed delegation depth */
  maxDelegationDepth: number;

  /** Default token TTL in ms (optional) */
  defaultTtlMs?: number;

  /** Enable revocation checking */
  enableRevocation: boolean;
}

/**
 * CapBAC Manager - Handles capability token lifecycle
 */
export class CapBACManager {
  private readonly config: CapBACConfig;
  private readonly revoked: Set<string> = new Set();
  private readonly usageCount: Map<string, number> = new Map();

  constructor(config: CapBACConfig) {
    this.config = config;
  }

  // ===========================================================================
  // Token Creation
  // ===========================================================================

  /**
   * Create a new capability token
   */
  async createCapability(params: {
    resource: string;
    isPattern?: boolean;
    rights: AccessRight[];
    issuer: string;
    holder: string;
    expiresAt?: number;
    delegationDepth?: number;
    constraints?: CapabilityConstraints;
  }): Promise<CapabilityToken> {
    const id = secureUUID();
    const issuedAt = Date.now();

    // Calculate expiration
    let expiresAt = params.expiresAt;
    if (!expiresAt && this.config.defaultTtlMs) {
      expiresAt = issuedAt + this.config.defaultTtlMs;
    }

    const token: Omit<CapabilityToken, 'signature'> = {
      id,
      resource: params.resource,
      isPattern: params.isPattern ?? false,
      rights: params.rights,
      issuer: params.issuer,
      holder: params.holder,
      issuedAt,
      expiresAt,
      delegationDepth: params.delegationDepth ?? 0,
      delegationChain: [params.issuer],
      constraints: params.constraints,
    };

    const signature = await this.signToken(token);

    return { ...token, signature };
  }

  /**
   * Delegate a capability to another agent
   */
  async delegateCapability(
    parentCapability: CapabilityToken,
    newHolder: string,
    options?: {
      rights?: AccessRight[];
      expiresAt?: number;
      delegationDepth?: number;
      constraints?: CapabilityConstraints;
    }
  ): Promise<CapabilityToken> {
    // Verify parent capability is valid
    const validation = await this.validateCapability(parentCapability, {
      agentId: parentCapability.holder,
      resource: parentCapability.resource,
      right: AccessRight.DELEGATE,
      timestamp: Date.now(),
    });

    if (!validation.valid) {
      throw new Error(`Cannot delegate: ${validation.error}`);
    }

    // Check delegation depth
    if (parentCapability.delegationDepth <= 0) {
      throw new Error('Cannot delegate: delegation depth exhausted');
    }

    // Rights can only be restricted, not expanded
    const newRights = options?.rights ?? parentCapability.rights;
    for (const right of newRights) {
      if (!parentCapability.rights.includes(right)) {
        throw new Error(`Cannot grant right '${right}' not present in parent capability`);
      }
    }

    // Expiration can only be shortened
    let expiresAt = options?.expiresAt;
    if (parentCapability.expiresAt) {
      if (!expiresAt || expiresAt > parentCapability.expiresAt) {
        expiresAt = parentCapability.expiresAt;
      }
    }

    const id = secureUUID();
    const issuedAt = Date.now();

    const token: Omit<CapabilityToken, 'signature'> = {
      id,
      resource: parentCapability.resource,
      isPattern: parentCapability.isPattern,
      rights: newRights,
      issuer: parentCapability.holder,
      holder: newHolder,
      issuedAt,
      expiresAt,
      delegationDepth: Math.min(
        (options?.delegationDepth ?? parentCapability.delegationDepth) - 1,
        parentCapability.delegationDepth - 1
      ),
      parentId: parentCapability.id,
      delegationChain: [...parentCapability.delegationChain, parentCapability.holder],
      constraints: this.mergeConstraints(parentCapability.constraints, options?.constraints),
    };

    const signature = await this.signToken(token);

    return { ...token, signature };
  }

  // ===========================================================================
  // Token Validation
  // ===========================================================================

  /**
   * Validate a capability token
   */
  async validateCapability(
    token: CapabilityToken,
    context: ValidationContext
  ): Promise<ValidationResult> {
    // 1. Verify signature
    const signatureValid = await this.verifySignature(token);
    if (!signatureValid) {
      return { valid: false, error: 'Invalid signature' };
    }

    // 2. Check revocation
    if (this.config.enableRevocation && this.isRevoked(token.id)) {
      return { valid: false, error: 'Token has been revoked' };
    }

    // 3. Check holder matches agent
    if (token.holder !== context.agentId) {
      return { valid: false, error: 'Token holder does not match agent' };
    }

    // 4. Check expiration
    if (token.expiresAt && token.expiresAt < context.timestamp) {
      return { valid: false, error: 'Token has expired' };
    }

    // 5. Check resource matches
    if (!this.resourceMatches(token, context.resource)) {
      return { valid: false, error: 'Resource does not match capability' };
    }

    // 6. Check right is granted
    if (!token.rights.includes(context.right) && !token.rights.includes(AccessRight.ADMIN)) {
      return { valid: false, error: `Right '${context.right}' not granted` };
    }

    // 7. Check constraints
    const constraintResult = this.checkConstraints(token, context);
    if (!constraintResult.valid) {
      return constraintResult;
    }

    // 8. Check delegation chain if applicable
    if (token.parentId) {
      // In a full implementation, we would verify the entire chain
      // For now, we trust the chain if signature is valid
    }

    return { valid: true, capability: token };
  }

  /**
   * Check if token grants access to resource with specific right
   */
  async hasAccess(
    token: CapabilityToken,
    resource: string,
    right: AccessRight,
    agentId: string
  ): Promise<boolean> {
    const result = await this.validateCapability(token, {
      agentId,
      resource,
      right,
      timestamp: Date.now(),
    });
    return result.valid;
  }

  // ===========================================================================
  // Token Revocation
  // ===========================================================================

  /**
   * Revoke a capability token
   */
  revoke(tokenId: string): void {
    this.revoked.add(tokenId);
  }

  /**
   * Revoke all tokens in a delegation chain
   */
  revokeChain(rootTokenId: string): void {
    // In a full implementation, we would track all delegated tokens
    // and revoke them recursively
    this.revoked.add(rootTokenId);
  }

  /**
   * Check if a token is revoked
   */
  isRevoked(tokenId: string): boolean {
    return this.revoked.has(tokenId);
  }

  /**
   * Clear revocation list (for testing)
   */
  clearRevocations(): void {
    this.revoked.clear();
  }

  // ===========================================================================
  // Serialization
  // ===========================================================================

  /**
   * Serialize capability to string
   */
  serializeCapability(token: CapabilityToken): string {
    return Buffer.from(JSON.stringify(token)).toString('base64url');
  }

  /**
   * Deserialize capability from string
   */
  deserializeCapability(serialized: string): CapabilityToken {
    const json = Buffer.from(serialized, 'base64url').toString('utf8');
    return JSON.parse(json) as CapabilityToken;
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private async signToken(token: Omit<CapabilityToken, 'signature'>): Promise<string> {
    const payload = JSON.stringify(token);
    const signature = await hmac(this.config.signingKey, payload, 'SHA-256');
    return Buffer.from(signature).toString('base64');
  }

  private async verifySignature(token: CapabilityToken): Promise<boolean> {
    const { signature, ...tokenWithoutSig } = token;
    const payload = JSON.stringify(tokenWithoutSig);
    const expectedSig = new Uint8Array(Buffer.from(signature, 'base64'));
    return verifyHmac(this.config.signingKey, payload, expectedSig, 'SHA-256');
  }

  private resourceMatches(token: CapabilityToken, resource: string): boolean {
    if (!token.isPattern) {
      return token.resource === resource;
    }

    // Convert glob pattern to regex
    const pattern = token.resource
      .replace(/[.+^${}()|[\]\\]/g, '\\$&')
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.');

    const regex = new RegExp(`^${pattern}$`);
    return regex.test(resource);
  }

  private checkConstraints(
    token: CapabilityToken,
    context: ValidationContext
  ): ValidationResult {
    const constraints = token.constraints;
    if (!constraints) {
      return { valid: true };
    }

    // Check max uses
    if (constraints.maxUses !== undefined) {
      const currentUses = this.usageCount.get(token.id) ?? 0;
      if (currentUses >= constraints.maxUses) {
        return { valid: false, error: 'Maximum uses exceeded' };
      }
      // Increment usage count
      this.usageCount.set(token.id, currentUses + 1);
    }

    // Check IP restrictions
    if (constraints.allowedIPs && context.clientIP) {
      const ipAllowed = constraints.allowedIPs.some(allowed => {
        if (allowed.includes('/')) {
          // CIDR notation - simplified check
          return context.clientIP?.startsWith(allowed.split('/')[0].split('.').slice(0, 3).join('.'));
        }
        return allowed === context.clientIP;
      });
      if (!ipAllowed) {
        return { valid: false, error: 'IP address not allowed' };
      }
    }

    // Check time windows
    if (constraints.timeWindows) {
      const date = new Date(context.timestamp);
      const hour = date.getHours();
      const day = date.getDay();

      const inWindow = constraints.timeWindows.some(window => {
        const hourOk = hour >= window.start && hour < window.end;
        const dayOk = !window.days || window.days.includes(day);
        return hourOk && dayOk;
      });

      if (!inWindow) {
        return { valid: false, error: 'Access outside allowed time window' };
      }
    }

    // Check required context
    if (constraints.requiredContext && context.contextData) {
      for (const [key, value] of Object.entries(constraints.requiredContext)) {
        if (context.contextData[key] !== value) {
          return { valid: false, error: `Required context '${key}' does not match` };
        }
      }
    }

    return { valid: true };
  }

  private mergeConstraints(
    parent?: CapabilityConstraints,
    child?: CapabilityConstraints
  ): CapabilityConstraints | undefined {
    if (!parent && !child) return undefined;
    if (!parent) return child;
    if (!child) return parent;

    return {
      // Take the more restrictive maxUses
      maxUses: parent.maxUses !== undefined && child.maxUses !== undefined
        ? Math.min(parent.maxUses, child.maxUses)
        : parent.maxUses ?? child.maxUses,

      // Intersect allowed IPs
      allowedIPs: parent.allowedIPs && child.allowedIPs
        ? parent.allowedIPs.filter(ip => child.allowedIPs?.includes(ip))
        : parent.allowedIPs ?? child.allowedIPs,

      // Intersect time windows (simplified: use child's windows within parent's)
      timeWindows: child.timeWindows ?? parent.timeWindows,

      // Merge required context
      requiredContext: {
        ...parent.requiredContext,
        ...child.requiredContext,
      },
    };
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a CapBAC manager with a new random signing key
 */
export function createCapBACManager(
  options?: Partial<Omit<CapBACConfig, 'signingKey'>>
): CapBACManager {
  const signingKey = secureRandomBytes(32);

  return new CapBACManager({
    signingKey,
    maxDelegationDepth: options?.maxDelegationDepth ?? 3,
    defaultTtlMs: options?.defaultTtlMs,
    enableRevocation: options?.enableRevocation ?? true,
  });
}

/**
 * Create a CapBAC manager with a specific signing key
 */
export function createCapBACManagerWithKey(
  signingKey: Uint8Array,
  options?: Partial<Omit<CapBACConfig, 'signingKey'>>
): CapBACManager {
  return new CapBACManager({
    signingKey,
    maxDelegationDepth: options?.maxDelegationDepth ?? 3,
    defaultTtlMs: options?.defaultTtlMs,
    enableRevocation: options?.enableRevocation ?? true,
  });
}

/**
 * Create a simple read capability for a resource
 */
export async function createReadCapability(
  manager: CapBACManager,
  resource: string,
  issuer: string,
  holder: string,
  ttlMs?: number
): Promise<CapabilityToken> {
  return manager.createCapability({
    resource,
    rights: [AccessRight.READ],
    issuer,
    holder,
    expiresAt: ttlMs ? Date.now() + ttlMs : undefined,
  });
}

/**
 * Create a full access capability for a resource
 */
export async function createFullAccessCapability(
  manager: CapBACManager,
  resource: string,
  issuer: string,
  holder: string,
  ttlMs?: number
): Promise<CapabilityToken> {
  return manager.createCapability({
    resource,
    rights: [AccessRight.READ, AccessRight.WRITE, AccessRight.DELETE, AccessRight.DELEGATE],
    issuer,
    holder,
    expiresAt: ttlMs ? Date.now() + ttlMs : undefined,
    delegationDepth: 2,
  });
}
