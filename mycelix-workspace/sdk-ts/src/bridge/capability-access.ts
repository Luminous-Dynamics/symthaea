/**
 * Capability-Based Access Control for Cross-hApp Messaging
 *
 * Implements fine-grained permission control using capability tokens.
 * Capabilities are unforgeable tokens that grant specific permissions
 * for cross-hApp operations.
 *
 * Features:
 * - Capability creation and granting
 * - Hierarchical capability inheritance
 * - Time-limited and usage-limited capabilities
 * - Capability delegation with attenuation
 * - Revocation support
 */

import { createHash } from 'crypto';

/**
 * Generate a hash of the input string (sync version for capability proofs)
 */
function hashSync(data: string): string {
  return createHash('sha256').update(data).digest('hex');
}

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Capability operations that can be granted
 */
export type CapabilityOperation =
  | 'read'
  | 'write'
  | 'execute'
  | 'delegate'
  | 'admin';

/**
 * Resource types that capabilities can apply to
 */
export type ResourceType =
  | 'identity'
  | 'knowledge'
  | 'governance'
  | 'justice'
  | 'finance'
  | 'property'
  | 'energy'
  | 'media'
  | 'bridge'
  | '*'; // Wildcard for all resources

/**
 * Constraint types for capability limitations
 */
export interface CapabilityConstraints {
  /** Maximum number of times the capability can be used */
  maxUses?: number;
  /** Capability expiration timestamp */
  expiresAt?: number;
  /** List of allowed IP addresses or agent IDs */
  allowedOrigins?: string[];
  /** Specific resource IDs this capability applies to */
  resourceIds?: string[];
  /** Time window constraints (e.g., only valid during business hours) */
  timeWindow?: {
    startHour: number;
    endHour: number;
    timezone?: string;
  };
  /** Rate limiting */
  rateLimit?: {
    maxRequests: number;
    windowMs: number;
  };
}

/**
 * A capability token granting specific permissions
 */
export interface Capability {
  /** Unique identifier for this capability */
  id: string;
  /** The DID of the entity that created this capability */
  issuerDid: string;
  /** The DID of the entity this capability is granted to */
  granteeDid: string;
  /** The resource type this capability applies to */
  resource: ResourceType;
  /** Specific resource ID (optional, if not set applies to all of type) */
  resourceId?: string;
  /** Operations allowed by this capability */
  operations: CapabilityOperation[];
  /** Constraints on this capability */
  constraints?: CapabilityConstraints;
  /** Parent capability ID if this was delegated */
  parentCapabilityId?: string;
  /** Chain of delegation (issuer DIDs from root to current) */
  delegationChain: string[];
  /** When this capability was created */
  createdAt: number;
  /** Cryptographic proof of the capability */
  proof: string;
  /** Whether this capability can be further delegated */
  delegatable: boolean;
  /** Current use count */
  useCount: number;
  /** Whether this capability has been revoked */
  revoked: boolean;
}

/**
 * Request to invoke a capability
 */
export interface CapabilityInvocation {
  /** The capability being invoked */
  capabilityId: string;
  /** The operation being performed */
  operation: CapabilityOperation;
  /** The resource being accessed */
  resource: ResourceType;
  /** Specific resource ID if applicable */
  resourceId?: string;
  /** The invoker's DID */
  invokerDid: string;
  /** Timestamp of invocation */
  timestamp: number;
  /** Additional context for the invocation */
  context?: Record<string, unknown>;
}

/**
 * Result of capability verification
 */
export interface CapabilityVerificationResult {
  /** Whether the capability is valid for the requested operation */
  valid: boolean;
  /** Reason for failure if not valid */
  reason?: string;
  /** The verified capability if valid */
  capability?: Capability;
  /** Remaining uses if limited */
  remainingUses?: number;
  /** Time until expiration if time-limited */
  expiresIn?: number;
}

/**
 * Delegation request for creating derived capabilities
 */
export interface DelegationRequest {
  /** The capability to delegate from */
  parentCapabilityId: string;
  /** The DID to delegate to */
  delegateeDid: string;
  /** Operations to include (must be subset of parent) */
  operations: CapabilityOperation[];
  /** Additional constraints to add (can only be more restrictive) */
  additionalConstraints?: Partial<CapabilityConstraints>;
  /** Whether the delegated capability can be further delegated */
  delegatable: boolean;
}

// ============================================================================
// Capability Manager
// ============================================================================

/**
 * Manages capability creation, storage, and verification
 */
export class CapabilityManager {
  private capabilities: Map<string, Capability> = new Map();
  private revokedCapabilities: Set<string> = new Set();
  private usageTracking: Map<string, { count: number; timestamps: number[] }> =
    new Map();

  /**
   * Create a new root capability
   */
  createCapability(params: {
    issuerDid: string;
    granteeDid: string;
    resource: ResourceType;
    resourceId?: string;
    operations: CapabilityOperation[];
    constraints?: CapabilityConstraints;
    delegatable?: boolean;
  }): Capability {
    const id = this.generateCapabilityId();
    const now = Date.now();

    const capability: Capability = {
      id,
      issuerDid: params.issuerDid,
      granteeDid: params.granteeDid,
      resource: params.resource,
      resourceId: params.resourceId,
      operations: [...params.operations],
      constraints: params.constraints,
      parentCapabilityId: undefined,
      delegationChain: [params.issuerDid],
      createdAt: now,
      proof: this.generateProof(id, params.issuerDid, now),
      delegatable: params.delegatable ?? false,
      useCount: 0,
      revoked: false,
    };

    this.capabilities.set(id, capability);
    return { ...capability };
  }

  /**
   * Delegate a capability to another entity
   */
  delegateCapability(request: DelegationRequest): Capability | null {
    const parent = this.capabilities.get(request.parentCapabilityId);

    if (!parent) {
      return null;
    }

    // Check if parent is delegatable
    if (!parent.delegatable) {
      return null;
    }

    // Check if parent is revoked
    if (parent.revoked || this.revokedCapabilities.has(parent.id)) {
      return null;
    }

    // Verify requested operations are subset of parent
    const validOperations = request.operations.every((op) =>
      parent.operations.includes(op)
    );
    if (!validOperations) {
      return null;
    }

    // Merge constraints (child can only be more restrictive)
    const mergedConstraints = this.mergeConstraints(
      parent.constraints,
      request.additionalConstraints
    );

    const id = this.generateCapabilityId();
    const now = Date.now();

    const delegatedCapability: Capability = {
      id,
      issuerDid: parent.granteeDid, // The grantee becomes the issuer
      granteeDid: request.delegateeDid,
      resource: parent.resource,
      resourceId: parent.resourceId,
      operations: [...request.operations],
      constraints: mergedConstraints,
      parentCapabilityId: parent.id,
      delegationChain: [...parent.delegationChain, parent.granteeDid],
      createdAt: now,
      proof: this.generateProof(id, parent.granteeDid, now),
      delegatable: request.delegatable && parent.delegatable,
      useCount: 0,
      revoked: false,
    };

    this.capabilities.set(id, delegatedCapability);
    return { ...delegatedCapability };
  }

  /**
   * Verify a capability invocation
   */
  verifyCapability(invocation: CapabilityInvocation): CapabilityVerificationResult {
    const capability = this.capabilities.get(invocation.capabilityId);

    // Check capability exists
    if (!capability) {
      return { valid: false, reason: 'Capability not found' };
    }

    // Check not revoked
    if (capability.revoked || this.revokedCapabilities.has(capability.id)) {
      return { valid: false, reason: 'Capability has been revoked' };
    }

    // Check delegation chain for revocations
    if (!this.verifyDelegationChain(capability)) {
      return { valid: false, reason: 'Parent capability in delegation chain has been revoked' };
    }

    // Check grantee matches invoker
    if (capability.granteeDid !== invocation.invokerDid) {
      return { valid: false, reason: 'Invoker does not match capability grantee' };
    }

    // Check operation is allowed
    if (!capability.operations.includes(invocation.operation)) {
      return {
        valid: false,
        reason: `Operation '${invocation.operation}' not allowed by this capability`,
      };
    }

    // Check resource matches
    if (
      capability.resource !== '*' &&
      capability.resource !== invocation.resource
    ) {
      return {
        valid: false,
        reason: `Resource '${invocation.resource}' not covered by this capability`,
      };
    }

    // Check resource ID if specified
    if (
      capability.resourceId &&
      invocation.resourceId &&
      capability.resourceId !== invocation.resourceId
    ) {
      return { valid: false, reason: 'Resource ID does not match capability' };
    }

    // Check constraints
    const constraintResult = this.checkConstraints(
      capability,
      invocation.timestamp
    );
    if (!constraintResult.valid) {
      return constraintResult;
    }

    // Update usage
    this.recordUsage(capability.id);
    capability.useCount++;

    return {
      valid: true,
      capability: { ...capability },
      remainingUses: capability.constraints?.maxUses
        ? capability.constraints.maxUses - capability.useCount
        : undefined,
      expiresIn: capability.constraints?.expiresAt
        ? capability.constraints.expiresAt - Date.now()
        : undefined,
    };
  }

  /**
   * Revoke a capability and all its delegated children
   */
  revokeCapability(capabilityId: string, revokerDid: string): boolean {
    const capability = this.capabilities.get(capabilityId);

    if (!capability) {
      return false;
    }

    // Check revoker has authority (must be issuer or in delegation chain)
    if (
      capability.issuerDid !== revokerDid &&
      !capability.delegationChain.includes(revokerDid)
    ) {
      return false;
    }

    // Revoke this capability
    capability.revoked = true;
    this.revokedCapabilities.add(capabilityId);

    // Revoke all children
    this.revokeChildren(capabilityId);

    return true;
  }

  /**
   * Get all capabilities for a grantee
   */
  getCapabilitiesForGrantee(granteeDid: string): Capability[] {
    const results: Capability[] = [];
    for (const capability of this.capabilities.values()) {
      if (capability.granteeDid === granteeDid && !capability.revoked) {
        results.push({ ...capability });
      }
    }
    return results;
  }

  /**
   * Get a specific capability by ID
   */
  getCapability(id: string): Capability | undefined {
    const cap = this.capabilities.get(id);
    return cap ? { ...cap } : undefined;
  }

  /**
   * Check if an entity has a specific capability
   */
  hasCapability(
    granteeDid: string,
    resource: ResourceType,
    operation: CapabilityOperation,
    resourceId?: string
  ): boolean {
    for (const capability of this.capabilities.values()) {
      if (
        capability.granteeDid === granteeDid &&
        !capability.revoked &&
        (capability.resource === '*' || capability.resource === resource) &&
        capability.operations.includes(operation) &&
        (!resourceId ||
          !capability.resourceId ||
          capability.resourceId === resourceId)
      ) {
        // Verify constraints at current time
        const constraintResult = this.checkConstraints(capability, Date.now());
        if (constraintResult.valid) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Clear all capabilities (for testing)
   */
  clear(): void {
    this.capabilities.clear();
    this.revokedCapabilities.clear();
    this.usageTracking.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private generateCapabilityId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).slice(2, 10);
    return `cap-${timestamp}-${random}`;
  }

  private generateProof(
    capabilityId: string,
    issuerDid: string,
    timestamp: number
  ): string {
    const data = `${capabilityId}:${issuerDid}:${timestamp}`;
    return hashSync(data);
  }

  private mergeConstraints(
    parent?: CapabilityConstraints,
    additional?: Partial<CapabilityConstraints>
  ): CapabilityConstraints | undefined {
    if (!parent && !additional) {
      return undefined;
    }

    const merged: CapabilityConstraints = {};

    // Take the more restrictive maxUses
    if (parent?.maxUses || additional?.maxUses) {
      merged.maxUses = Math.min(
        parent?.maxUses ?? Infinity,
        additional?.maxUses ?? Infinity
      );
    }

    // Take the earlier expiration
    if (parent?.expiresAt || additional?.expiresAt) {
      merged.expiresAt = Math.min(
        parent?.expiresAt ?? Infinity,
        additional?.expiresAt ?? Infinity
      );
    }

    // Intersect allowed origins
    if (parent?.allowedOrigins || additional?.allowedOrigins) {
      const parentOrigins = new Set(parent?.allowedOrigins ?? []);
      const additionalOrigins = additional?.allowedOrigins ?? [];
      if (parentOrigins.size > 0 && additionalOrigins.length > 0) {
        merged.allowedOrigins = additionalOrigins.filter((o) =>
          parentOrigins.has(o)
        );
      } else {
        merged.allowedOrigins =
          parent?.allowedOrigins ?? additional?.allowedOrigins;
      }
    }

    // Intersect resource IDs
    if (parent?.resourceIds || additional?.resourceIds) {
      const parentIds = new Set(parent?.resourceIds ?? []);
      const additionalIds = additional?.resourceIds ?? [];
      if (parentIds.size > 0 && additionalIds.length > 0) {
        merged.resourceIds = additionalIds.filter((id) => parentIds.has(id));
      } else {
        merged.resourceIds = parent?.resourceIds ?? additional?.resourceIds;
      }
    }

    // Take the more restrictive time window
    if (parent?.timeWindow || additional?.timeWindow) {
      const parentWindow = parent?.timeWindow;
      const additionalWindow = additional?.timeWindow;
      if (parentWindow && additionalWindow) {
        merged.timeWindow = {
          startHour: Math.max(parentWindow.startHour, additionalWindow.startHour),
          endHour: Math.min(parentWindow.endHour, additionalWindow.endHour),
          timezone: additionalWindow.timezone ?? parentWindow.timezone,
        };
      } else {
        merged.timeWindow = parentWindow ?? additionalWindow;
      }
    }

    // Take the more restrictive rate limit
    if (parent?.rateLimit || additional?.rateLimit) {
      const parentLimit = parent?.rateLimit;
      const additionalLimit = additional?.rateLimit;
      if (parentLimit && additionalLimit) {
        merged.rateLimit = {
          maxRequests: Math.min(
            parentLimit.maxRequests,
            additionalLimit.maxRequests
          ),
          windowMs: Math.min(parentLimit.windowMs, additionalLimit.windowMs),
        };
      } else {
        merged.rateLimit = parentLimit ?? additionalLimit;
      }
    }

    return Object.keys(merged).length > 0 ? merged : undefined;
  }

  private checkConstraints(
    capability: Capability,
    timestamp: number
  ): CapabilityVerificationResult {
    const constraints = capability.constraints;
    if (!constraints) {
      return { valid: true };
    }

    // Check max uses
    if (constraints.maxUses !== undefined) {
      if (capability.useCount >= constraints.maxUses) {
        return { valid: false, reason: 'Capability has exceeded maximum uses' };
      }
    }

    // Check expiration
    if (constraints.expiresAt !== undefined) {
      if (timestamp >= constraints.expiresAt) {
        return { valid: false, reason: 'Capability has expired' };
      }
    }

    // Check time window
    if (constraints.timeWindow) {
      const date = new Date(timestamp);
      const hour = date.getHours();
      if (
        hour < constraints.timeWindow.startHour ||
        hour >= constraints.timeWindow.endHour
      ) {
        return {
          valid: false,
          reason: 'Capability not valid during current time window',
        };
      }
    }

    // Check rate limit
    if (constraints.rateLimit) {
      const usage = this.usageTracking.get(capability.id);
      if (usage) {
        const windowStart = timestamp - constraints.rateLimit.windowMs;
        const recentUses = usage.timestamps.filter((t) => t >= windowStart);
        if (recentUses.length >= constraints.rateLimit.maxRequests) {
          return { valid: false, reason: 'Rate limit exceeded' };
        }
      }
    }

    return { valid: true };
  }

  private recordUsage(capabilityId: string): void {
    const now = Date.now();
    let usage = this.usageTracking.get(capabilityId);
    if (!usage) {
      usage = { count: 0, timestamps: [] };
      this.usageTracking.set(capabilityId, usage);
    }
    usage.count++;
    usage.timestamps.push(now);

    // Keep only last hour of timestamps for rate limiting
    const oneHourAgo = now - 3600000;
    usage.timestamps = usage.timestamps.filter((t) => t >= oneHourAgo);
  }

  private verifyDelegationChain(capability: Capability): boolean {
    // Check if any capability in the delegation chain has been revoked
    let currentId = capability.parentCapabilityId;
    while (currentId) {
      if (this.revokedCapabilities.has(currentId)) {
        return false;
      }
      const parent = this.capabilities.get(currentId);
      if (!parent) {
        return false;
      }
      if (parent.revoked) {
        return false;
      }
      currentId = parent.parentCapabilityId;
    }
    return true;
  }

  private revokeChildren(parentId: string): void {
    for (const capability of this.capabilities.values()) {
      if (capability.parentCapabilityId === parentId && !capability.revoked) {
        capability.revoked = true;
        this.revokedCapabilities.add(capability.id);
        this.revokeChildren(capability.id);
      }
    }
  }
}

// ============================================================================
// Capability-Guarded Bridge
// ============================================================================

/**
 * A bridge wrapper that enforces capability-based access control
 */
export class CapabilityGuardedBridge {
  constructor(
    private capabilityManager: CapabilityManager,
    private ownerDid: string
  ) {}

  /**
   * Execute a cross-hApp operation with capability verification
   */
  async executeWithCapability<T>(
    invocation: CapabilityInvocation,
    operation: () => Promise<T>
  ): Promise<{ success: boolean; result?: T; error?: string }> {
    const verification = this.capabilityManager.verifyCapability(invocation);

    if (!verification.valid) {
      return {
        success: false,
        error: `Access denied: ${verification.reason}`,
      };
    }

    try {
      const result = await operation();
      return { success: true, result };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Grant a capability to another entity
   */
  grantCapability(params: {
    granteeDid: string;
    resource: ResourceType;
    resourceId?: string;
    operations: CapabilityOperation[];
    constraints?: CapabilityConstraints;
    delegatable?: boolean;
  }): Capability {
    return this.capabilityManager.createCapability({
      issuerDid: this.ownerDid,
      ...params,
    });
  }

  /**
   * Revoke a previously granted capability
   */
  revokeCapability(capabilityId: string): boolean {
    return this.capabilityManager.revokeCapability(capabilityId, this.ownerDid);
  }

  /**
   * Check if the bridge owner has a specific capability
   */
  canAccess(
    resource: ResourceType,
    operation: CapabilityOperation,
    resourceId?: string
  ): boolean {
    return this.capabilityManager.hasCapability(
      this.ownerDid,
      resource,
      operation,
      resourceId
    );
  }

  /**
   * Get all capabilities granted to this bridge's owner
   */
  getMyCapabilities(): Capability[] {
    return this.capabilityManager.getCapabilitiesForGrantee(this.ownerDid);
  }
}

// ============================================================================
// Predefined Capability Templates
// ============================================================================

/**
 * Common capability templates for quick creation
 */
export const CapabilityTemplates = {
  /** Read-only access to a resource */
  readOnly: (): CapabilityOperation[] => ['read'],

  /** Read and write access */
  readWrite: (): CapabilityOperation[] => [
    'read',
    'write',
  ],

  /** Full access including execution */
  fullAccess: (): CapabilityOperation[] => [
    'read',
    'write',
    'execute',
  ],

  /** Admin access with delegation */
  admin: (): CapabilityOperation[] => [
    'read',
    'write',
    'execute',
    'delegate',
    'admin',
  ],

  /** Time-limited capability (1 hour) */
  oneHourConstraint: (): CapabilityConstraints => ({
    expiresAt: Date.now() + 3600000,
  }),

  /** Time-limited capability (24 hours) */
  oneDayConstraint: (): CapabilityConstraints => ({
    expiresAt: Date.now() + 86400000,
  }),

  /** Single-use capability */
  singleUse: (): CapabilityConstraints => ({
    maxUses: 1,
  }),

  /** Business hours only (9 AM - 5 PM) */
  businessHoursOnly: (): CapabilityConstraints => ({
    timeWindow: {
      startHour: 9,
      endHour: 17,
    },
  }),

  /** Rate limited (10 requests per minute) */
  rateLimited: (): CapabilityConstraints => ({
    rateLimit: {
      maxRequests: 10,
      windowMs: 60000,
    },
  }),
};

// ============================================================================
// Singleton Factory Functions
// ============================================================================

let capabilityManagerInstance: CapabilityManager | null = null;

/**
 * Get the singleton CapabilityManager instance
 */
export function getCapabilityManager(): CapabilityManager {
  if (!capabilityManagerInstance) {
    capabilityManagerInstance = new CapabilityManager();
  }
  return capabilityManagerInstance;
}

/**
 * Create a capability-guarded bridge for a specific owner
 */
export function createCapabilityGuardedBridge(
  ownerDid: string,
  manager?: CapabilityManager
): CapabilityGuardedBridge {
  return new CapabilityGuardedBridge(
    manager ?? getCapabilityManager(),
    ownerDid
  );
}

/**
 * Reset the singleton (for testing)
 */
export function resetCapabilityManager(): void {
  if (capabilityManagerInstance) {
    capabilityManagerInstance.clear();
  }
  capabilityManagerInstance = null;
}
