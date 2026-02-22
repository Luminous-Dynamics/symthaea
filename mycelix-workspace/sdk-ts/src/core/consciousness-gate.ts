/**
 * Consciousness Gate Error Handling & Middleware
 *
 * Provides structured error types for consciousness gate rejections,
 * pre-flight eligibility checks, and retry-on-expired middleware.
 *
 * @module @mycelix/sdk/core/consciousness-gate
 */

import { SdkError, SdkErrorCode } from './errors';

// ============================================================================
// Types
// ============================================================================

/** Consciousness tier levels matching Rust ConsciousnessTier enum */
export type ConsciousnessTier =
  | 'Observer'
  | 'Participant'
  | 'Citizen'
  | 'Steward'
  | 'Guardian';

/** 4-dimensional consciousness profile */
export interface ConsciousnessProfile {
  identity: number;
  reputation: number;
  community: number;
  engagement: number;
}

/** Time-limited consciousness credential */
export interface ConsciousnessCredential {
  did: string;
  profile: ConsciousnessProfile;
  tier: ConsciousnessTier;
  issued_at: number;
  expires_at: number;
  issuer: string;
}

/** Structured rejection details from a consciousness gate */
export interface ConsciousnessGateRejection {
  /** The action that was attempted */
  action: string;
  /** The zome where the gate check occurred */
  zome: string;
  /** The agent's actual consciousness tier */
  actualTier: ConsciousnessTier;
  /** The minimum tier required by the action */
  requiredTier: ConsciousnessTier;
  /** Vote weight in basis points (0-10000) */
  weightBp: number;
  /** Specific rejection reasons */
  reasons: string[];
  /** Whether the credential was expired */
  expired: boolean;
}

/** Result of a pre-flight gate eligibility check */
export interface GateEligibility {
  /** Whether the agent meets all requirements */
  eligible: boolean;
  /** The agent's current consciousness tier */
  tier: ConsciousnessTier;
  /** Specific reasons if ineligible */
  reasons: string[];
  /** Whether the credential is nearing expiry */
  nearingExpiry: boolean;
}

/** Audit entry from a governance gate decision */
export interface GateAuditEntry {
  /** The extern function that triggered the gate check */
  action_name: string;
  /** The zome that performed the check */
  zome_name: string;
  /** Whether the agent met all requirements */
  eligible: boolean;
  /** The agent's derived consciousness tier (Debug-formatted) */
  actual_tier: string;
  /** The minimum tier required (Debug-formatted) */
  required_tier: string;
  /** Progressive vote weight in basis points (0-10000) */
  weight_bp: number;
  /** Optional correlation ID for cross-cluster audit trail */
  correlation_id?: string;
}

/** Filter for querying governance gate audit events */
export interface GovernanceAuditFilter {
  /** Filter by the gated action name */
  action_name?: string;
  /** Filter by the zome that performed the check */
  zome_name?: string;
  /** Filter by eligibility outcome */
  eligible?: boolean;
  /** Start of time range (inclusive), microseconds since epoch */
  from_us?: number;
  /** End of time range (inclusive), microseconds since epoch */
  to_us?: number;
}

/** Result of a governance audit query */
export interface GovernanceAuditResult {
  /** Matched audit entries */
  entries: GateAuditEntry[];
  /** Number of entries that matched the filter */
  total_matched: number;
}

/** Interface for zome-callable clients */
export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// ConsciousnessGateError
// ============================================================================

/**
 * Specialized error for consciousness gate rejections.
 *
 * Parses the structured error message from Rust `require_consciousness()`
 * rejections and provides typed access to rejection details.
 *
 * @example
 * ```typescript
 * try {
 *   await client.callZome({ ... });
 * } catch (e) {
 *   const gateErr = ConsciousnessGateError.fromWasmError(e);
 *   if (gateErr?.rejection.expired) {
 *     // Credential expired — trigger refresh and retry
 *   }
 * }
 * ```
 */
export class ConsciousnessGateError extends SdkError {
  public readonly rejection: ConsciousnessGateRejection;

  constructor(message: string, rejection: ConsciousnessGateRejection) {
    super(SdkErrorCode.FORBIDDEN, message, {
      action: rejection.action,
      zomeName: rejection.zome,
      actualTier: rejection.actualTier,
      requiredTier: rejection.requiredTier,
    });
    this.name = 'ConsciousnessGateError';
    this.rejection = rejection;
  }

  /**
   * Attempt to parse a WASM/zome error into a ConsciousnessGateError.
   *
   * Returns null if the error is not a consciousness gate rejection.
   * Matches the pattern from Rust:
   *   "Consciousness gate: tier <tier> insufficient. Reasons: ..."
   *   or "Credential expired at ..."
   */
  static fromWasmError(error: unknown): ConsciousnessGateError | null {
    const msg = error instanceof Error ? error.message : String(error);

    // Match consciousness gate rejection pattern
    const gateMatch = msg.match(
      /Consciousness gate:\s*tier\s+(\w+)\s+insufficient\.\s*Reasons:\s*(.*)/i
    );

    if (gateMatch) {
      const actualTier = gateMatch[1] as ConsciousnessTier;
      const reasonsStr = gateMatch[2];
      const reasons = reasonsStr
        .split(',')
        .map((r) => r.trim())
        .filter(Boolean);
      const expired = reasons.some((r) => r.toLowerCase().includes('expired'));

      // Try to extract the required tier from reasons
      const tierMatch = reasonsStr.match(/below required (\w+)/);
      const requiredTier = (tierMatch?.[1] ?? 'Unknown') as ConsciousnessTier;

      // Try to extract weight from reasons
      const weightMatch = reasonsStr.match(/weight:\s*(\d+)\s*bp/i);
      const weightBp = weightMatch ? parseInt(weightMatch[1], 10) : 0;

      // Try to extract action/zome from context
      const actionMatch = msg.match(/(\w+)\s*requires/i);
      const action = actionMatch?.[1] ?? 'unknown';

      return new ConsciousnessGateError(msg, {
        action,
        zome: 'unknown',
        actualTier,
        requiredTier,
        weightBp,
        reasons,
        expired,
      });
    }

    // Match expired credential pattern
    const expiredMatch = msg.match(/Credential expired at (\d+)/i);
    if (expiredMatch) {
      return new ConsciousnessGateError(msg, {
        action: 'unknown',
        zome: 'unknown',
        actualTier: 'Observer',
        requiredTier: 'Participant',
        weightBp: 0,
        reasons: [msg],
        expired: true,
      });
    }

    return null;
  }
}

// ============================================================================
// Pre-flight eligibility check
// ============================================================================

/** Tier ordering for comparison */
const TIER_ORDER: Record<ConsciousnessTier, number> = {
  Observer: 0,
  Participant: 1,
  Citizen: 2,
  Steward: 3,
  Guardian: 4,
};

/** Minimum combined scores for each tier */
const TIER_THRESHOLDS: Record<ConsciousnessTier, number> = {
  Observer: 0.0,
  Participant: 0.3,
  Citizen: 0.4,
  Steward: 0.6,
  Guardian: 0.8,
};

/** Refresh window: 2 hours in microseconds */
const REFRESH_WINDOW_US = 7_200_000_000;

/**
 * Compute the combined consciousness score from a profile.
 *
 * Weights: identity 25%, reputation 25%, community 30%, engagement 20%
 */
export function combinedScore(profile: ConsciousnessProfile): number {
  return (
    profile.identity * 0.25 +
    profile.reputation * 0.25 +
    profile.community * 0.3 +
    profile.engagement * 0.2
  );
}

/**
 * Derive the consciousness tier from a combined score.
 */
export function tierFromScore(score: number): ConsciousnessTier {
  if (score >= 0.8) return 'Guardian';
  if (score >= 0.6) return 'Steward';
  if (score >= 0.4) return 'Citizen';
  if (score >= 0.3) return 'Participant';
  return 'Observer';
}

/**
 * Pre-flight check: can the current agent perform an action requiring
 * a specific consciousness tier?
 *
 * Fetches the agent's credential from the bridge and evaluates locally.
 * This mirrors the Rust `evaluate_governance` logic in TypeScript.
 *
 * @param client - Holochain AppClient
 * @param bridgeZome - Bridge zome name (e.g., "commons_bridge")
 * @param roleName - DNA role name (e.g., "commons")
 * @param requiredTier - Minimum tier for the action
 * @returns Eligibility result with tier and reasons
 */
export async function canPerform(
  client: ZomeCallable,
  bridgeZome: string,
  roleName: string,
  requiredTier: ConsciousnessTier
): Promise<GateEligibility> {
  try {
    const credential = (await client.callZome({
      role_name: roleName,
      zome_name: bridgeZome,
      fn_name: 'get_consciousness_credential',
      payload: null,
    })) as ConsciousnessCredential;

    const now = Date.now() * 1000; // Convert to microseconds
    const reasons: string[] = [];

    // Check expiry
    if (now >= credential.expires_at) {
      return {
        eligible: false,
        tier: credential.tier,
        reasons: ['Credential expired — refresh needed'],
        nearingExpiry: true,
      };
    }

    // Evaluate tier
    const score = combinedScore(credential.profile);
    const actualTier = tierFromScore(score);

    if (TIER_ORDER[actualTier] < TIER_ORDER[requiredTier]) {
      reasons.push(
        `Tier ${actualTier} below required ${requiredTier} (score ${score.toFixed(3)}, need >= ${TIER_THRESHOLDS[requiredTier].toFixed(3)})`
      );
    }

    const nearingExpiry =
      credential.expires_at > now &&
      credential.expires_at - now < REFRESH_WINDOW_US;

    return {
      eligible: reasons.length === 0,
      tier: actualTier,
      reasons,
      nearingExpiry,
    };
  } catch (err) {
    return {
      eligible: false,
      tier: 'Observer',
      reasons: [`Failed to check credential: ${err}`],
      nearingExpiry: false,
    };
  }
}

// ============================================================================
// Retry-on-expired middleware
// ============================================================================

/**
 * Wrap a zome call with automatic retry on expired credential.
 *
 * If the call fails with a consciousness gate error where the credential
 * is expired, triggers a credential refresh via the bridge and retries once.
 *
 * @param call - The zome call to execute
 * @param refreshFn - Function that refreshes the consciousness credential
 * @returns The result of the zome call
 *
 * @example
 * ```typescript
 * const result = await withGateRetry(
 *   () => client.callZome({ ... }),
 *   () => bridge.refreshCredential()
 * );
 * ```
 */
export async function withGateRetry<T>(
  call: () => Promise<T>,
  refreshFn: () => Promise<void>
): Promise<T> {
  try {
    return await call();
  } catch (e) {
    const gateErr = ConsciousnessGateError.fromWasmError(e);
    if (gateErr?.rejection.expired) {
      await refreshFn();
      return await call(); // Single retry after refresh
    }
    throw e;
  }
}

// ============================================================================
// Governance audit query
// ============================================================================

/**
 * Query governance gate audit events from a bridge zome.
 *
 * Calls the bridge's `get_governance_audit_trail` extern to retrieve
 * typed audit entries matching the given filter.
 *
 * @param client - Holochain AppClient
 * @param bridgeZome - Bridge zome name (e.g., "commons_bridge")
 * @param roleName - DNA role name (e.g., "commons")
 * @param filter - Filter criteria for audit events
 * @returns Typed audit result with matched entries
 */
export async function queryGovernanceAudit(
  client: ZomeCallable,
  bridgeZome: string,
  roleName: string,
  filter: GovernanceAuditFilter
): Promise<GovernanceAuditResult> {
  const result = (await client.callZome({
    role_name: roleName,
    zome_name: bridgeZome,
    fn_name: 'get_governance_audit_trail',
    payload: filter,
  })) as GovernanceAuditResult;
  return result;
}
