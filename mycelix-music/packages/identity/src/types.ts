// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/identity - Core Types
 *
 * The Mycelix Identity represents a unified soul across the ecosystem.
 * Unlike traditional accounts, a Mycelix Soul is:
 * - Persistent across all Mycelix applications
 * - Enriched by experiences (Proof of Presence)
 * - Connected through the Mycelium network
 * - Sovereign and self-custodial
 */

// ============================================================================
// Core Identity Types
// ============================================================================

/**
 * The fundamental identity in the Mycelix ecosystem.
 * A Soul is non-transferable and represents authentic presence.
 */
export interface MycelixSoul {
  /** Unique soul identifier (derived from wallet or created via account abstraction) */
  id: string;

  /** The primary wallet address (if connected) */
  address?: `0x${string}`;

  /** Human-readable identity */
  profile: SoulProfile;

  /** Soul creation timestamp */
  bornAt: Date;

  /** Total resonance accumulated across all experiences */
  totalResonance: number;

  /** Network connections count */
  connections: number;

  /** Verified credentials and attestations */
  credentials: SoulCredential[];

  /** Active sessions across Mycelix apps */
  sessions: SoulSession[];
}

/**
 * Human-readable profile information
 */
export interface SoulProfile {
  /** Display name */
  name: string;

  /** Optional avatar (IPFS URI or URL) */
  avatar?: string;

  /** Short bio */
  bio?: string;

  /** Soul color (used in UI representations) */
  color: string;

  /** Preferred genres (for Music app) */
  genres?: string[];

  /** Location (optional, for contextual features) */
  location?: {
    city?: string;
    country?: string;
    timezone?: string;
  };

  /** Social links */
  links?: {
    website?: string;
    twitter?: string;
    instagram?: string;
    soundcloud?: string;
    spotify?: string;
  };
}

/**
 * Soulbound credentials that cannot be transferred
 */
export interface SoulCredential {
  /** Credential type */
  type: CredentialType;

  /** Issuer (app or contract that issued it) */
  issuer: string;

  /** When it was issued */
  issuedAt: Date;

  /** Optional expiry */
  expiresAt?: Date;

  /** Credential-specific data */
  data: Record<string, unknown>;

  /** On-chain proof (if applicable) */
  proof?: {
    chainId: number;
    contractAddress: `0x${string}`;
    tokenId: string;
  };
}

export type CredentialType =
  | 'artist_verified'      // Verified artist status
  | 'patron_tier'          // Patronage level
  | 'presence_token'       // Proof of Presence
  | 'collaboration_credit' // Collaboration participation
  | 'milestone_achieved'   // Artist milestone
  | 'mentor_status'        // Approved mentor
  | 'residency_alumni'     // Completed residency
  | 'early_adopter'        // Early platform participant
  | 'community_leader';    // Community leadership role

/**
 * Active session in a Mycelix application
 */
export interface SoulSession {
  /** Session ID */
  id: string;

  /** Application identifier */
  appId: MycelixApp;

  /** Device/client identifier */
  clientId: string;

  /** Session start time */
  startedAt: Date;

  /** Last activity timestamp */
  lastActiveAt: Date;

  /** Session metadata */
  metadata?: Record<string, unknown>;
}

export type MycelixApp =
  | 'music'       // Mycelix Music
  | 'studio'      // Collaborative Studio
  | 'market'      // Marketplace
  | 'community'   // Community Hub
  | 'governance'; // DAO Governance

// ============================================================================
// Wallet Types
// ============================================================================

export type WalletProvider =
  | 'metamask'
  | 'walletconnect'
  | 'coinbase'
  | 'rainbow'
  | 'phantom'
  | 'email'           // Email-based account abstraction
  | 'social'          // Social login account abstraction
  | 'passkey';        // WebAuthn passkey

export interface WalletConnection {
  /** Provider used to connect */
  provider: WalletProvider;

  /** Connected address */
  address: `0x${string}`;

  /** Chain ID */
  chainId: number;

  /** ENS name if available */
  ensName?: string;

  /** ENS avatar if available */
  ensAvatar?: string;

  /** Is this an account abstraction wallet? */
  isSmartAccount: boolean;

  /** Connection timestamp */
  connectedAt: Date;
}

export interface WalletState {
  /** Current connection status */
  status: 'disconnected' | 'connecting' | 'connected' | 'error';

  /** Active wallet connection */
  connection: WalletConnection | null;

  /** Error if status is 'error' */
  error?: string;

  /** Available chains */
  supportedChains: ChainConfig[];
}

export interface ChainConfig {
  id: number;
  name: string;
  network: string;
  nativeCurrency: {
    name: string;
    symbol: string;
    decimals: number;
  };
  rpcUrls: string[];
  blockExplorers?: {
    name: string;
    url: string;
  }[];
  contracts?: {
    soulRegistry?: `0x${string}`;
    presenceToken?: `0x${string}`;
    patronageRegistry?: `0x${string}`;
    sporeToken?: `0x${string}`;
    nutrientFlow?: `0x${string}`;
  };
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface AuthState {
  /** Is user authenticated? */
  isAuthenticated: boolean;

  /** Current soul (if authenticated) */
  soul: MycelixSoul | null;

  /** Authentication method used */
  authMethod: AuthMethod | null;

  /** Loading state */
  isLoading: boolean;

  /** Auth error */
  error?: string;
}

export type AuthMethod =
  | { type: 'wallet'; provider: WalletProvider }
  | { type: 'email'; email: string }
  | { type: 'social'; provider: 'google' | 'apple' | 'discord' }
  | { type: 'passkey'; credentialId: string };

export interface SignInOptions {
  /** Preferred authentication method */
  method?: AuthMethod['type'];

  /** Specific wallet provider (if method is 'wallet') */
  walletProvider?: WalletProvider;

  /** Social provider (if method is 'social') */
  socialProvider?: 'google' | 'apple' | 'discord';

  /** Redirect URL after authentication */
  redirectUrl?: string;

  /** Application requesting auth */
  appId: MycelixApp;
}

export interface SignMessageRequest {
  /** Message to sign */
  message: string;

  /** Purpose of the signature */
  purpose: 'authenticate' | 'authorize' | 'verify';

  /** Requesting application */
  appId: MycelixApp;
}

// ============================================================================
// Cross-App Communication
// ============================================================================

export interface CrossAppMessage {
  /** Message type */
  type: CrossAppMessageType;

  /** Source application */
  from: MycelixApp;

  /** Target application (or 'broadcast' for all) */
  to: MycelixApp | 'broadcast';

  /** Message payload */
  payload: unknown;

  /** Timestamp */
  timestamp: Date;

  /** Message ID for correlation */
  id: string;
}

export type CrossAppMessageType =
  | 'soul:updated'           // Soul profile changed
  | 'wallet:connected'       // Wallet connected
  | 'wallet:disconnected'    // Wallet disconnected
  | 'presence:earned'        // New Proof of Presence
  | 'credential:issued'      // New credential
  | 'session:started'        // Session started in an app
  | 'session:ended'          // Session ended
  | 'network:connection';    // New mycelium connection

// ============================================================================
// Event Types
// ============================================================================

export type IdentityEvent =
  | { type: 'soul:created'; soul: MycelixSoul }
  | { type: 'soul:updated'; soul: MycelixSoul; changes: Partial<SoulProfile> }
  | { type: 'wallet:connected'; connection: WalletConnection }
  | { type: 'wallet:disconnected'; address: `0x${string}` }
  | { type: 'wallet:chainChanged'; chainId: number }
  | { type: 'credential:added'; credential: SoulCredential }
  | { type: 'session:started'; session: SoulSession }
  | { type: 'session:ended'; sessionId: string }
  | { type: 'resonance:updated'; total: number; delta: number }
  | { type: 'error'; code: string; message: string };

export type IdentityEventHandler = (event: IdentityEvent) => void;
