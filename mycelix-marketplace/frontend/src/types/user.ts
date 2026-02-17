/**
 * User Type Definitions
 *
 * Core data structures for user profiles and authentication
 */

/**
 * User role in the marketplace
 */
export type UserRole = 'buyer' | 'seller' | 'arbitrator' | 'admin';

/**
 * Complete user profile
 */
export interface UserProfile {
  /** User agent ID (Holochain) */
  agent_id: string;

  /** Username/display name */
  username: string;

  /** Email address */
  email?: string;

  /** Profile bio */
  bio?: string;

  /** Avatar IPFS CID */
  avatar_cid?: string;

  /** PoGQ trust score (0-1 or 0-100) */
  trust_score: number;

  /** User roles */
  roles: UserRole[];

  /** Account creation timestamp */
  member_since: number;

  /** Last active timestamp */
  last_active?: number;

  /** Total listings created */
  total_listings: number;

  /** Total sales completed */
  total_sales: number;

  /** Total purchases made */
  total_purchases: number;

  /** Total reviews received */
  total_reviews: number;

  /** Average rating received (1-5) */
  average_rating: number;

  /** Whether profile is verified */
  is_verified?: boolean;
}

/**
 * User statistics for dashboard
 */
export interface UserStats {
  /** Number of active listings */
  active_listings: number;

  /** Number of sold listings */
  sold_listings: number;

  /** Number of pending transactions (as buyer) */
  pending_purchases: number;

  /** Number of pending transactions (as seller) */
  pending_sales: number;

  /** Number of disputes filed */
  disputes_filed: number;

  /** Number of reviews given */
  reviews_given: number;

  /** Number of reviews received */
  reviews_received: number;
}

/**
 * Trust score breakdown for PoGQ visualization
 */
export interface TrustBreakdown {
  /** Total completed transactions */
  transactionCount: number;

  /** Number of positive reviews */
  positiveReviews: number;

  /** Average rating received */
  averageRating: number;

  /** Account age in days */
  accountAge: number;

  /** Member since timestamp */
  memberSince: number;

  /** Response time (hours) */
  averageResponseTime?: number;

  /** Dispute resolution rate */
  disputeResolutionRate?: number;
}

/**
 * User authentication state
 */
export interface AuthState {
  /** Whether user is authenticated */
  isAuthenticated: boolean;

  /** Current user profile */
  user: UserProfile | null;

  /** Authentication token */
  token?: string;

  /** Token expiration timestamp */
  tokenExpiry?: number;
}

/**
 * User preferences
 */
export interface UserPreferences {
  /** Email notifications enabled */
  emailNotifications: boolean;

  /** Browser notifications enabled */
  browserNotifications: boolean;

  /** Default currency */
  currency: 'USD' | 'EUR' | 'BTC' | 'ETH';

  /** Theme preference */
  theme: 'light' | 'dark' | 'auto';

  /** Language preference */
  language: string;
}
