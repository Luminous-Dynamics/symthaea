/**
 * Listing Type Definitions
 *
 * Core data structures for marketplace listings
 */

/**
 * Category options for listings
 */
export type ListingCategory =
  | 'Electronics'
  | 'Fashion'
  | 'Home & Garden'
  | 'Sports & Outdoors'
  | 'Books & Media'
  | 'Toys & Games'
  | 'Health & Beauty'
  | 'Automotive'
  | 'Art & Collectibles'
  | 'Other';

/**
 * Listing status
 */
export type ListingStatus = 'active' | 'sold' | 'inactive' | 'deleted';

/**
 * Complete listing data structure
 */
export interface Listing {
  /** Unique listing identifier (Holochain action hash) */
  id: string;

  /** Listing title */
  title: string;

  /** Detailed description */
  description: string;

  /** Price in USD */
  price: number;

  /** Product category */
  category: ListingCategory;

  /** IPFS content identifiers for product photos */
  photos_ipfs_cids: string[];

  /** Seller's agent ID */
  seller_agent_id: string;

  /** Listing creation timestamp (milliseconds since epoch) */
  created_at: number;

  /** Listing last update timestamp */
  updated_at?: number;

  /** Current status */
  status: ListingStatus;

  /** Number of items available */
  quantity_available?: number;

  /** View count */
  views?: number;

  /** Listing hash (for Holochain DHT) */
  listing_hash?: string;
}

/**
 * Seller information associated with a listing
 */
export interface SellerInfo {
  /** Agent ID */
  agent_id: string;

  /** Display name/username */
  username: string;

  /** PoGQ trust score (0-1 or 0-100) */
  trust_score: number;

  /** Total number of listings */
  total_listings: number;

  /** Total completed sales */
  total_sales: number;

  /** Average rating (1-5) */
  average_rating?: number;

  /** Account creation timestamp */
  member_since: number;

  /** Profile photo IPFS CID */
  avatar_cid?: string;
}

/**
 * Review for a listing
 */
export interface Review {
  /** Review ID */
  id: string;

  /** Listing hash being reviewed */
  listing_hash: string;

  /** Reviewer agent ID */
  reviewer_agent_id: string;

  /** Reviewer username */
  reviewer_name: string;

  /** Star rating (1-5) */
  rating: number;

  /** Review comment */
  comment: string;

  /** Review timestamp */
  created_at: number;

  /** Associated transaction hash */
  transaction_hash?: string;
}

/**
 * Listing with full context (seller + reviews)
 */
export interface ListingWithContext {
  listing: Listing;
  seller: SellerInfo;
  reviews: Review[];
}

/**
 * Listing creation input
 */
export interface CreateListingInput {
  title: string;
  description: string;
  price: number;
  category: ListingCategory;
  photos_ipfs_cids: string[];
  quantity_available?: number;
}

/**
 * Listing update input
 */
export interface UpdateListingInput {
  listing_hash: string;
  title?: string;
  description?: string;
  price?: number;
  category?: ListingCategory;
  photos_ipfs_cids?: string[];
  quantity_available?: number;
  status?: ListingStatus;
}

/**
 * Listing search/filter criteria
 */
export interface ListingFilters {
  /** Search query (matches title or description) */
  searchQuery?: string;

  /** Filter by category */
  category?: ListingCategory | 'All Categories';

  /** Minimum price */
  minPrice?: number;

  /** Maximum price */
  maxPrice?: number;

  /** Minimum trust score (0-100) */
  minTrustScore?: number;

  /** Sort field */
  sortBy?: 'newest' | 'oldest' | 'price_low' | 'price_high' | 'trust_score';
}

/**
 * Listing sort function
 */
export type ListingSortFn = (a: Listing, b: Listing) => number;
