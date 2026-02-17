/**
 * Intent-based buying types
 */
export interface IntentRequest {
  title: string;
  categories: string[];
  budgetMin: number;
  budgetMax: number;
  deliveryDays: number;
  region: string;
  mustHaveProof: boolean;
  allowBundles: boolean;
  notes?: string;
}

export interface IntentBundleItem {
  listing_id: string;
  title: string;
  seller: string;
  price: number;
  proof_status: 'fulfilled' | 'pending' | 'none';
  risk_score?: number;
}

export interface IntentBundleSuggestion {
  id: string;
  total: number;
  deliveryEstimate: string;
  items: IntentBundleItem[];
  fitScore: number; // 0-1
}
