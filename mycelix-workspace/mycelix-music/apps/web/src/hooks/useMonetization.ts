// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Monetization Hook
 *
 * Revenue and monetization features:
 * - Stem marketplace
 * - Licensing system
 * - Subscription management
 * - Revenue analytics
 */

import { useState, useCallback, useEffect } from 'react';

// Types
export interface StemListing {
  id: string;
  title: string;
  description: string;
  type: 'drums' | 'bass' | 'melody' | 'vocals' | 'fx' | 'full';
  bpm: number;
  key: string;
  genre: string;
  tags: string[];
  duration: number;
  previewUrl: string;
  price: number;
  currency: string;
  seller: SellerProfile;
  licenses: LicenseOption[];
  stats: ListingStats;
  createdAt: Date;
  featured: boolean;
}

export interface SellerProfile {
  id: string;
  username: string;
  displayName: string;
  avatar?: string;
  verified: boolean;
  totalSales: number;
  rating: number;
  reviews: number;
}

export interface LicenseOption {
  id: string;
  name: string;
  type: 'basic' | 'premium' | 'exclusive' | 'custom';
  price: number;
  currency: string;
  rights: LicenseRights;
  popular?: boolean;
}

export interface LicenseRights {
  commercialUse: boolean;
  unlimitedStreams: boolean;
  maxStreams?: number;
  radioPlay: boolean;
  musicVideos: boolean;
  advertising: boolean;
  sync: boolean;
  modification: boolean;
  exclusiveOwnership: boolean;
  creditRequired: boolean;
  territory: 'worldwide' | 'regional';
  duration: 'perpetual' | 'limited';
  durationYears?: number;
}

export interface ListingStats {
  views: number;
  plays: number;
  downloads: number;
  sales: number;
  revenue: number;
  conversionRate: number;
}

export interface Purchase {
  id: string;
  listing: StemListing;
  license: LicenseOption;
  buyer: { id: string; username: string };
  price: number;
  currency: string;
  transactionHash?: string;
  status: 'pending' | 'completed' | 'failed' | 'refunded';
  createdAt: Date;
  downloadUrl?: string;
}

export interface Subscription {
  id: string;
  tier: SubscriptionTier;
  status: 'active' | 'cancelled' | 'past_due' | 'trialing';
  currentPeriodStart: Date;
  currentPeriodEnd: Date;
  cancelAtPeriodEnd: boolean;
  paymentMethod?: PaymentMethod;
}

export interface SubscriptionTier {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: 'month' | 'year';
  features: string[];
  limits: {
    monthlyDownloads?: number;
    storageGB?: number;
    collaborators?: number;
    aiCredits?: number;
  };
}

export interface PaymentMethod {
  id: string;
  type: 'card' | 'crypto' | 'paypal';
  last4?: string;
  brand?: string;
  walletAddress?: string;
}

export interface RevenueData {
  period: string;
  sales: number;
  revenue: number;
  streams: number;
  streamRevenue: number;
  tips: number;
  subscriptions: number;
  total: number;
}

export interface PayoutInfo {
  pending: number;
  available: number;
  nextPayoutDate: Date;
  minimumPayout: number;
  payoutMethod?: PaymentMethod;
  history: PayoutHistory[];
}

export interface PayoutHistory {
  id: string;
  amount: number;
  currency: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  createdAt: Date;
  completedAt?: Date;
  transactionId?: string;
}

export interface MonetizationState {
  isLoading: boolean;
  myListings: StemListing[];
  myPurchases: Purchase[];
  subscription: Subscription | null;
  revenue: RevenueData[];
  payout: PayoutInfo | null;
  error: string | null;
}

// Subscription tiers
export const SUBSCRIPTION_TIERS: SubscriptionTier[] = [
  {
    id: 'free',
    name: 'Free',
    price: 0,
    currency: 'USD',
    interval: 'month',
    features: [
      'Basic streaming',
      'Create playlists',
      'Limited downloads',
      '1GB storage',
    ],
    limits: {
      monthlyDownloads: 5,
      storageGB: 1,
      collaborators: 0,
      aiCredits: 10,
    },
  },
  {
    id: 'creator',
    name: 'Creator',
    price: 9.99,
    currency: 'USD',
    interval: 'month',
    features: [
      'Unlimited streaming',
      'Stem separation',
      '50 downloads/month',
      '10GB storage',
      'Basic AI features',
      'Collaboration (2 members)',
    ],
    limits: {
      monthlyDownloads: 50,
      storageGB: 10,
      collaborators: 2,
      aiCredits: 100,
    },
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 19.99,
    currency: 'USD',
    interval: 'month',
    features: [
      'Everything in Creator',
      'Unlimited downloads',
      '100GB storage',
      'Full AI suite',
      'Priority support',
      'Collaboration (10 members)',
      'Sell on marketplace',
      'Advanced analytics',
    ],
    limits: {
      monthlyDownloads: -1,
      storageGB: 100,
      collaborators: 10,
      aiCredits: 500,
    },
  },
  {
    id: 'studio',
    name: 'Studio',
    price: 49.99,
    currency: 'USD',
    interval: 'month',
    features: [
      'Everything in Pro',
      'Unlimited storage',
      'Unlimited AI credits',
      'White-label exports',
      'API access',
      'Dedicated support',
      'Custom licensing',
      'Revenue share boost',
    ],
    limits: {
      monthlyDownloads: -1,
      storageGB: -1,
      collaborators: -1,
      aiCredits: -1,
    },
  },
];

// License templates
export const LICENSE_TEMPLATES: Omit<LicenseOption, 'id' | 'price' | 'currency'>[] = [
  {
    name: 'Basic',
    type: 'basic',
    rights: {
      commercialUse: true,
      unlimitedStreams: false,
      maxStreams: 100000,
      radioPlay: false,
      musicVideos: false,
      advertising: false,
      sync: false,
      modification: true,
      exclusiveOwnership: false,
      creditRequired: true,
      territory: 'worldwide',
      duration: 'perpetual',
    },
  },
  {
    name: 'Premium',
    type: 'premium',
    popular: true,
    rights: {
      commercialUse: true,
      unlimitedStreams: true,
      radioPlay: true,
      musicVideos: true,
      advertising: false,
      sync: false,
      modification: true,
      exclusiveOwnership: false,
      creditRequired: false,
      territory: 'worldwide',
      duration: 'perpetual',
    },
  },
  {
    name: 'Exclusive',
    type: 'exclusive',
    rights: {
      commercialUse: true,
      unlimitedStreams: true,
      radioPlay: true,
      musicVideos: true,
      advertising: true,
      sync: true,
      modification: true,
      exclusiveOwnership: true,
      creditRequired: false,
      territory: 'worldwide',
      duration: 'perpetual',
    },
  },
];

export function useMonetization(userId?: string) {
  const [state, setState] = useState<MonetizationState>({
    isLoading: false,
    myListings: [],
    myPurchases: [],
    subscription: null,
    revenue: [],
    payout: null,
    error: null,
  });

  /**
   * Create a new stem listing
   */
  const createListing = useCallback(async (
    data: {
      title: string;
      description: string;
      type: StemListing['type'];
      bpm: number;
      key: string;
      genre: string;
      tags: string[];
      audioFile: File;
      previewFile?: File;
      licenses: Omit<LicenseOption, 'id'>[];
    }
  ): Promise<StemListing | null> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const listing: StemListing = {
        id: `listing-${Date.now()}`,
        title: data.title,
        description: data.description,
        type: data.type,
        bpm: data.bpm,
        key: data.key,
        genre: data.genre,
        tags: data.tags,
        duration: 120,
        previewUrl: URL.createObjectURL(data.previewFile || data.audioFile),
        price: data.licenses[0]?.price || 0,
        currency: 'USD',
        seller: {
          id: userId || 'me',
          username: 'current_user',
          displayName: 'Current User',
          verified: false,
          totalSales: 0,
          rating: 0,
          reviews: 0,
        },
        licenses: data.licenses.map((l, i) => ({ ...l, id: `license-${i}` })),
        stats: {
          views: 0,
          plays: 0,
          downloads: 0,
          sales: 0,
          revenue: 0,
          conversionRate: 0,
        },
        createdAt: new Date(),
        featured: false,
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        myListings: [...prev.myListings, listing],
      }));

      return listing;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to create listing',
      }));
      return null;
    }
  }, [userId]);

  /**
   * Update a listing
   */
  const updateListing = useCallback(async (
    listingId: string,
    updates: Partial<StemListing>
  ): Promise<boolean> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      setState(prev => ({
        ...prev,
        isLoading: false,
        myListings: prev.myListings.map(l =>
          l.id === listingId ? { ...l, ...updates } : l
        ),
      }));

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to update listing' }));
      return false;
    }
  }, []);

  /**
   * Delete a listing
   */
  const deleteListing = useCallback(async (listingId: string): Promise<boolean> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      setState(prev => ({
        ...prev,
        isLoading: false,
        myListings: prev.myListings.filter(l => l.id !== listingId),
      }));

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to delete listing' }));
      return false;
    }
  }, []);

  /**
   * Purchase a stem
   */
  const purchaseStem = useCallback(async (
    listingId: string,
    licenseId: string
  ): Promise<Purchase | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Simulated purchase
      const purchase: Purchase = {
        id: `purchase-${Date.now()}`,
        listing: { id: listingId } as StemListing,
        license: { id: licenseId } as LicenseOption,
        buyer: { id: userId || 'me', username: 'current_user' },
        price: 29.99,
        currency: 'USD',
        status: 'completed',
        createdAt: new Date(),
        downloadUrl: '/downloads/stem.wav',
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        myPurchases: [...prev.myPurchases, purchase],
      }));

      return purchase;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Purchase failed' }));
      return null;
    }
  }, [userId]);

  /**
   * Subscribe to a tier
   */
  const subscribe = useCallback(async (
    tierId: string,
    paymentMethodId?: string
  ): Promise<Subscription | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const tier = SUBSCRIPTION_TIERS.find(t => t.id === tierId);
      if (!tier) throw new Error('Invalid tier');

      const subscription: Subscription = {
        id: `sub-${Date.now()}`,
        tier,
        status: 'active',
        currentPeriodStart: new Date(),
        currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
        cancelAtPeriodEnd: false,
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        subscription,
      }));

      return subscription;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Subscription failed' }));
      return null;
    }
  }, []);

  /**
   * Cancel subscription
   */
  const cancelSubscription = useCallback(async (
    immediately: boolean = false
  ): Promise<boolean> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      setState(prev => ({
        ...prev,
        isLoading: false,
        subscription: prev.subscription
          ? {
              ...prev.subscription,
              status: immediately ? 'cancelled' : prev.subscription.status,
              cancelAtPeriodEnd: !immediately,
            }
          : null,
      }));

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to cancel' }));
      return false;
    }
  }, []);

  /**
   * Get revenue analytics
   */
  const getRevenue = useCallback(async (
    period: 'week' | 'month' | 'year' = 'month'
  ): Promise<RevenueData[]> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 600));

      // Generate sample data
      const data: RevenueData[] = [];
      const periods = period === 'week' ? 7 : period === 'month' ? 30 : 12;

      for (let i = 0; i < periods; i++) {
        const sales = Math.floor(Math.random() * 10);
        const streams = Math.floor(Math.random() * 10000);
        const tips = Math.random() * 50;

        data.push({
          period: `Day ${i + 1}`,
          sales,
          revenue: sales * 25,
          streams,
          streamRevenue: streams * 0.003,
          tips,
          subscriptions: 0,
          total: sales * 25 + streams * 0.003 + tips,
        });
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        revenue: data,
      }));

      return data;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to load revenue' }));
      return [];
    }
  }, []);

  /**
   * Get payout information
   */
  const getPayoutInfo = useCallback(async (): Promise<PayoutInfo | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 400));

      const payout: PayoutInfo = {
        pending: 127.45,
        available: 543.21,
        nextPayoutDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        minimumPayout: 50,
        history: [
          {
            id: 'payout-1',
            amount: 250.00,
            currency: 'USD',
            status: 'completed',
            createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
            completedAt: new Date(Date.now() - 28 * 24 * 60 * 60 * 1000),
          },
        ],
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        payout,
      }));

      return payout;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to load payout info' }));
      return null;
    }
  }, []);

  /**
   * Request payout
   */
  const requestPayout = useCallback(async (amount: number): Promise<boolean> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      setState(prev => ({
        ...prev,
        isLoading: false,
        payout: prev.payout
          ? {
              ...prev.payout,
              available: prev.payout.available - amount,
              history: [
                {
                  id: `payout-${Date.now()}`,
                  amount,
                  currency: 'USD',
                  status: 'pending',
                  createdAt: new Date(),
                },
                ...prev.payout.history,
              ],
            }
          : null,
      }));

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Payout request failed' }));
      return false;
    }
  }, []);

  return {
    ...state,
    subscriptionTiers: SUBSCRIPTION_TIERS,
    licenseTemplates: LICENSE_TEMPLATES,
    createListing,
    updateListing,
    deleteListing,
    purchaseStem,
    subscribe,
    cancelSubscription,
    getRevenue,
    getPayoutInfo,
    requestPayout,
  };
}

export default useMonetization;
