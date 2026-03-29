// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import useSWR from 'swr';
import { Song } from '../../data/mockSongs';
import { mockSongs } from '../../data/mockSongs';
import { PaymentModel } from '@/lib';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3100';

interface ArtistStats {
  artistAddress: string;
  totalSongs: number;
  totalPlays: number;
  totalEarnings: number;
}

interface AnalyticsData {
  days: number;
  timeseries: Array<{ day: string; plays: number; earnings: number }>;
  topSongs: Array<{ id: string; title: string; plays: number; earnings: number }>;
}

// Map API response to frontend Song type
function mapApiSong(apiSong: any): Song {
  const paymentModelMap: Record<string, PaymentModel> = {
    pay_per_stream: PaymentModel.PAY_PER_STREAM,
    freemium: PaymentModel.FREEMIUM,
    pay_what_you_want: PaymentModel.PAY_WHAT_YOU_WANT,
    patronage: PaymentModel.PATRONAGE,
    gift_economy: PaymentModel.PAY_WHAT_YOU_WANT,
    pay_per_download: PaymentModel.PAY_PER_STREAM,
    subscription: PaymentModel.PATRONAGE,
    nft_gated: PaymentModel.PAY_PER_STREAM,
    staking_gated: PaymentModel.PAY_PER_STREAM,
    token_tip: PaymentModel.PAY_WHAT_YOU_WANT,
    time_barter: PaymentModel.PAY_WHAT_YOU_WANT,
    auction: PaymentModel.PAY_PER_STREAM,
  };

  return {
    id: apiSong.id,
    title: apiSong.title,
    artist: apiSong.artist,
    genre: apiSong.genre,
    description: apiSong.description || '',
    ipfsHash: apiSong.ipfs_hash,
    paymentModel: paymentModelMap[apiSong.payment_model] || PaymentModel.PAY_PER_STREAM,
    plays: apiSong.plays || 0,
    earnings: String(apiSong.earnings || '0'),
    coverArt: apiSong.cover_art || 'https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=600&h=600&fit=crop',
    audioUrl: apiSong.audio_url,
    freePlaysRemaining: apiSong.payment_model === 'freemium' ? 3 : undefined,
    tipAmount: apiSong.payment_model === 'pay_what_you_want' ? 'avg $0.15' : undefined,
  };
}

const fetcher = async (url: string) => {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
};

export function useArtistStats(address: string | undefined) {
  const url = address ? `${API_URL}/api/artists/${address}/stats` : null;

  const { data, error, isLoading, mutate } = useSWR<ArtistStats>(url, fetcher, {
    revalidateOnFocus: false,
  });

  // Fallback to mock data if no API data
  const mockStats: ArtistStats = {
    artistAddress: address || '',
    totalSongs: mockSongs.slice(0, 5).length,
    totalPlays: mockSongs.slice(0, 5).reduce((sum, s) => sum + s.plays, 0),
    totalEarnings: mockSongs.slice(0, 5).reduce((sum, s) => sum + parseFloat(s.earnings.replace(',', '')), 0),
  };

  return {
    stats: data || mockStats,
    isLoading,
    error: error || null,
    mutate,
  };
}

export function useArtistSongs(address: string | undefined) {
  const url = address ? `${API_URL}/api/artists/${address}/songs` : null;

  const { data, error, isLoading, mutate } = useSWR(url, async (url) => {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    const total = parseInt(resp.headers.get('X-Total-Count') || '0', 10);
    const songs = await resp.json();
    return { songs, total };
  }, {
    revalidateOnFocus: false,
    fallbackData: { songs: mockSongs.slice(0, 5), total: 5 },
  });

  // Map API songs to frontend format
  const songs = data?.songs.map((s: any) => {
    if (s.paymentModel !== undefined) return s;
    return mapApiSong(s);
  }) || mockSongs.slice(0, 5);

  return {
    songs,
    total: data?.total || songs.length,
    isLoading,
    error: error || null,
    mutate,
  };
}

export function useArtistAnalytics(address: string | undefined, days: number = 30) {
  const url = address ? `${API_URL}/api/analytics/artist/${address}?days=${days}` : null;

  const { data, error, isLoading, mutate } = useSWR<AnalyticsData>(url, fetcher, {
    revalidateOnFocus: false,
  });

  // Generate mock timeseries data
  const mockData: AnalyticsData = {
    days,
    timeseries: [
      { day: new Date(Date.now() - 25 * 86400000).toISOString(), plays: 150, earnings: 12.50 },
      { day: new Date(Date.now() - 20 * 86400000).toISOString(), plays: 220, earnings: 18.30 },
      { day: new Date(Date.now() - 15 * 86400000).toISOString(), plays: 310, earnings: 25.80 },
      { day: new Date(Date.now() - 10 * 86400000).toISOString(), plays: 380, earnings: 32.40 },
      { day: new Date(Date.now() - 5 * 86400000).toISOString(), plays: 490, earnings: 41.20 },
      { day: new Date().toISOString(), plays: 620, earnings: 52.30 },
    ],
    topSongs: mockSongs.slice(0, 5).map(s => ({
      id: s.id,
      title: s.title,
      plays: s.plays,
      earnings: parseFloat(s.earnings.replace(',', '')),
    })),
  };

  return {
    analytics: data || mockData,
    isLoading,
    error: error || null,
    mutate,
  };
}
