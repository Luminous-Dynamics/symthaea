// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import useSWR from 'swr';
import { Song } from '../../data/mockSongs';
import { mockSongs } from '../../data/mockSongs';
import { PaymentModel } from '@/lib';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3100';

interface UseSongsParams {
  genre?: string;
  model?: string;
  q?: string;
  limit?: number;
  offset?: number;
  sort?: 'created_at' | 'plays' | 'earnings';
  order?: 'asc' | 'desc';
}

interface SongsResponse {
  songs: Song[];
  total: number;
  isLoading: boolean;
  error: Error | null;
  mutate: () => void;
}

const fetcher = async (url: string): Promise<{ songs: any[]; total: number }> => {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`API error: ${resp.status}`);
  }
  const total = parseInt(resp.headers.get('X-Total-Count') || '0', 10);
  const songs = await resp.json();
  return { songs, total };
};

// Map API response to frontend Song type
function mapApiSong(apiSong: any): Song {
  // Map payment_model from API to PaymentModel enum
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

export function useSongs(params: UseSongsParams = {}): SongsResponse {
  const { genre, model, q, limit = 50, offset = 0, sort = 'created_at', order = 'desc' } = params;

  // Build query string
  const queryParams = new URLSearchParams();
  if (genre && genre !== 'all') queryParams.set('genre', genre);
  if (model && model !== 'all') queryParams.set('model', model);
  if (q) queryParams.set('q', q);
  queryParams.set('limit', String(limit));
  queryParams.set('offset', String(offset));
  queryParams.set('sort', sort);
  queryParams.set('order', order);

  const url = `${API_URL}/api/songs?${queryParams.toString()}`;

  const { data, error, isLoading, mutate } = useSWR(url, fetcher, {
    // Fallback to mock data on error
    fallbackData: { songs: mockSongs, total: mockSongs.length },
    revalidateOnFocus: false,
    dedupingInterval: 5000,
  });

  // Map API songs to frontend format, or use mock data
  const songs = data?.songs.map((s: any) => {
    // Check if it's already in frontend format (mock data)
    if (s.paymentModel !== undefined) return s;
    return mapApiSong(s);
  }) || mockSongs;

  return {
    songs,
    total: data?.total || songs.length,
    isLoading,
    error: error || null,
    mutate,
  };
}

export function useSong(id: string) {
  const url = `${API_URL}/api/songs/${id}`;

  const { data, error, isLoading, mutate } = useSWR(id ? url : null, async (url) => {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
  });

  return {
    song: data ? mapApiSong(data) : null,
    isLoading,
    error: error || null,
    mutate,
  };
}
