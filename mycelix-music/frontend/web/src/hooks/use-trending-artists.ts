'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';

interface TrendingArtist {
  id: string;
  name: string;
  imageUrl?: string;
  verified: boolean;
  followers: number;
  trendChange?: number;
}

export function useTrendingArtists() {
  return useQuery({
    queryKey: ['trending-artists'],
    queryFn: async () => {
      const response = await api.get<TrendingArtist[]>(endpoints.artists.trending);
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
