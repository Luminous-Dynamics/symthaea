'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';

interface RecentlyPlayedItem {
  id: string;
  type: 'track' | 'album' | 'playlist';
  title: string;
  imageUrl?: string;
  playedAt: string;
}

export function useRecentlyPlayed(userId?: string) {
  return useQuery({
    queryKey: ['recently-played', userId],
    queryFn: async () => {
      const response = await api.get<RecentlyPlayedItem[]>(endpoints.library.history);
      return response.data;
    },
    enabled: !!userId,
    staleTime: 60 * 1000, // 1 minute
  });
}
