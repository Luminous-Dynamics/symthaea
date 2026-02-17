'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';

interface FeaturedHeroItem {
  id: string;
  type: 'album' | 'artist' | 'playlist';
  title: string;
  subtitle: string;
  imageUrl: string;
  href: string;
}

interface FeaturedPlaylist {
  id: string;
  name: string;
  description?: string;
  coverUrl?: string;
  trackCount: number;
}

interface FeaturedContent {
  hero: FeaturedHeroItem[];
  playlists: FeaturedPlaylist[];
}

export function useFeaturedContent() {
  return useQuery({
    queryKey: ['featured'],
    queryFn: async () => {
      const response = await api.get<FeaturedContent>(endpoints.discover.featured);
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
