'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';

interface PreviewTrack {
  coverUrl?: string;
}

interface PersonalizedMix {
  id: string;
  name: string;
  description: string;
  type: 'daily' | 'genre' | 'artist';
  previewTracks?: PreviewTrack[];
}

export function usePersonalizedMixes(userId?: string) {
  return useQuery({
    queryKey: ['personalized-mixes', userId],
    queryFn: async () => {
      const response = await api.get<PersonalizedMix[]>(endpoints.discover.personalized);
      return response.data;
    },
    enabled: !!userId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
