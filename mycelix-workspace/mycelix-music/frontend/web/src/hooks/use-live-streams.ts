// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';

interface LiveStream {
  id: string;
  title: string;
  thumbnailUrl?: string;
  viewerCount: number;
  startedAt: string;
  category: string;
  tags?: string[];
  isLive: boolean;
  artist: {
    id: string;
    name: string;
    imageUrl?: string;
  };
}

export function useLiveStreams() {
  return useQuery({
    queryKey: ['live-streams'],
    queryFn: async () => {
      const response = await api.get<LiveStream[]>(endpoints.live.streams);
      return response.data;
    },
    staleTime: 30 * 1000, // 30 seconds - refresh more often for live content
    refetchInterval: 30 * 1000,
  });
}
