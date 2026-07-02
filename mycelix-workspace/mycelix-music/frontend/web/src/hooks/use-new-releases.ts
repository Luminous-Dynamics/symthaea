// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';

interface NewRelease {
  id: string;
  title: string;
  coverUrl?: string;
  type: 'album' | 'ep' | 'single';
  artist: {
    id: string;
    name: string;
  };
  releaseDate: string;
}

export function useNewReleases() {
  return useQuery({
    queryKey: ['new-releases'],
    queryFn: async () => {
      const response = await api.get<NewRelease[]>(endpoints.albums.newReleases);
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
