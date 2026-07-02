// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useQuery } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';
import { useAuth } from './use-auth';

interface FriendActivityItem {
  id: string;
  type: 'listening' | 'live' | 'played';
  timestamp: string;
  user: {
    id: string;
    displayName: string;
    avatarUrl?: string;
    isOnline: boolean;
  };
  track?: {
    id: string;
    title: string;
    coverUrl?: string;
    artist: {
      id: string;
      name: string;
    };
  };
}

export function useFriendActivity() {
  const { isAuthenticated } = useAuth();

  return useQuery({
    queryKey: ['friend-activity'],
    queryFn: async () => {
      const response = await api.get<FriendActivityItem[]>(endpoints.social.activity);
      return response.data;
    },
    enabled: isAuthenticated,
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 30 * 1000,
  });
}
