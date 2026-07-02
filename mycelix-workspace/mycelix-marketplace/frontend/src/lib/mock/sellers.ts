// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { SellerInfo } from '$types';

const sellers: SellerInfo[] = [
  {
    agent_id: 'agent_alpha',
    username: 'Alice Artisans',
    trust_score: 92,
    total_listings: 12,
    total_sales: 58,
    average_rating: 4.8,
    member_since: Date.now() - 1000 * 60 * 60 * 24 * 365 * 2,
    avatar_cid: undefined,
  },
  {
    agent_id: 'agent_beta',
    username: 'Beta Bazaar',
    trust_score: 88,
    total_listings: 20,
    total_sales: 134,
    average_rating: 4.6,
    member_since: Date.now() - 1000 * 60 * 60 * 24 * 365 * 3,
    avatar_cid: undefined,
  },
];

export function getMockSeller(agentId: string): SellerInfo | undefined {
  return sellers.find((s) => s.agent_id === agentId);
}
