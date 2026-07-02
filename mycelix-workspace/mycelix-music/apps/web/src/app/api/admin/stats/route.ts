// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextResponse } from 'next/server';

export async function GET() {
  // In production, this would fetch from the database
  const stats = {
    users: {
      total: 15234,
      active: 8921,
      newToday: 127,
    },
    songs: {
      total: 45678,
      pending: 23,
      flagged: 5,
    },
    reports: {
      total: 156,
      pending: 12,
      resolved: 144,
    },
    revenue: {
      total: '1,234.56 ETH',
      today: '12.34 ETH',
    },
  };

  return NextResponse.json(stats);
}
