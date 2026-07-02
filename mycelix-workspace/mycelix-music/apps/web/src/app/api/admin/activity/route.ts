// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextResponse } from 'next/server';

export async function GET() {
  // In production, this would fetch from the database
  const activities = [
    {
      id: '1',
      type: 'report',
      message: 'New content report submitted',
      timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    },
    {
      id: '2',
      type: 'user',
      message: 'New artist registration: CryptoBeats',
      timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    },
    {
      id: '3',
      type: 'song',
      message: 'Song "Digital Dreams" uploaded for review',
      timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    },
    {
      id: '4',
      type: 'moderation',
      message: 'Content report resolved by admin',
      timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    },
    {
      id: '5',
      type: 'user',
      message: 'User account suspended: SpamBot123',
      timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    },
  ];

  return NextResponse.json({ activities });
}
