// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextResponse } from 'next/server';

export async function GET() {
  // In production, this would fetch auto-flagged content from AI moderation system
  const flaggedContent = [
    {
      id: 'flag-1',
      type: 'song',
      reason: 'Potential copyright match',
      confidence: 0.87,
      content: {
        id: 'song-999',
        title: 'Similar Melody',
        artist: 'NewArtist',
      },
      flaggedAt: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: 'flag-2',
      type: 'comment',
      reason: 'Offensive language detected',
      confidence: 0.92,
      content: {
        id: 'comment-456',
        text: 'This is a [filtered] song...',
      },
      flaggedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: 'flag-3',
      type: 'song',
      reason: 'Explicit content not marked',
      confidence: 0.78,
      content: {
        id: 'song-888',
        title: 'Uncensored Track',
        artist: 'Underground MC',
      },
      flaggedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
    },
  ];

  return NextResponse.json({ content: flaggedContent });
}
