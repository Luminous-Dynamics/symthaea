// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const status = searchParams.get('status') || 'pending';

  // In production, this would fetch from the database with proper filtering
  const allReports = [
    {
      id: '1',
      type: 'song',
      reason: 'Copyright infringement',
      status: 'pending',
      reportedBy: {
        address: '0x1234567890abcdef1234567890abcdef12345678',
        name: 'MusicLabel Official',
      },
      target: {
        id: 'song-123',
        type: 'song',
        title: 'Stolen Melody',
      },
      createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      description: 'This song uses our copyrighted melody without permission.',
    },
    {
      id: '2',
      type: 'user',
      reason: 'Harassment',
      status: 'pending',
      reportedBy: {
        address: '0xabcdef1234567890abcdef1234567890abcdef12',
        name: 'ArtistX',
      },
      target: {
        id: 'user-456',
        type: 'user',
        name: 'TrollAccount',
      },
      createdAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
      description: 'This user has been sending harassing messages.',
    },
    {
      id: '3',
      type: 'comment',
      reason: 'Spam',
      status: 'reviewing',
      reportedBy: {
        address: '0x9876543210fedcba9876543210fedcba98765432',
      },
      target: {
        id: 'comment-789',
        type: 'comment',
        title: 'Buy crypto now!!!',
      },
      createdAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
    },
  ];

  const reports = status === 'all'
    ? allReports
    : allReports.filter(r => r.status === status);

  return NextResponse.json({ reports });
}
