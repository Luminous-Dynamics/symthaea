// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get('page') || '1', 10);
  const limit = parseInt(searchParams.get('limit') || '20', 10);
  const search = searchParams.get('search') || '';
  const role = searchParams.get('role');
  const status = searchParams.get('status');

  // In production, this would fetch from the database with proper pagination
  const mockUsers = [
    {
      id: '1',
      address: '0x1234567890abcdef1234567890abcdef12345678',
      name: 'CryptoBeats',
      email: 'crypto@beats.io',
      role: 'artist',
      status: 'active',
      createdAt: '2024-01-15T10:30:00Z',
      lastActive: '2024-12-30T14:20:00Z',
      stats: { songs: 45, followers: 12500, plays: 1250000 },
    },
    {
      id: '2',
      address: '0xabcdef1234567890abcdef1234567890abcdef12',
      name: 'SynthMaster',
      role: 'artist',
      status: 'active',
      createdAt: '2024-02-20T08:15:00Z',
      lastActive: '2024-12-29T22:45:00Z',
      stats: { songs: 23, followers: 8900, plays: 890000 },
    },
    {
      id: '3',
      address: '0x9876543210fedcba9876543210fedcba98765432',
      name: 'MusicFan99',
      role: 'user',
      status: 'active',
      createdAt: '2024-03-10T16:00:00Z',
      lastActive: '2024-12-30T18:30:00Z',
      stats: { songs: 0, followers: 150, plays: 0 },
    },
    {
      id: '4',
      address: '0xfedcba9876543210fedcba9876543210fedcba98',
      name: 'SpamBot123',
      role: 'user',
      status: 'suspended',
      createdAt: '2024-11-01T12:00:00Z',
      lastActive: '2024-11-15T09:00:00Z',
      stats: { songs: 0, followers: 5, plays: 0 },
    },
    {
      id: '5',
      address: '0x5678901234abcdef5678901234abcdef56789012',
      name: 'ModeratorJane',
      email: 'jane@mycelix.io',
      role: 'moderator',
      status: 'active',
      createdAt: '2024-01-01T00:00:00Z',
      lastActive: '2024-12-30T20:00:00Z',
      stats: { songs: 0, followers: 0, plays: 0 },
    },
  ];

  // Apply filters
  let filtered = mockUsers;

  if (search) {
    const lowerSearch = search.toLowerCase();
    filtered = filtered.filter(
      u =>
        u.name?.toLowerCase().includes(lowerSearch) ||
        u.address.toLowerCase().includes(lowerSearch) ||
        u.email?.toLowerCase().includes(lowerSearch)
    );
  }

  if (role && role !== 'all') {
    filtered = filtered.filter(u => u.role === role);
  }

  if (status && status !== 'all') {
    filtered = filtered.filter(u => u.status === status);
  }

  // Pagination
  const total = filtered.length;
  const totalPages = Math.ceil(total / limit);
  const start = (page - 1) * limit;
  const users = filtered.slice(start, start + limit);

  return NextResponse.json({
    users,
    page,
    limit,
    total,
    totalPages,
  });
}
