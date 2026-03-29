// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  // In production, fetch user from database
  const user = {
    id,
    address: '0x1234567890abcdef1234567890abcdef12345678',
    name: 'CryptoBeats',
    email: 'crypto@beats.io',
    role: 'artist',
    status: 'active',
    createdAt: '2024-01-15T10:30:00Z',
    lastActive: '2024-12-30T14:20:00Z',
    stats: { songs: 45, followers: 12500, plays: 1250000 },
    biography: 'Electronic music producer creating beats for the future.',
    socialLinks: {
      twitter: '@cryptobeats',
      instagram: '@cryptobeats',
    },
    verificationStatus: 'verified',
  };

  return NextResponse.json(user);
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await request.json();
  const { action } = body;

  // In production, this would update the database
  let newStatus: string;
  switch (action) {
    case 'suspend':
      newStatus = 'suspended';
      break;
    case 'ban':
      newStatus = 'banned';
      break;
    case 'unban':
      newStatus = 'active';
      break;
    case 'verify':
      // Handle verification
      return NextResponse.json({
        id,
        verificationStatus: 'verified',
        updatedAt: new Date().toISOString(),
      });
    default:
      return NextResponse.json(
        { error: 'Invalid action' },
        { status: 400 }
      );
  }

  return NextResponse.json({
    id,
    status: newStatus,
    updatedAt: new Date().toISOString(),
  });
}
