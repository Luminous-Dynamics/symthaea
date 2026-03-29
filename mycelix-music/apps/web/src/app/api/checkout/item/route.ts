// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const type = searchParams.get('type');
  const id = searchParams.get('id');

  if (!type || !id) {
    return NextResponse.json(
      { error: 'Missing type or id' },
      { status: 400 }
    );
  }

  // In production, fetch from database
  if (type === 'nft') {
    // Mock NFT data
    const nft = {
      type: 'nft',
      id,
      name: 'Digital Dreams #42',
      description: 'Exclusive NFT with special perks including backstage access and unreleased tracks',
      price: '0.1',
      priceUSD: '$199.99',
      image: '/nft-placeholder.jpg',
      metadata: {
        artist: 'CryptoBeats',
        artistAddress: '0x1234567890abcdef1234567890abcdef12345678',
        edition: '42/100',
        perks: [
          'Backstage access to virtual concerts',
          'Unreleased track downloads',
          'Exclusive Discord channel',
          'Signed digital artwork',
        ],
      },
    };
    return NextResponse.json(nft);
  }

  if (type === 'tip') {
    // Mock artist tip data
    const tip = {
      type: 'tip',
      id,
      name: 'Tip for CryptoBeats',
      description: 'Show your support with a direct tip to the artist',
      price: id, // The id is the tip amount
      priceUSD: `≈ $${(parseFloat(id) * 2000).toFixed(2)}`, // Assuming 1 ETH = $2000
      image: '/artist-placeholder.jpg',
      metadata: {
        artistName: 'CryptoBeats',
        artistAddress: '0x1234567890abcdef1234567890abcdef12345678',
      },
    };
    return NextResponse.json(tip);
  }

  return NextResponse.json(
    { error: 'Invalid checkout type' },
    { status: 400 }
  );
}
