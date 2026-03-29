// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { type, itemId, txHash, address } = body;

  if (!type || !itemId || !txHash || !address) {
    return NextResponse.json(
      { error: 'Missing required fields' },
      { status: 400 }
    );
  }

  // In production:
  // 1. Verify the transaction on-chain
  // 2. Update user subscription/ownership in database
  // 3. Send confirmation email
  // 4. Trigger any webhooks

  // Mock verification
  const confirmation = {
    success: true,
    type,
    itemId,
    txHash,
    address,
    confirmedAt: new Date().toISOString(),
    message: getConfirmationMessage(type),
  };

  // Log the purchase
  console.log('Purchase confirmed:', confirmation);

  return NextResponse.json(confirmation);
}

function getConfirmationMessage(type: string): string {
  switch (type) {
    case 'subscription':
      return 'Your subscription is now active. Enjoy ad-free listening!';
    case 'nft':
      return 'NFT minted successfully! Check your wallet for the new token.';
    case 'tip':
      return 'Tip sent successfully! The artist has been notified.';
    default:
      return 'Purchase completed successfully.';
  }
}
