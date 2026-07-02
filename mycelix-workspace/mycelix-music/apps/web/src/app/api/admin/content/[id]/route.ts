// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await request.json();
  const { action } = body;

  // In production, this would update the content status in the database
  // and potentially notify the content owner
  let result: { status: string; message: string };

  switch (action) {
    case 'approve':
      result = {
        status: 'approved',
        message: 'Content has been approved and is now visible.',
      };
      break;
    case 'remove':
      result = {
        status: 'removed',
        message: 'Content has been removed from the platform.',
      };
      break;
    case 'warn':
      result = {
        status: 'warning_sent',
        message: 'Warning has been sent to the content owner.',
      };
      break;
    default:
      return NextResponse.json(
        { error: 'Invalid action' },
        { status: 400 }
      );
  }

  return NextResponse.json({
    id,
    ...result,
    updatedAt: new Date().toISOString(),
  });
}
