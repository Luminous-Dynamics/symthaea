// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await request.json();
  const { action } = body;

  // In production, this would update the database
  // action can be: 'resolve', 'dismiss', 'escalate'

  let newStatus: string;
  switch (action) {
    case 'resolve':
      newStatus = 'resolved';
      break;
    case 'dismiss':
      newStatus = 'dismissed';
      break;
    case 'escalate':
      newStatus = 'escalated';
      break;
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
