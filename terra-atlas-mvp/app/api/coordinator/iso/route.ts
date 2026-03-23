// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { fetchLMP, fetchGenerationMix, fetchGridStatus, type ISOName } from '@/lib/iso/market-bridge'

const VALID_ISOS: ISOName[] = ['ERCOT', 'CAISO', 'PJM']

/**
 * GET /api/coordinator/iso
 *
 * National Energy Transition Coordinator — ISO/RTO Market Bridge API
 *
 * Fetches real-time wholesale electricity market data from public ISO APIs.
 * Read-only — does not submit bids or participate in markets.
 *
 * Query params:
 *   iso    - ERCOT | CAISO | PJM (required)
 *   data   - lmp | generation | status | all (default: all)
 *   node   - Pricing node for LMP (optional, default varies by ISO)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const iso = (searchParams.get('iso') || '').toUpperCase() as ISOName
    const dataType = searchParams.get('data') || 'all'
    const node = searchParams.get('node') || undefined

    if (!VALID_ISOS.includes(iso)) {
      return NextResponse.json(
        { error: `iso must be one of: ${VALID_ISOS.join(', ')}` },
        { status: 400 }
      )
    }

    const result: Record<string, unknown> = { iso, timestamp: new Date().toISOString() }

    if (dataType === 'lmp' || dataType === 'all') {
      result.lmp = await fetchLMP(iso, node)
    }

    if (dataType === 'generation' || dataType === 'all') {
      result.generation = await fetchGenerationMix(iso)
    }

    if (dataType === 'status' || dataType === 'all') {
      result.grid_status = await fetchGridStatus(iso)
    }

    result.disclaimer = 'Data from public ISO feeds. May be delayed 5-15 minutes. Not for trading decisions.'

    return NextResponse.json(result)
  } catch (error) {
    console.error('ISO API error:', error)
    return NextResponse.json(
      { error: 'ISO data fetch failed', detail: String(error) },
      { status: 500 }
    )
  }
}
