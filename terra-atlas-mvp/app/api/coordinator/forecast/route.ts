// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { forecastGeneration } from '@/lib/forecast/generation-forecaster'

/**
 * GET /api/coordinator/forecast
 *
 * National Energy Transition Coordinator — Generation Forecast API
 *
 * Forecasts solar or wind generation using real weather data from Open-Meteo
 * (free, no API key). Physics-based models (PVWatts for solar, IEC power
 * curve for wind).
 *
 * Query params:
 *   lat          - Project latitude (required)
 *   lon          - Project longitude (required)
 *   type         - solar | wind (required)
 *   capacity_mw  - Capacity in MW (required)
 *   hours        - Forecast horizon, 1-168 (default: 48)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const lat = parseFloat(searchParams.get('lat') || '')
    const lon = parseFloat(searchParams.get('lon') || '')
    const type = searchParams.get('type') || ''
    const capacityMw = parseFloat(searchParams.get('capacity_mw') || '')
    const hours = Math.min(parseInt(searchParams.get('hours') || '48'), 168)

    if (isNaN(lat) || isNaN(lon) || isNaN(capacityMw)) {
      return NextResponse.json({ error: 'Required: lat, lon, type, capacity_mw' }, { status: 400 })
    }

    if (type !== 'solar' && type !== 'wind') {
      return NextResponse.json({ error: 'type must be "solar" or "wind"' }, { status: 400 })
    }

    const forecast = await forecastGeneration(type, capacityMw, lat, lon, hours)

    return NextResponse.json({
      query: { lat, lon, type, capacity_mw: capacityMw, hours },
      forecast: {
        ...forecast,
        // Trim hourly to just the summary for large responses
        hourly: forecast.hourly.length > 72
          ? [...forecast.hourly.slice(0, 48), { note: `... ${forecast.hourly.length - 48} more hours available` } as any]
          : forecast.hourly,
      },
    })
  } catch (error) {
    console.error('Forecast API error:', error)
    return NextResponse.json(
      { error: 'Forecast failed', detail: String(error) },
      { status: 500 }
    )
  }
}
