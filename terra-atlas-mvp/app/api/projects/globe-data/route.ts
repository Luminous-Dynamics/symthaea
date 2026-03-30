// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { requireServerEnv, serverEnv } from '@/lib/env.server'
import logger from '@/lib/logger'

const supabaseUrl = serverEnv.NEXT_PUBLIC_SUPABASE_URL
const supabaseServiceKey = requireServerEnv('SUPABASE_SERVICE_ROLE_KEY')
const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)
const MAX_PAGE_SIZE = 2000
const DEFAULT_LIMIT = 1200
const CACHE_TTL_MS = 5 * 60 * 1000

type CacheEntry = {
  expiresAt: number
  payload: any
}

const globeCache = new Map<string, CacheEntry>()

async function getAuthenticatedUser(request: NextRequest) {
  const authHeader = request.headers.get('authorization')
  if (!authHeader?.startsWith('Bearer ')) return null

  const token = authHeader.replace('Bearer ', '')
  const { data, error } = await supabaseAdmin.auth.getUser(token)
  if (error || !data.user) return null
  return data.user
}

export async function GET(request: NextRequest) {
  try {
    const user = await getAuthenticatedUser(request)
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const searchParams = request.nextUrl.searchParams
    const requestedLimit = parseInt(searchParams.get('limit') || `${DEFAULT_LIMIT}`, 10)
    const requestedOffset = parseInt(searchParams.get('offset') || '0', 10)
    const limit = Math.min(
      Number.isFinite(requestedLimit) && requestedLimit > 0 ? requestedLimit : DEFAULT_LIMIT,
      MAX_PAGE_SIZE
    )
    const offset = Math.max(Number.isFinite(requestedOffset) ? requestedOffset : 0, 0)
    const cacheKey = `${limit}:${offset}`
    const cached = globeCache.get(cacheKey)

    if (cached && cached.expiresAt > Date.now()) {
      return NextResponse.json(cached.payload)
    }

    const { data: sites, error, count } = await supabaseAdmin
      .from('sites')
      .select('id, name, type, latitude, longitude, power_mw, status', { count: 'exact' })
      .not('latitude', 'is', null)
      .not('longitude', 'is', null)
      .order('power_mw', { ascending: false, nullsFirst: false })
      .range(offset, offset + limit - 1)

    if (error) {
      logger.error('Globe data error:', error)
      return NextResponse.json({ sites: getDemoSites() }, { status: 500 })
    }

    const formattedSites = (sites || []).map((site: any) => ({
      id: site.id,
      name: site.name,
      lat: site.latitude,
      lng: site.longitude,
      type: site.type,
      power: site.power_mw || 0,
      status: site.status || 'operational',
      color: getColorByType(site.type),
      size: Math.min(Math.max((site.power_mw || 0) / 100, 0.5), 5)
    }))

    const payload = {
      sites: formattedSites.length > 0 ? formattedSites : getDemoSites(),
      total: count ?? formattedSites.length,
      limit,
      offset
    }

    globeCache.set(cacheKey, { payload, expiresAt: Date.now() + CACHE_TTL_MS })

    return NextResponse.json(payload)
  } catch (error) {
    logger.error('Globe data error:', error)
    return NextResponse.json({ sites: getDemoSites() }, { status: 500 })
  }
}

function getColorByType(type: string): string {
  const colors: { [key: string]: string } = {
    solar: '#FCD34D',      // Amber/Yellow
    wind: '#60A5FA',       // Cyan/Blue
    hydro: '#34D399',      // Emerald/Green
    geothermal: '#F87171', // Red/Orange
    nuclear: '#A78BFA',    // Purple
    storage: '#FB923C',    // Orange
    biomass: '#84CC16',    // Lime green
    hybrid: '#EC4899'      // Pink (mixed/multi-technology)
  }
  return colors[type] || '#10B981' // Default emerald
}

function getDemoSites() {
  // Fallback demo data showcasing all energy types globally
  return [
    // Solar
    { id: 1, name: 'Mojave Solar Park', lat: 35.0, lng: -116.5, type: 'solar', power: 280, color: '#FCD34D', size: 2.8, status: 'operational' },
    { id: 8, name: 'Bhadla Solar Park', lat: 27.5, lng: 71.9, type: 'solar', power: 2245, color: '#FCD34D', size: 5, status: 'operational' },
    { id: 10, name: 'Fukushima Renewable', lat: 37.5, lng: 141.0, type: 'solar', power: 100, color: '#FCD34D', size: 1, status: 'operational' },
    { id: 17, name: 'Noor Complex', lat: 31.0, lng: -6.7, type: 'solar', power: 580, color: '#FCD34D', size: 4, status: 'operational' },

    // Wind
    { id: 2, name: 'London Array', lat: 51.6, lng: 1.5, type: 'wind', power: 630, color: '#60A5FA', size: 5, status: 'operational' },
    { id: 7, name: 'Gansu Wind Farm', lat: 40.5, lng: 95.8, type: 'wind', power: 20000, color: '#60A5FA', size: 5, status: 'operational' },
    { id: 14, name: 'Hornsea Wind Farm', lat: 54.0, lng: 1.5, type: 'wind', power: 1218, color: '#60A5FA', size: 5, status: 'operational' },
    { id: 18, name: 'Alta Wind Energy', lat: 34.6, lng: -118.3, type: 'wind', power: 1548, color: '#60A5FA', size: 5, status: 'operational' },

    // Hydro
    { id: 3, name: 'Three Gorges Dam', lat: 30.8, lng: 111.0, type: 'hydro', power: 22500, color: '#34D399', size: 5, status: 'operational' },
    { id: 9, name: 'Grand Coulee Dam', lat: 47.9, lng: -119.0, type: 'hydro', power: 6809, color: '#34D399', size: 5, status: 'operational' },
    { id: 15, name: 'Itaipu Dam', lat: -25.4, lng: -54.6, type: 'hydro', power: 14000, color: '#34D399', size: 5, status: 'operational' },

    // Geothermal
    { id: 4, name: 'Hellisheiði Station', lat: 64.0, lng: -21.4, type: 'geothermal', power: 303, color: '#F87171', size: 3, status: 'operational' },
    { id: 16, name: 'Geysers Complex', lat: 38.8, lng: -122.8, type: 'geothermal', power: 1517, color: '#F87171', size: 5, status: 'operational' },

    // Nuclear
    { id: 5, name: 'Vogtle Nuclear', lat: 33.1, lng: -81.8, type: 'nuclear', power: 2234, color: '#A78BFA', size: 5, status: 'construction' },
    { id: 13, name: 'Barakah Nuclear', lat: 24.5, lng: 52.2, type: 'nuclear', power: 5600, color: '#A78BFA', size: 5, status: 'operational' },

    // Storage
    { id: 6, name: 'Hornsdale Battery', lat: -32.5, lng: 138.5, type: 'storage', power: 150, color: '#FB923C', size: 1.5, status: 'operational' },
    { id: 12, name: 'Gateway Energy', lat: 33.7, lng: -117.8, type: 'storage', power: 230, color: '#FB923C', size: 2.3, status: 'operational' },

    // Biomass
    { id: 11, name: 'Drax Power Station', lat: 53.7, lng: -1.0, type: 'biomass', power: 2600, color: '#84CC16', size: 5, status: 'operational' },
    { id: 19, name: 'Alholmens Kraft', lat: 63.7, lng: 22.7, type: 'biomass', power: 265, color: '#84CC16', size: 2.6, status: 'operational' },

    // Hybrid
    { id: 20, name: 'Kennedy Energy Park', lat: -28.8, lng: 143.4, type: 'hybrid', power: 73, color: '#EC4899', size: 0.7, status: 'operational' },
    { id: 21, name: 'Blythe Solar+Storage', lat: 33.6, lng: -114.6, type: 'hybrid', power: 485, color: '#EC4899', size: 4, status: 'operational' },
  ]
}
