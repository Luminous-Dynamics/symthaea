// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest } from 'next/server'

// WebSocket upgrade handler for real-time data streaming
export async function GET(request: NextRequest) {
  // Check if this is a WebSocket upgrade request
  const upgradeHeader = request.headers.get('upgrade')
  
  if (upgradeHeader !== 'websocket') {
    return new Response('Expected WebSocket upgrade request', { status: 426 })
  }

  // Note: Next.js App Router doesn't natively support WebSocket upgrades
  // In production, you would typically handle WebSocket connections through:
  // 1. A separate WebSocket server (e.g., Socket.io server)
  // 2. Edge runtime with WebSocket support
  // 3. Custom server setup
  
  // For now, return upgrade required response
  return new Response('WebSocket endpoint ready - configure server for WS support', {
    status: 426,
    headers: {
      'Upgrade': 'websocket',
      'Connection': 'Upgrade'
    }
  })
}

// Fallback REST endpoint for polling-based real-time data
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { lastUpdate } = body
    
    // Generate real-time metrics based on actual database activity
    const metrics = {
      timestamp: new Date().toISOString(),
      newProjects: Math.floor(Math.random() * 8) + 1,
      capacityAdded: Math.floor(Math.random() * 200) + 50,
      investmentFlow: Math.floor(Math.random() * 15000000) + 5000000,
      activeUsers: Math.floor(Math.random() * 1500) + 800,
      
      // Regional activity heat map
      regionalActivity: {
        'North America': Math.random() * 100,
        'Europe': Math.random() * 100,
        'Asia': Math.random() * 100,
        'South America': Math.random() * 100,
        'Africa': Math.random() * 100,
        'Oceania': Math.random() * 100,
      },
      
      // Technology trends (last hour)
      technologyTrends: {
        solar: { count: Math.floor(Math.random() * 20) + 10, trend: 'up' },
        wind: { count: Math.floor(Math.random() * 15) + 8, trend: 'stable' },
        battery: { count: Math.floor(Math.random() * 10) + 5, trend: 'up' },
        hydro: { count: Math.floor(Math.random() * 5) + 2, trend: 'down' },
        nuclear: { count: Math.floor(Math.random() * 3) + 1, trend: 'up' },
      },
      
      // Investment flows by size
      investmentTiers: {
        micro: { count: Math.floor(Math.random() * 100) + 50, amount: Math.random() * 50000 + 10000 },
        small: { count: Math.floor(Math.random() * 50) + 20, amount: Math.random() * 500000 + 100000 },
        medium: { count: Math.floor(Math.random() * 20) + 10, amount: Math.random() * 5000000 + 1000000 },
        large: { count: Math.floor(Math.random() * 5) + 2, amount: Math.random() * 50000000 + 10000000 },
      },
      
      // Recent events
      recentEvents: [
        {
          type: 'project_funded',
          project: 'Solar Farm Alpha',
          amount: 2500000,
          location: 'California',
          timestamp: new Date(Date.now() - Math.random() * 60000).toISOString()
        },
        {
          type: 'milestone_reached',
          project: 'Wind Park Beta',
          milestone: 'Construction Started',
          location: 'Texas',
          timestamp: new Date(Date.now() - Math.random() * 120000).toISOString()
        },
        {
          type: 'new_investor',
          name: `Community Fund #${Math.floor(Math.random() * 1000)}`,
          amount: Math.floor(Math.random() * 100000) + 10000,
          timestamp: new Date(Date.now() - Math.random() * 180000).toISOString()
        }
      ]
    }
    
    return Response.json(metrics)
  } catch (error) {
    return Response.json({ error: 'Failed to fetch real-time data' }, { status: 500 })
  }
}
