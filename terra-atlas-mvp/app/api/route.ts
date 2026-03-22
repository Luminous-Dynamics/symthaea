import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({
    name: 'Terra Atlas API',
    version: '0.1.0',
    description: 'The Planetary Nervous System - Real-time environmental data API',
    status: 'operational',
    endpoints: {
      '/api/data/fires': 'NASA FIRMS active fire detections',
      '/api/data/earthquakes': 'USGS earthquake monitoring',
      '/api/data/weather': 'OpenWeatherMap climate data',
      '/api/data/emissions': 'Carbon Monitor emissions tracking'
    },
    documentation: 'https://terra-atlas.earth/docs',
    rate_limits: {
      anonymous: '1000 requests/day',
      authenticated: '10000 requests/day'
    },
    data_sources: [
      {
        name: 'NASA FIRMS',
        update_frequency: '3 hours',
        license: 'Public Domain'
      },
      {
        name: 'USGS',
        update_frequency: 'Real-time',
        license: 'Public Domain'
      },
      {
        name: 'OpenWeatherMap',
        update_frequency: 'Hourly',
        license: 'CC BY-SA 4.0'
      },
      {
        name: 'Carbon Monitor',
        update_frequency: 'Daily',
        license: 'CC BY 4.0'
      }
    ],
    contact: 'info@terra-atlas.earth'
  })
}