import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'
import logger from '@/lib/logger'

// Check if we have real API keys configured
const hasRealData = {
  fires: !!process.env.FIRMS_API_KEY,
  weather: !!process.env.OPENWEATHER_API_KEY,
  earthquakes: true, // USGS doesn't need API key
  emissions: !!process.env.CARBON_API_KEY,
  'noaa-alerts': true, // NOAA doesn't need API key
  'nasa-eonet': true, // NASA EONET doesn't need API key
  'air-quality': true, // OpenAQ basic access doesn't need API key
  'volcanoes': true, // Demo/synthetic data
  'solar-flares': true // NOAA SWPC doesn't need API key
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ layer: string }> }
) {
  const { layer } = await params

  // Map layer names to data files
  const dataFiles: Record<string, string> = {
    'fires': 'nasa-firms.json',
    'earthquakes': 'usgs-earthquakes.json',
    'weather': 'openweather.json',
    'emissions': 'carbon-monitor.json',
    'noaa-alerts': 'noaa-alerts.json',
    'nasa-eonet': 'nasa-eonet.json',
    'air-quality': 'air-quality.json',
    'volcanoes': 'volcanoes.json',
    'solar-flares': 'solar-flares.json'
  }

  const fileName = dataFiles[layer]
  if (!fileName) {
    return NextResponse.json(
      { error: 'Invalid layer specified' },
      { status: 400 }
    )
  }

  try {
    // Read data from public/data directory
    const filePath = path.join(process.cwd(), 'public', 'data', fileName)
    
    // Check if file exists, if not return empty GeoJSON
    try {
      await fs.access(filePath)
      const data = await fs.readFile(filePath, 'utf-8')
      const jsonData = JSON.parse(data)
      
      // Enrich with metadata about data source
      const enrichedData = {
        ...jsonData,
        metadata: {
          ...jsonData.metadata,
          isRealData: hasRealData[layer as keyof typeof hasRealData] || false,
          dataMode: hasRealData[layer as keyof typeof hasRealData] ? 'live' : 'demo',
          apiKeyPresent: layer === 'earthquakes' ? 'not_required' : 
                        (hasRealData[layer as keyof typeof hasRealData] ? 'yes' : 'no')
        }
      }
      
      return NextResponse.json(enrichedData)
    } catch {
      // Return empty GeoJSON if file doesn't exist
      return NextResponse.json({
        type: 'FeatureCollection',
        metadata: {
          source: layer,
          last_updated: new Date().toISOString(),
          total_features: 0,
          quality_score: 0,
          isRealData: false,
          dataMode: 'empty'
        },
        features: []
      })
    }
  } catch (error) {
    logger.error('Error reading data', error)
    return NextResponse.json(
      { error: 'Failed to load data' },
      { status: 500 }
    )
  }
}