// Terra Atlas Discovery API - Find Transmission Capacity
// TypeScript implementation for Next.js API routes

import type { NextApiRequest, NextApiResponse } from 'next';
import { db } from '../../../lib/drizzle/db';
import { transmissionLines } from '../../../lib/drizzle/schema-energy';
import { and, gte, lte } from 'drizzle-orm';

interface TransmissionLine {
  line: string;
  distance_miles: number;
  available_capacity_mw: number;
  voltage_kv: number;
  owner: string;
  upgrade_status: string;
}

interface Substation {
  name: string;
  distance_miles: number;
  available_bays: number;
  recent_connections: number;
}

interface TransmissionResponse {
  nearbyTransmission: TransmissionLine[];
  substations: Substation[];
}

// Helper function to calculate distance between two points
function calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 3959; // Earth's radius in miles
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return Math.round(R * c * 10) / 10;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<TransmissionResponse | { error: string }>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { lat, lng, radius = '50' } = req.query;
    const latitude = parseFloat(lat as string);
    const longitude = parseFloat(lng as string);
    const searchRadius = parseFloat(radius as string);
    
    if (isNaN(latitude) || isNaN(longitude)) {
      return res.status(400).json({ error: 'Invalid coordinates' });
    }
    
    // Calculate bounding box for search
    const latDelta = searchRadius / 69; // ~69 miles per degree latitude
    const lngDelta = searchRadius / (69 * Math.cos(latitude * Math.PI / 180));
    
    const nearbyTransmission = await db
      .select()
      .from(transmissionLines)
      .where(
        and(
          gte(transmissionLines.startLat, (latitude - latDelta).toString()),
          lte(transmissionLines.startLat, (latitude + latDelta).toString()),
          gte(transmissionLines.startLng, (longitude - lngDelta).toString()),
          lte(transmissionLines.startLng, (longitude + lngDelta).toString())
        )
      )
      .limit(5);
    
    // Mock some substations for demo (in production, would query substations table)
    const mockSubstations: Substation[] = [
      {
        name: 'Sweetwater Substation',
        distance_miles: 12.5,
        available_bays: 2,
        recent_connections: 3
      },
      {
        name: 'Big Spring Collector Station',
        distance_miles: 18.3,
        available_bays: 1,
        recent_connections: 5
      }
    ];
    
    res.status(200).json({
      nearbyTransmission: nearbyTransmission.map(line => ({
        line: line.name || 'Unnamed 345kV Line',
        distance_miles: calculateDistance(
          latitude, 
          longitude,
          parseFloat(line.startLat || '0'), 
          parseFloat(line.startLng || '0')
        ),
        available_capacity_mw: parseFloat(line.capacityMw || '230'),
        voltage_kv: line.voltagekV || 345,
        owner: line.owner || 'ERCOT',
        upgrade_status: 'Planned 2026'
      })),
      substations: mockSubstations
    });
  } catch (error) {
    console.error('Error finding transmission capacity:', error);
    res.status(500).json({ error: 'Failed to find transmission capacity' });
  }
}