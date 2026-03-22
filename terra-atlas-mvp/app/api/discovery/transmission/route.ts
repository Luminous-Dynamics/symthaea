// Terra Atlas Discovery API - Find Transmission Capacity

import logger from '@/lib/logger'
import { NextRequest, NextResponse } from 'next/server';
import { db } from '../../../../lib/drizzle/db';
import { transmissionLines } from '../../../../lib/drizzle/schema-energy';
import { and, gte, lte } from 'drizzle-orm';

interface Substation {
  name: string;
  distance_miles: number;
  available_bays: number;
  recent_connections: number;
}

function calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 3959;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Math.round(R * c * 10) / 10;
}

export async function GET(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;
    const latitude = Number(params.get('lat'));
    const longitude = Number(params.get('lng'));
    const searchRadius = Number(params.get('radius') || '50');

    if (Number.isNaN(latitude) || Number.isNaN(longitude)) {
      return NextResponse.json({ error: 'Invalid coordinates' }, { status: 400 });
    }

    const latDelta = searchRadius / 69;
    const lngDelta = searchRadius / (69 * Math.cos((latitude * Math.PI) / 180));

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

    return NextResponse.json({
      nearbyTransmission: nearbyTransmission.map((line) => ({
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
    logger.error('Error finding transmission capacity:', error);
    return NextResponse.json({ error: 'Failed to find transmission capacity' }, { status: 500 });
  }
}
