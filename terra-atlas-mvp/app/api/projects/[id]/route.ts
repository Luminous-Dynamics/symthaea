// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import logger from '@/lib/logger'
import { db } from '@/lib/drizzle/db'
import { energyProjects } from '@/lib/drizzle/schema-energy'
import { eq } from 'drizzle-orm'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params

    // Get single project by ID
    const [project] = await db
      .select({
        id: energyProjects.id,
        name: energyProjects.name,
        type: energyProjects.projectType,
        developer: energyProjects.developer,
        owner_type: energyProjects.owner,
        state: energyProjects.state,
        county: energyProjects.city,
        region: energyProjects.state,
        country: energyProjects.country,
        latitude: energyProjects.latitude,
        longitude: energyProjects.longitude,
        capacity_mw: energyProjects.capacityMw,
        energy_source: energyProjects.projectType,
        technology_type: energyProjects.subType,
        status: energyProjects.status,
        operational: energyProjects.status,
        year_completed: energyProjects.codDate,
        investment: energyProjects.totalCostMillion,
        investment_needed: energyProjects.totalCostMillion,
        annual_revenue_potential: energyProjects.annualGenerationGwh,
        co2_saved_annual: energyProjects.co2AvoidedTonsYear,
        expected_irr: energyProjects.expectedIrr,
        created_at: energyProjects.createdAt,
      })
      .from(energyProjects)
      .where(eq(energyProjects.id, id))
      .limit(1)

    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Add calculated fields
    const capacityMw = Number(project.capacity_mw) || 0
    const investmentNeeded = Number(project.investment_needed) || 0
    const expectedIrr = Number(project.expected_irr) || (project.type === 'Solar' ? 12 : 11)

    const enhancedProject = {
      ...project,
      investment_raised: investmentNeeded * 0.35, // 35% raised
      min_investment: 100,
      irr: expectedIrr,
      roi_percentage: expectedIrr,
      completion_date: project.year_completed ? `${project.year_completed}` : '2027-Q3',
      power_offtaker: 'Regional Utility Co.',
      total_homes_powered: Math.round(capacityMw * 750),
      location: project.state ? `${project.state}, ${project.country || 'USA'}` : 'United States',
      description: `This ${capacityMw} MW ${project.type || 'energy'} project represents a significant advancement in renewable energy infrastructure for ${project.state || 'the region'}.`,
    }

    return NextResponse.json(enhancedProject)
  } catch (error) {
    logger.error('Error fetching project:', error)
    return NextResponse.json({ error: 'Failed to fetch project' }, { status: 500 })
  }
}
