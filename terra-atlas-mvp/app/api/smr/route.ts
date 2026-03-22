import { NextRequest, NextResponse } from 'next/server'
import smrData from '@/data/smr-projects.json'
import logger from '@/lib/logger'

// GET /api/smr - Fetch SMR projects with filtering
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const developer = searchParams.get('developer')
    const status = searchParams.get('status')
    const country = searchParams.get('country')
    const minInvestment = searchParams.get('minInvestment')
    const maxInvestment = searchParams.get('maxInvestment')
    const investmentType = searchParams.get('investmentType')
    const sort = searchParams.get('sort') || 'roi'
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')

    let projects = [...smrData.smr_projects]

    // Apply filters
    if (developer) {
      projects = projects.filter(p => 
        p.developer.toLowerCase().includes(developer.toLowerCase())
      )
    }

    if (status) {
      projects = projects.filter(p => 
        p.status.toLowerCase() === status.toLowerCase()
      )
    }

    if (country) {
      projects = projects.filter(p => 
        p.country.toLowerCase() === country.toLowerCase()
      )
    }

    if (minInvestment) {
      const min = parseInt(minInvestment)
      projects = projects.filter(p => p.min_investment >= min)
    }

    if (maxInvestment) {
      const max = parseInt(maxInvestment)
      projects = projects.filter(p => p.min_investment <= max)
    }

    if (investmentType) {
      projects = projects.filter(p => 
        p.investment_type.toLowerCase() === investmentType.toLowerCase()
      )
    }

    // Apply sorting
    projects.sort((a, b) => {
      switch(sort) {
        case 'roi':
          return b.roi_percentage - a.roi_percentage
        case 'capacity':
          return b.capacity_mw - a.capacity_mw
        case 'investment':
          return b.total_investment - a.total_investment
        case 'needed':
          return b.investment_needed - a.investment_needed
        case 'completion':
          return a.commercial_operation.localeCompare(b.commercial_operation)
        default:
          return 0
      }
    })

    // Apply pagination
    const total = projects.length
    projects = projects.slice(offset, offset + limit)

    // Calculate aggregated stats
    const stats = {
      total_projects: smrData.smr_projects.length,
      total_capacity_mw: smrData.metadata.total_capacity_mw,
      total_investment_needed: smrData.metadata.total_investment_needed,
      average_roi: smrData.metadata.average_roi,
      countries: [...new Set(smrData.smr_projects.map(p => p.country))].length,
      developers: [...new Set(smrData.smr_projects.map(p => p.developer))].length,
      status_breakdown: {
        construction: smrData.smr_projects.filter(p => p.status === 'Construction').length,
        development: smrData.smr_projects.filter(p => p.status === 'Development').length,
        planning: smrData.smr_projects.filter(p => p.status === 'Planning').length
      }
    }

    return NextResponse.json({
      projects,
      stats,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total
      }
    })
  } catch (error) {
    logger.error('Error fetching SMR projects:', error)
    return NextResponse.json(
      { error: 'Failed to fetch SMR projects' },
      { status: 500 }
    )
  }
}

// GET specific SMR project by ID
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { id } = body

    const project = smrData.smr_projects.find(p => p.id === id)
    
    if (!project) {
      return NextResponse.json(
        { error: 'SMR project not found' },
        { status: 404 }
      )
    }

    return NextResponse.json(project)
  } catch (error) {
    logger.error('Error fetching SMR project:', error)
    return NextResponse.json(
      { error: 'Failed to fetch SMR project' },
      { status: 500 }
    )
  }
}