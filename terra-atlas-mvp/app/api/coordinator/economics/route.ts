// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { calculateIRA } from '@/lib/incentives/ira-calculator'
import { generateProForma } from '@/lib/economics/pro-forma'

/**
 * GET /api/coordinator/economics
 *
 * National Energy Transition Coordinator — Economics & Incentive API
 *
 * Computes IRA tax credits, LCOE, and 20-year pro forma for a proposed project.
 *
 * Query params:
 *   type          - Project type (required)
 *   capacity_mw   - Capacity in MW (required)
 *   capital_cost   - Total cost in USD (optional, estimated if missing)
 *   state          - State code (required for energy community check)
 *   prevailing_wage - true/false (default: true)
 *   apprenticeship  - true/false (default: true)
 *   domestic_content - true/false (default: false)
 *   energy_community - true/false (default: false)
 *   low_income       - true/false (default: false)
 *   ppa_price        - PPA price $/MWh (optional)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const type = searchParams.get('type') || ''
    const capacityMw = parseFloat(searchParams.get('capacity_mw') || '')
    const state = searchParams.get('state') || 'TX'

    if (!type || isNaN(capacityMw)) {
      return NextResponse.json(
        { error: 'Required: type, capacity_mw' },
        { status: 400 }
      )
    }

    // Estimate capital cost if not provided ($/kW by technology)
    const costPerKw: Record<string, number> = {
      solar: 1200, wind: 1500, nuclear: 6500, hydro: 2500,
      storage: 1400, geothermal: 4000, 'offshore wind': 4500,
    }
    const typeKey = type.toLowerCase()
    const defaultCostPerKw = costPerKw[typeKey] || 2000
    const capitalCost = parseFloat(searchParams.get('capital_cost') || '') || (capacityMw * 1000 * defaultCostPerKw)

    const parseBool = (key: string, def: boolean) => {
      const v = searchParams.get(key)
      return v === null ? def : v === 'true'
    }

    // Capacity factor by type
    const capFactors: Record<string, number> = {
      solar: 0.25, wind: 0.35, nuclear: 0.92, hydro: 0.40,
      geothermal: 0.90, storage: 0.15, 'offshore wind': 0.45,
    }
    const capFactor = capFactors[typeKey] || 0.30
    const annualGenMwh = capacityMw * capFactor * 8760

    // 1. IRA incentives
    const ira = calculateIRA({
      project_type: type,
      capacity_mw: capacityMw,
      capital_cost: capitalCost,
      annual_generation_mwh: annualGenMwh,
      prevailing_wage: parseBool('prevailing_wage', true),
      apprenticeship: parseBool('apprenticeship', true),
      domestic_content: parseBool('domestic_content', false),
      energy_community: parseBool('energy_community', false),
      low_income: parseBool('low_income', false),
      state,
    })

    // 2. Pro forma with IRA credits applied
    const ppaPrice = parseFloat(searchParams.get('ppa_price') || '') || 45
    const proForma = generateProForma({
      project_type: type,
      capacity_mw: capacityMw,
      capital_cost: capitalCost,
      annual_generation_mwh: annualGenMwh,
      ppa_price_per_mwh: ppaPrice,
      ira_credit_value: ira.credit_type === 'ITC' ? ira.estimated_value : 0,
      ira_annual_ptc: ira.annual_ptc_value || 0,
    })

    return NextResponse.json({
      query: {
        type, capacity_mw: capacityMw, capital_cost: capitalCost,
        state, ppa_price: ppaPrice,
      },
      ira_incentives: ira,
      pro_forma: {
        lcoe: proForma.lcoe,
        irr: proForma.irr,
        npv: proForma.npv,
        payback_years: proForma.payback_years,
        summary: proForma.summary,
        sensitivity: proForma.sensitivity,
        // First 5 years + last year of cashflows (full 25-year available on request)
        cashflow_preview: [
          ...proForma.annual_cashflows.slice(0, 5),
          proForma.annual_cashflows[proForma.annual_cashflows.length - 1],
        ],
      },
      disclaimer: 'This is an estimate for informational purposes only. Not legal or tax advice. Eligibility requires IRS determination.',
    })
  } catch (error) {
    console.error('Economics API error:', error)
    return NextResponse.json(
      { error: 'Economics calculation failed', detail: String(error) },
      { status: 500 }
    )
  }
}
