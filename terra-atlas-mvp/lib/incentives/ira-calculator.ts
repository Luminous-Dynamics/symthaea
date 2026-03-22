// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * IRA (Inflation Reduction Act) Incentive Calculator
 *
 * Rules-based calculator mapping project attributes to IRA tax credit provisions.
 * Based on IRC Sections 45, 48, 45V, 45X, 45Q as enacted August 2022.
 *
 * DISCLAIMER: This is an estimate for informational purposes only.
 * Not legal or tax advice. Eligibility requires IRS determination.
 * Consult qualified tax counsel before relying on these calculations.
 */

export interface IRAInput {
  project_type: string    // Solar, Wind, Nuclear, Storage, Hydro, Hydrogen, CCS
  capacity_mw: number
  capital_cost: number    // Total project cost in USD
  annual_generation_mwh?: number
  prevailing_wage: boolean       // Meets prevailing wage requirements
  apprenticeship: boolean        // Meets registered apprenticeship requirements
  domestic_content: boolean      // Meets domestic content requirements
  energy_community: boolean      // Located in energy community (coal closure, brownfield)
  low_income: boolean            // Benefits low-income communities
  state: string
}

export interface IRAResult {
  /** Primary credit type: ITC or PTC */
  credit_type: 'ITC' | 'PTC' | 'Both'
  /** Base credit rate before adders */
  base_rate: number
  /** Full credit rate with all qualifying adders */
  full_rate: number
  /** Individual adder breakdown */
  adders: IRAAdder[]
  /** Estimated total credit value in USD */
  estimated_value: number
  /** Annual PTC value (if applicable) */
  annual_ptc_value?: number
  /** Credit duration in years */
  credit_duration_years: number
  /** Applicable IRC sections */
  sections: string[]
  /** Warnings and caveats */
  warnings: string[]
  /** Confidence in the estimate */
  confidence: 'high' | 'medium' | 'low'
}

export interface IRAAdder {
  name: string
  rate: number
  qualified: boolean
  description: string
}

// IRA PTC rate (2024 inflation-adjusted, $/kWh)
const PTC_BASE = 0.0055
const PTC_BONUS = 0.0275  // 5x with prevailing wage + apprenticeship

// Capacity factors by technology (approximate)
const CAPACITY_FACTORS: Record<string, number> = {
  solar: 0.25,
  wind: 0.35,
  'offshore wind': 0.45,
  nuclear: 0.92,
  hydro: 0.40,
  geothermal: 0.90,
  storage: 0.15,  // round-trip efficiency adjusted
}

export function calculateIRA(input: IRAInput): IRAResult {
  const type = input.project_type.toLowerCase()
  const warnings: string[] = []
  const sections: string[] = []

  // Determine primary credit: ITC (Section 48) vs PTC (Section 45)
  // Solar/Storage/Geothermal typically use ITC
  // Wind/Nuclear typically use PTC
  // Projects can elect either (IRA allows choice for most technologies)
  const useITC = type.includes('solar') || type.includes('storage') || type.includes('geothermal')
  const usePTC = type.includes('wind') || type.includes('nuclear') || type.includes('hydro')
  const creditType = useITC ? 'ITC' : 'PTC'

  // Base rates
  const meetsLaborStandards = input.prevailing_wage && input.apprenticeship
  const itcBase = meetsLaborStandards ? 0.30 : 0.06
  const ptcRate = meetsLaborStandards ? PTC_BONUS : PTC_BASE

  // Adders
  const adders: IRAAdder[] = []

  // Prevailing wage + apprenticeship (5x multiplier)
  adders.push({
    name: 'Prevailing Wage & Apprenticeship',
    rate: meetsLaborStandards ? (useITC ? 0.24 : PTC_BONUS - PTC_BASE) : 0,
    qualified: meetsLaborStandards,
    description: meetsLaborStandards
      ? 'Meets prevailing wage and registered apprenticeship requirements (5x base credit)'
      : 'Not meeting labor standards — base credit only (6% ITC or $0.0055/kWh PTC)',
  })

  // Domestic content bonus (+10% ITC or +10% PTC)
  adders.push({
    name: 'Domestic Content',
    rate: input.domestic_content ? 0.10 : 0,
    qualified: input.domestic_content,
    description: input.domestic_content
      ? 'Steel/iron from US manufacturing, 40%+ manufactured components domestic'
      : 'Does not meet domestic content threshold (40% manufactured components)',
  })

  // Energy community bonus (+10% ITC or +10% PTC)
  adders.push({
    name: 'Energy Community',
    rate: input.energy_community ? 0.10 : 0,
    qualified: input.energy_community,
    description: input.energy_community
      ? 'Located in coal closure community, brownfield, or fossil fuel employment area'
      : 'Not in designated energy community (check Treasury community list)',
  })

  // Low-income bonus (+10% or +20% ITC only)
  if (useITC) {
    adders.push({
      name: 'Low-Income Community',
      rate: input.low_income ? 0.10 : 0,
      qualified: input.low_income,
      description: input.low_income
        ? 'Located in or benefiting low-income community (Section 48(e))'
        : 'Not qualifying as low-income community benefit',
    })
  }

  // Calculate total rate
  const adderTotal = adders.reduce((s, a) => s + a.rate, 0)

  let baseRate: number
  let fullRate: number
  let estimatedValue: number
  let annualPtcValue: number | undefined
  let creditDuration: number

  if (useITC) {
    // ITC: percentage of capital cost
    sections.push('IRC §48 (Investment Tax Credit)')
    baseRate = 0.06
    fullRate = itcBase + adderTotal - (meetsLaborStandards ? 0.24 : 0) // don't double count PWA
    fullRate = itcBase + (input.domestic_content ? 0.10 : 0) + (input.energy_community ? 0.10 : 0) + (input.low_income ? 0.10 : 0)
    estimatedValue = input.capital_cost * fullRate
    creditDuration = 1 // ITC is one-time
  } else {
    // PTC: per kWh over 10 years
    // Domestic content and energy community each add 10% to the PTC rate
    sections.push('IRC §45 (Production Tax Credit)')
    baseRate = PTC_BASE
    let ptcFull = ptcRate
    if (input.domestic_content) ptcFull += ptcRate * 0.10  // +10% of PTC rate
    if (input.energy_community) ptcFull += ptcRate * 0.10  // +10% of PTC rate
    fullRate = ptcFull

    const capFactor = CAPACITY_FACTORS[type] || 0.30
    const annualMwh = input.annual_generation_mwh || (input.capacity_mw * capFactor * 8760)
    annualPtcValue = annualMwh * 1000 * ptcFull // MWh → kWh (×1000), then × $/kWh
    estimatedValue = annualPtcValue * 10 // 10-year PTC duration
    creditDuration = 10
  }

  // Special provisions
  if (type.includes('nuclear')) {
    sections.push('IRC §45J (Nuclear Production Tax Credit)')
    warnings.push('Nuclear projects may also qualify for §45J existing nuclear PTC ($15/MWh)')
  }
  if (type.includes('hydrogen')) {
    sections.push('IRC §45V (Clean Hydrogen Production Tax Credit)')
    warnings.push('§45V credit ($0.60-$3.00/kg) depends on lifecycle carbon intensity')
  }
  if (type.includes('ccs') || type.includes('carbon capture')) {
    sections.push('IRC §45Q (Carbon Capture Tax Credit)')
    warnings.push('§45Q credit ($85/ton for geological storage) requires EPA Class VI permit')
  }

  // Standard disclaimers
  warnings.push('This is an estimate. Actual eligibility requires IRS determination.')
  warnings.push('Prevailing wage/apprenticeship must be documented throughout construction.')
  if (input.domestic_content) {
    warnings.push('Domestic content certification requires detailed supply chain documentation.')
  }

  // Confidence
  const confidence = (meetsLaborStandards && (useITC || usePTC))
    ? 'high' as const
    : 'medium' as const

  return {
    credit_type: creditType as 'ITC' | 'PTC',
    base_rate: Math.round(baseRate * 10000) / 10000,
    full_rate: Math.round(fullRate * 10000) / 10000,
    adders,
    estimated_value: Math.round(estimatedValue),
    annual_ptc_value: annualPtcValue ? Math.round(annualPtcValue) : undefined,
    credit_duration_years: creditDuration,
    sections,
    warnings,
    confidence,
  }
}
