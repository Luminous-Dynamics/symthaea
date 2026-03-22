// Terra Atlas Discovery API - FERC Queue Intelligence

import logger from '@/lib/logger'
import { NextRequest, NextResponse } from 'next/server';

const REGIONAL_DROPOUT_RATES: Record<string, number> = {
  ERCOT: 0.68,
  PJM: 0.75,
  CAISO: 0.72,
  SPP: 0.70,
  MISO: 0.73,
  'ISO-NE': 0.71,
  NYISO: 0.74,
  DEFAULT: 0.72
};

const REGIONAL_PROCESSING_TIMES: Record<string, number> = {
  ERCOT: 1.8,
  PJM: 2.5,
  CAISO: 2.2,
  SPP: 1.9,
  MISO: 2.3,
  'ISO-NE': 2.1,
  NYISO: 2.4,
  DEFAULT: 2.2
};

export async function GET(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;
    const region = (params.get('region') || 'DEFAULT').toUpperCase();
    const queuePosition = parseInt(params.get('position') || '847', 10);

    const dropoutRate = REGIONAL_DROPOUT_RATES[region] ?? REGIONAL_DROPOUT_RATES.DEFAULT;
    const expectedSurvivorsAhead = Math.floor((queuePosition - 1) * (1 - dropoutRate));
    const processingTime =
      REGIONAL_PROCESSING_TIMES[region] ?? REGIONAL_PROCESSING_TIMES.DEFAULT;
    const estimatedTimeYears = (expectedSurvivorsAhead / 100) * processingTime;

    const opportunities = buildOpportunities(region, queuePosition);

    return NextResponse.json({
      queue_analysis: {
        your_position: queuePosition,
        projects_ahead: queuePosition - 1,
        typical_dropout_rate: `${(dropoutRate * 100).toFixed(0)}%`,
        expected_survivors_ahead: expectedSurvivorsAhead,
        estimated_time_to_process: `${estimatedTimeYears.toFixed(1)} years`
      },
      acceleration_opportunities: opportunities
    });
  } catch (error) {
    logger.error('Error analyzing queue:', error);
    return NextResponse.json({ error: 'Failed to analyze queue' }, { status: 500 });
  }
}

function buildOpportunities(region: string, queuePosition: number) {
  const opportunities = [];

  if (['ERCOT', 'CAISO', 'PJM'].includes(region)) {
    opportunities.push({
      strategy: `Join ${
        region === 'ERCOT'
          ? 'West Texas'
          : region === 'CAISO'
          ? 'Central Valley'
          : 'Mid-Atlantic'
      } Corridor`,
      time_savings: '2-3 years',
      cost_savings: '$30-50M',
      probability_of_success: '85%'
    });
  }

  opportunities.push(
    {
      strategy: 'Phase project into 100MW segments',
      time_savings: '1.5 years',
      cost_savings: '$15-25M',
      probability_of_success: '75%'
    },
    {
      strategy: 'Add battery storage for grid services',
      time_savings: '1 year',
      cost_savings: '$10-20M',
      probability_of_success: '70%'
    }
  );

  if (queuePosition < 500) {
    opportunities.push({
      strategy: 'Fast-track study process available',
      time_savings: '6 months',
      cost_savings: '$5-10M',
      probability_of_success: '90%'
    });
  }

  return opportunities;
}
