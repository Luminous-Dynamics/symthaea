// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import logger from '@/lib/logger'
// Terra Atlas Portfolio API - Get User Portfolio

import { NextRequest, NextResponse } from 'next/server';
import { db } from '../../../lib/drizzle/db';
import { portfolios, investments, investmentReturns } from '../../../lib/drizzle/schema-investment';
import { energyProjects } from '../../../lib/drizzle/schema-energy';
import { eq, sql } from 'drizzle-orm';
import jwt from 'jsonwebtoken';
import { requireServerEnv } from '@/lib/env.server';

interface PortfolioMetrics {
  totalInvested: number;
  totalReturns: number;
  totalPledged: number;
  activeInvestments: number;
  portfolioIrr: number;
  totalCo2Avoided: number;
  totalMwhGenerated: number;
}

interface InvestmentSummary {
  id: string;
  projectName: string;
  projectType: string;
  location: string;
  amountInvested: number;
  currentValue: number;
  returns: number;
  status: string;
  expectedReturn: number;
  co2Avoided: number;
  mwhGenerated: number;
}

const JWT_SECRET = requireServerEnv('JWT_SECRET');

export async function GET(
  req: NextRequest
) {
  try {
    // Get user from JWT token
    const token = req.headers.get('authorization')?.replace('Bearer ', '');
    if (!token) {
      return NextResponse.json({ 
        success: false, 
        message: 'Authentication required' 
      }, { status: 401 });
    }

    const decoded = jwt.verify(token, JWT_SECRET) as { userId: string };
    const userId = decoded.userId;

    // Get user portfolio
    let [portfolio] = await db
      .select()
      .from(portfolios)
      .where(eq(portfolios.userId, userId))
      .limit(1);

    if (!portfolio) {
      // Create empty portfolio for new user
      const [newPortfolio] = await db
        .insert(portfolios)
        .values({
          userId,
          totalInvested: '0',
          totalReturns: '0',
          totalPledged: '0',
          activeInvestments: 0,
          totalCo2Avoided: '0',
          totalMwhGenerated: '0',
          createdAt: new Date(),
          updatedAt: new Date()
        })
        .returning();
      
      portfolio = newPortfolio;
    }

    // Get all user investments with project details
    const userInvestments = await db
      .select({
        investment: investments,
        project: energyProjects
      })
      .from(investments)
      .innerJoin(energyProjects, eq(investments.projectId, energyProjects.id))
      .where(eq(investments.userId, userId));

    // Get investment returns
    const returns = await db
      .select({
        investmentId: investmentReturns.investmentId,
        totalReturns: sql<number>`SUM(${investmentReturns.amountUsd})`.as('totalReturns')
      })
      .from(investmentReturns)
      .where(
        sql`${investmentReturns.investmentId} IN (
          SELECT id FROM ${investments} WHERE user_id = ${userId}
        )`
      )
      .groupBy(investmentReturns.investmentId);

    // Create returns map
    const returnsMap = new Map(
      returns.map(r => [r.investmentId, parseFloat(r.totalReturns?.toString() || '0')])
    );

    // Process investments
    const investmentSummaries: InvestmentSummary[] = userInvestments.map(({ investment, project }) => {
      const amountInvested = parseFloat(investment.amountUsd);
      const totalReturns = returnsMap.get(investment.id) || 0;
      const currentValue = amountInvested + totalReturns;

      return {
        id: investment.id,
        projectName: project.name,
        projectType: project.projectType,
        location: `${project.state}, ${project.country}`,
        amountInvested,
        currentValue,
        returns: totalReturns,
        status: investment.status,
        expectedReturn: parseFloat(investment.expectedReturn || '0'),
        co2Avoided: parseFloat(investment.co2AvoidedTons || '0'),
        mwhGenerated: parseFloat(investment.mwhGenerated || '0')
      };
    });

    // Calculate allocations
    const byTechnology: Record<string, number> = {};
    const byRegion: Record<string, number> = {};
    const byStatus: Record<string, number> = {};

    investmentSummaries.forEach(inv => {
      // By technology
      byTechnology[inv.projectType] = (byTechnology[inv.projectType] || 0) + inv.amountInvested;
      
      // By region (state)
      const region = inv.location.split(',')[0];
      byRegion[region] = (byRegion[region] || 0) + inv.amountInvested;
      
      // By status
      byStatus[inv.status] = (byStatus[inv.status] || 0) + inv.amountInvested;
    });

    // Calculate performance metrics
    const totalInvested = investmentSummaries.reduce((sum, inv) => sum + inv.amountInvested, 0);
    const totalReturnsAmount = investmentSummaries.reduce((sum, inv) => sum + inv.returns, 0);
    const totalValue = investmentSummaries.reduce((sum, inv) => sum + inv.currentValue, 0);
    
    const totalReturn = totalInvested > 0 ? ((totalValue - totalInvested) / totalInvested * 100) : 0;
    
    // Find best and worst performers
    const sortedByReturn = investmentSummaries
      .filter(inv => inv.amountInvested > 0)
      .sort((a, b) => (b.returns / b.amountInvested) - (a.returns / a.amountInvested));
    
    const bestPerformer = sortedByReturn[0]?.projectName || 'N/A';
    const worstPerformer = sortedByReturn[sortedByReturn.length - 1]?.projectName || 'N/A';

    // Calculate impact metrics
    const co2AvoidedTotal = investmentSummaries.reduce((sum, inv) => sum + inv.co2Avoided, 0);
    const mwhGeneratedTotal = investmentSummaries.reduce((sum, inv) => sum + inv.mwhGenerated, 0);
    const homesEquivalent = Math.floor(mwhGeneratedTotal / 10); // ~10 MWh per home per year
    const treesEquivalent = Math.floor(co2AvoidedTotal / 0.039); // ~39 kg CO2 per tree per year

    // Update portfolio metrics
    await db
      .update(portfolios)
      .set({
        totalInvested: totalInvested.toString(),
        totalReturns: totalReturnsAmount.toString(),
        activeInvestments: investmentSummaries.filter(inv => inv.status === 'confirmed').length,
        totalCo2Avoided: co2AvoidedTotal.toString(),
        totalMwhGenerated: mwhGeneratedTotal.toString(),
        portfolioIrr: (totalReturn / Math.max(1, userInvestments.length)).toString(),
        updatedAt: new Date()
      })
      .where(eq(portfolios.id, portfolio.id));

    return NextResponse.json({
      success: true,
      portfolio: {
        metrics: {
          totalInvested,
          totalReturns: totalReturnsAmount,
          totalPledged: parseFloat(portfolio.totalPledged || '0'),
          activeInvestments: investmentSummaries.filter(inv => inv.status === 'confirmed').length,
          portfolioIrr: totalReturn,
          totalCo2Avoided: co2AvoidedTotal,
          totalMwhGenerated: mwhGeneratedTotal
        },
        investments: investmentSummaries,
        allocation: {
          byTechnology,
          byRegion,
          byStatus
        },
        performance: {
          totalReturn,
          annualizedReturn: totalReturn / Math.max(1, userInvestments.length), // Simplified
          bestPerformer,
          worstPerformer
        },
        impact: {
          co2AvoidedTotal,
          mwhGeneratedTotal,
          homesEquivalent,
          treesEquivalent
        }
      }
    });

  } catch (error) {
    logger.error('Error fetching portfolio:', error);
    return NextResponse.json({ 
      success: false, 
      message: 'Failed to fetch portfolio data' 
    }, { status: 500 });
  }
}
