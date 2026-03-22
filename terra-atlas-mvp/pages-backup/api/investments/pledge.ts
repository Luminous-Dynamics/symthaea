// Terra Atlas Investment API - Create Pledge
// TypeScript implementation for investment pledges

import type { NextApiRequest, NextApiResponse } from 'next';
import { db } from '../../../lib/drizzle/db';
import { investments, portfolios } from '../../../lib/drizzle/schema-investment';
import { energyProjects } from '../../../lib/drizzle/schema-energy';
import { eq } from 'drizzle-orm';
import jwt from 'jsonwebtoken';

interface PledgeRequest {
  projectId: string;
  amountUsd: number;
  investmentType: 'equity' | 'debt' | 'revenue_share' | 'crowdfunding' | 'green_bond' | 'ppa';
  expectedReturn?: number;
  investmentPeriodYears?: number;
  notes?: string;
}

interface PledgeResponse {
  success: boolean;
  pledgeId?: string;
  message: string;
  pledge?: {
    id: string;
    projectName: string;
    amountUsd: number;
    expectedReturn: number;
    status: string;
  };
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<PledgeResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ 
      success: false, 
      message: 'Method not allowed' 
    });
  }

  try {
    // Get user from JWT token
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({ 
        success: false, 
        message: 'Authentication required' 
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'dev-secret') as { userId: string };
    const userId = decoded.userId;

    const { 
      projectId, 
      amountUsd, 
      investmentType, 
      expectedReturn,
      investmentPeriodYears,
      notes 
    } = req.body as PledgeRequest;

    // Validate input
    if (!projectId || !amountUsd || !investmentType) {
      return res.status(400).json({ 
        success: false, 
        message: 'Missing required fields' 
      });
    }

    if (amountUsd < 10) {
      return res.status(400).json({ 
        success: false, 
        message: 'Minimum investment is $10' 
      });
    }

    // Check if project exists
    const [project] = await db
      .select()
      .from(energyProjects)
      .where(eq(energyProjects.id, projectId))
      .limit(1);

    if (!project) {
      return res.status(404).json({ 
        success: false, 
        message: 'Project not found' 
      });
    }

    // Get or create user portfolio
    let [portfolio] = await db
      .select()
      .from(portfolios)
      .where(eq(portfolios.userId, userId))
      .limit(1);

    if (!portfolio) {
      // Create portfolio for new investor
      const [newPortfolio] = await db
        .insert(portfolios)
        .values({
          userId,
          totalPledged: amountUsd.toString(),
          activeInvestments: 0,
          preferredTechnologies: [project.projectType],
          createdAt: new Date(),
          updatedAt: new Date()
        })
        .returning();
      
      portfolio = newPortfolio;
    } else {
      // Update portfolio pledged amount
      await db
        .update(portfolios)
        .set({
          totalPledged: (parseFloat(portfolio.totalPledged || '0') + amountUsd).toString(),
          updatedAt: new Date()
        })
        .where(eq(portfolios.id, portfolio.id));
    }

    // Calculate expected return if not provided
    const calculatedReturn = expectedReturn || parseFloat(project.expectedIrr || '10');

    // Calculate impact metrics
    const projectCapacityMw = parseFloat(project.capacityMw || '0');
    const projectCostMillion = parseFloat(project.totalCostMillion || '100');
    const investmentShare = amountUsd / (projectCostMillion * 1000000);
    
    const co2AvoidedTons = (parseFloat(project.co2AvoidedTonsYear || '0') * investmentShare).toFixed(2);
    const mwhGenerated = (parseFloat(project.annualGenerationGwh || '0') * 1000 * investmentShare).toFixed(2);
    const homesEquivalent = Math.floor(parseFloat(mwhGenerated) / 10); // ~10 MWh per home per year

    // Create investment pledge
    const [pledge] = await db
      .insert(investments)
      .values({
        userId,
        portfolioId: portfolio.id,
        projectId,
        investmentType,
        amountUsd: amountUsd.toString(),
        status: 'pledged',
        expectedReturn: calculatedReturn.toString(),
        investmentPeriodYears: (investmentPeriodYears || 10).toString(),
        minimumHoldYears: '2',
        pledgeDate: new Date(),
        co2AvoidedTons,
        mwhGenerated,
        homesEquivalent,
        notes,
        metadata: {
          projectName: project.name,
          projectType: project.projectType,
          projectLocation: `${project.state}, ${project.country}`,
          pledgedVia: 'Terra Atlas Web'
        },
        createdAt: new Date(),
        updatedAt: new Date()
      })
      .returning();

    res.status(200).json({
      success: true,
      pledgeId: pledge.id,
      message: `Successfully pledged $${amountUsd.toLocaleString()} to ${project.name}`,
      pledge: {
        id: pledge.id,
        projectName: project.name,
        amountUsd,
        expectedReturn: calculatedReturn,
        status: 'pledged'
      }
    });

  } catch (error) {
    console.error('Error creating pledge:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to create investment pledge' 
    });
  }
}