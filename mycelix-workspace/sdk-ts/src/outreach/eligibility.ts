/**
 * Predictive Eligibility for Proactive Outreach
 *
 * Predicts potential eligibility for benefits and services
 * based on citizen profile and life events.
 *
 * Philosophy: Lower barriers, not raise them. When in doubt, encourage application.
 *
 * @module outreach/eligibility
 */

import type { LifeEvent, LifeEventType } from './life-events.js';

/**
 * Benefit program types for eligibility prediction
 */
export type BenefitProgram =
  | 'snap'
  | 'tanf'
  | 'medicaid'
  | 'medicare'
  | 'housing'
  | 'liheap'
  | 'childcare'
  | 'wic'
  | 'unemployment'
  | 'disability'
  | 'social_security'
  | 'veterans'
  | 'job_training'
  | 'transportation'
  | 'legal_aid';

/**
 * Eligibility prediction result
 */
export interface EligibilityPrediction {
  /** Program being evaluated */
  program: BenefitProgram;
  /** Prediction: likely, possible, unlikely */
  prediction: 'likely' | 'possible' | 'unlikely';
  /** Confidence in prediction (0-1) */
  confidence: number;
  /** Factors that suggest eligibility */
  positiveFactors: string[];
  /** Factors that may affect eligibility */
  cautionFactors: string[];
  /** Missing information that would improve prediction */
  missingInfo: string[];
  /** Suggested next steps */
  nextSteps: string[];
  /** Related life events that triggered this prediction */
  relatedEvents?: LifeEventType[];
}

/**
 * Eligibility criteria for a program
 */
export interface EligibilityCriteria {
  /** Program name */
  program: BenefitProgram;
  /** Human-readable name */
  displayName: string;
  /** Brief description */
  description: string;
  /** Income threshold as percent of Federal Poverty Level */
  incomeFPLPercent?: number;
  /** Asset limit in dollars */
  assetLimit?: number;
  /** Age requirements */
  ageRequirement?: { min?: number; max?: number };
  /** Household composition requirements */
  householdRequirements?: string[];
  /** Citizenship/immigration requirements */
  citizenshipRequired?: boolean;
  /** Other categorical requirements */
  categoricalRequirements?: string[];
  /** Life events that commonly trigger eligibility */
  triggerEvents: LifeEventType[];
}

/**
 * Federal Poverty Level guidelines (2024)
 */
const FPL_2024: Record<number, number> = {
  1: 15060,
  2: 20440,
  3: 25820,
  4: 31200,
  5: 36580,
  6: 41960,
  7: 47340,
  8: 52720,
};

/**
 * Get FPL for household size
 */
function getFPL(householdSize: number): number {
  if (householdSize <= 8) return FPL_2024[householdSize];
  // Each additional person adds $5380
  return FPL_2024[8] + (householdSize - 8) * 5380;
}

/**
 * Program eligibility criteria
 */
export const PROGRAM_CRITERIA: Record<BenefitProgram, EligibilityCriteria> = {
  snap: {
    program: 'snap',
    displayName: 'SNAP (Food Assistance)',
    description: 'Monthly benefits for purchasing food',
    incomeFPLPercent: 130,
    assetLimit: 2750, // $3500 for elderly/disabled
    householdRequirements: ['Prepare meals together'],
    citizenshipRequired: true,
    triggerEvents: ['income_change', 'job_loss', 'birth', 'divorce'],
  },
  tanf: {
    program: 'tanf',
    displayName: 'TANF (Cash Assistance)',
    description: 'Temporary cash assistance for families with children',
    incomeFPLPercent: 100,
    householdRequirements: ['Children under 18 in household'],
    citizenshipRequired: true,
    triggerEvents: ['job_loss', 'divorce', 'birth', 'housing_instability'],
  },
  medicaid: {
    program: 'medicaid',
    displayName: 'Medicaid',
    description: 'Free or low-cost health coverage',
    incomeFPLPercent: 138, // ACA expansion states
    categoricalRequirements: ['Children, pregnant, elderly, or disabled OR income-eligible'],
    citizenshipRequired: true,
    triggerEvents: ['income_change', 'job_loss', 'birth', 'disability_onset', 'health_crisis'],
  },
  medicare: {
    program: 'medicare',
    displayName: 'Medicare',
    description: 'Health insurance for 65+ or disabled',
    ageRequirement: { min: 65 },
    categoricalRequirements: ['Age 65+ OR disability for 24+ months'],
    triggerEvents: ['age_milestone', 'disability_onset'],
  },
  housing: {
    program: 'housing',
    displayName: 'Housing Assistance',
    description: 'Help with rent or finding affordable housing',
    incomeFPLPercent: 200, // Varies by program
    citizenshipRequired: true,
    triggerEvents: ['income_change', 'job_loss', 'housing_instability', 'natural_disaster'],
  },
  liheap: {
    program: 'liheap',
    displayName: 'LIHEAP (Energy Assistance)',
    description: 'Help paying heating and cooling bills',
    incomeFPLPercent: 150,
    citizenshipRequired: true,
    triggerEvents: ['income_change', 'job_loss'],
  },
  childcare: {
    program: 'childcare',
    displayName: 'Child Care Assistance',
    description: 'Subsidized child care while working or in school',
    incomeFPLPercent: 200, // Varies by state
    householdRequirements: ['Children under 13', 'Working or in training'],
    citizenshipRequired: true,
    triggerEvents: ['birth', 'job_gain', 'income_change'],
  },
  wic: {
    program: 'wic',
    displayName: 'WIC',
    description: 'Nutrition for pregnant women, infants, and children',
    incomeFPLPercent: 185,
    householdRequirements: ['Pregnant, postpartum, or children under 5'],
    citizenshipRequired: false, // Not required for WIC
    triggerEvents: ['birth'],
  },
  unemployment: {
    program: 'unemployment',
    displayName: 'Unemployment Insurance',
    description: 'Weekly benefits while searching for work',
    categoricalRequirements: ['Lost job through no fault', 'Able and available to work'],
    triggerEvents: ['job_loss'],
  },
  disability: {
    program: 'disability',
    displayName: 'Disability Benefits (SSI/SSDI)',
    description: 'Monthly income for disabled individuals',
    categoricalRequirements: ['Unable to work due to disability', 'Expected to last 12+ months'],
    triggerEvents: ['disability_onset', 'health_crisis'],
  },
  social_security: {
    program: 'social_security',
    displayName: 'Social Security Retirement',
    description: 'Retirement benefits based on work history',
    ageRequirement: { min: 62 },
    categoricalRequirements: ['Earned sufficient work credits'],
    triggerEvents: ['age_milestone'],
  },
  veterans: {
    program: 'veterans',
    displayName: 'Veterans Benefits',
    description: 'Healthcare, disability, education, housing for veterans',
    categoricalRequirements: ['Served in US military', 'Discharge status requirements'],
    triggerEvents: ['veteran_discharge'],
  },
  job_training: {
    program: 'job_training',
    displayName: 'Job Training Programs',
    description: 'Skills training and job placement assistance',
    categoricalRequirements: ['Unemployed or underemployed'],
    triggerEvents: ['job_loss', 'incarceration_release', 'education_completion'],
  },
  transportation: {
    program: 'transportation',
    displayName: 'Transportation Assistance',
    description: 'Help with transportation for work or medical care',
    incomeFPLPercent: 200,
    triggerEvents: ['job_gain', 'address_change'],
  },
  legal_aid: {
    program: 'legal_aid',
    displayName: 'Legal Aid',
    description: 'Free legal assistance for civil matters',
    incomeFPLPercent: 125,
    triggerEvents: ['divorce', 'housing_instability', 'incarceration_release'],
  },
};

/**
 * Citizen profile for eligibility prediction
 */
export interface CitizenEligibilityProfile {
  /** Household size */
  householdSize: number;
  /** Monthly gross income */
  monthlyIncome?: number;
  /** Annual gross income */
  annualIncome?: number;
  /** Total countable assets */
  assets?: number;
  /** Age of primary applicant */
  age?: number;
  /** US citizen or qualified immigrant */
  isCitizen?: boolean;
  /** Has children in household */
  hasChildren?: boolean;
  /** Ages of children */
  childrenAges?: number[];
  /** Is pregnant */
  isPregnant?: boolean;
  /** Has disability */
  hasDisability?: boolean;
  /** Is veteran */
  isVeteran?: boolean;
  /** Currently employed */
  isEmployed?: boolean;
  /** Current zip code */
  zipCode?: string;
  /** Recent life events */
  recentEvents?: LifeEvent[];
}

/**
 * Eligibility Predictor
 *
 * Predicts potential eligibility for benefits based on citizen profile.
 * Designed to encourage application, not discourage it.
 */
export class EligibilityPredictor {
  /**
   * Predict eligibility for all programs
   */
  predictAll(profile: CitizenEligibilityProfile): EligibilityPrediction[] {
    const predictions: EligibilityPrediction[] = [];

    for (const program of Object.keys(PROGRAM_CRITERIA) as BenefitProgram[]) {
      const prediction = this.predictProgram(profile, program);
      predictions.push(prediction);
    }

    // Sort by likelihood (likely first) then confidence
    return predictions.sort((a, b) => {
      const order = { likely: 0, possible: 1, unlikely: 2 };
      const orderDiff = order[a.prediction] - order[b.prediction];
      if (orderDiff !== 0) return orderDiff;
      return b.confidence - a.confidence;
    });
  }

  /**
   * Predict eligibility for specific program
   */
  predictProgram(
    profile: CitizenEligibilityProfile,
    program: BenefitProgram
  ): EligibilityPrediction {
    const criteria = PROGRAM_CRITERIA[program];
    const positiveFactors: string[] = [];
    const cautionFactors: string[] = [];
    const missingInfo: string[] = [];
    const relatedEvents: LifeEventType[] = [];

    // Check life events
    if (profile.recentEvents) {
      for (const event of profile.recentEvents) {
        if (criteria.triggerEvents.includes(event.type)) {
          positiveFactors.push(`Recent ${event.type.replace(/_/g, ' ')} event`);
          relatedEvents.push(event.type);
        }
      }
    }

    // Check income
    if (criteria.incomeFPLPercent !== undefined) {
      const income = profile.annualIncome ?? (profile.monthlyIncome ?? 0) * 12;
      const fpl = getFPL(profile.householdSize);
      const threshold = fpl * (criteria.incomeFPLPercent / 100);

      if (income === 0 && profile.monthlyIncome === undefined) {
        missingInfo.push('Income information');
      } else if (income <= threshold) {
        positiveFactors.push(`Income below ${criteria.incomeFPLPercent}% FPL`);
      } else if (income <= threshold * 1.1) {
        cautionFactors.push('Income near threshold - deductions may help');
      } else {
        cautionFactors.push(`Income may exceed ${criteria.incomeFPLPercent}% FPL`);
      }
    }

    // Check assets
    if (criteria.assetLimit !== undefined) {
      if (profile.assets === undefined) {
        missingInfo.push('Asset information');
      } else if (profile.assets <= criteria.assetLimit) {
        positiveFactors.push('Assets within limit');
      } else {
        cautionFactors.push('Assets may exceed limit');
      }
    }

    // Check age requirements
    if (criteria.ageRequirement) {
      if (profile.age === undefined) {
        missingInfo.push('Age');
      } else {
        if (criteria.ageRequirement.min && profile.age >= criteria.ageRequirement.min) {
          positiveFactors.push(`Age ${profile.age} meets minimum`);
        } else if (criteria.ageRequirement.min && profile.age < criteria.ageRequirement.min) {
          cautionFactors.push(`Age ${profile.age} below minimum ${criteria.ageRequirement.min}`);
        }
        if (criteria.ageRequirement.max && profile.age <= criteria.ageRequirement.max) {
          positiveFactors.push(`Age ${profile.age} within range`);
        }
      }
    }

    // Check citizenship
    if (criteria.citizenshipRequired) {
      if (profile.isCitizen === undefined) {
        missingInfo.push('Citizenship/immigration status');
      } else if (profile.isCitizen) {
        positiveFactors.push('Citizenship requirement met');
      } else {
        cautionFactors.push('May need to verify immigration status');
      }
    }

    // Check household requirements
    if (criteria.householdRequirements) {
      for (const req of criteria.householdRequirements) {
        if (req.includes('Children') && profile.hasChildren === undefined) {
          missingInfo.push('Household composition');
        } else if (req.includes('Children') && profile.hasChildren) {
          positiveFactors.push('Has children in household');
        }
      }
    }

    // Check categorical requirements
    if (criteria.categoricalRequirements) {
      if (criteria.program === 'disability' && profile.hasDisability) {
        positiveFactors.push('Has qualifying disability');
      }
      if (criteria.program === 'veterans' && profile.isVeteran) {
        positiveFactors.push('Veteran status verified');
      }
      if (criteria.program === 'unemployment' && !profile.isEmployed) {
        positiveFactors.push('Currently unemployed');
      }
    }

    // WIC-specific checks
    if (criteria.program === 'wic') {
      if (profile.isPregnant) {
        positiveFactors.push('Currently pregnant');
      }
      if (profile.childrenAges?.some((age) => age < 5)) {
        positiveFactors.push('Has children under 5');
      }
    }

    // Calculate prediction
    const { prediction, confidence } = this.calculatePrediction(
      positiveFactors.length,
      cautionFactors.length,
      missingInfo.length,
      relatedEvents.length
    );

    // Generate next steps
    const nextSteps = this.generateNextSteps(program, prediction, missingInfo);

    return {
      program,
      prediction,
      confidence,
      positiveFactors,
      cautionFactors,
      missingInfo,
      nextSteps,
      relatedEvents: relatedEvents.length > 0 ? relatedEvents : undefined,
    };
  }

  /**
   * Get programs triggered by a life event
   */
  getProgramsForEvent(eventType: LifeEventType): BenefitProgram[] {
    return (Object.keys(PROGRAM_CRITERIA) as BenefitProgram[]).filter((program) =>
      PROGRAM_CRITERIA[program].triggerEvents.includes(eventType)
    );
  }

  /**
   * Get quick eligibility summary for SMS
   */
  getQuickSummary(predictions: EligibilityPrediction[]): string {
    const likely = predictions.filter((p) => p.prediction === 'likely');
    const possible = predictions.filter((p) => p.prediction === 'possible');

    let summary = '';

    if (likely.length > 0) {
      summary += 'You MAY qualify for:\n';
      likely.slice(0, 3).forEach((p) => {
        summary += `- ${PROGRAM_CRITERIA[p.program].displayName}\n`;
      });
    }

    if (possible.length > 0 && likely.length < 3) {
      summary += '\nWorth checking:\n';
      possible.slice(0, 2).forEach((p) => {
        summary += `- ${PROGRAM_CRITERIA[p.program].displayName}\n`;
      });
    }

    summary += '\nThis is a screening only. Apply for official determination.';

    return summary;
  }

  // Private methods

  private calculatePrediction(
    positiveCount: number,
    cautionCount: number,
    missingCount: number,
    eventCount: number
  ): { prediction: 'likely' | 'possible' | 'unlikely'; confidence: number } {
    // Bias toward "possible" when uncertain (benefit of doubt)
    const netScore = positiveCount + eventCount * 0.5 - cautionCount * 0.5;

    let prediction: 'likely' | 'possible' | 'unlikely';
    let confidence: number;

    if (netScore >= 2) {
      prediction = 'likely';
      confidence = Math.min(0.95, 0.6 + netScore * 0.1);
    } else if (netScore >= 0 || missingCount > 0) {
      prediction = 'possible';
      confidence = 0.5 + Math.min(0.3, netScore * 0.1);
    } else {
      prediction = 'unlikely';
      confidence = Math.max(0.3, 0.6 + netScore * 0.1);
    }

    // Reduce confidence when missing info
    if (missingCount > 0) {
      confidence = Math.max(0.3, confidence - missingCount * 0.1);
    }

    return { prediction, confidence };
  }

  private generateNextSteps(
    program: BenefitProgram,
    prediction: 'likely' | 'possible' | 'unlikely',
    missingInfo: string[]
  ): string[] {
    const steps: string[] = [];

    if (missingInfo.length > 0) {
      steps.push(`Gather: ${missingInfo.slice(0, 2).join(', ')}`);
    }

    if (prediction !== 'unlikely') {
      steps.push(`Apply for ${PROGRAM_CRITERIA[program].displayName}`);
    }

    if (prediction === 'likely') {
      steps.push('Gather documents: ID, income proof, residency');
    } else if (prediction === 'possible') {
      steps.push('Contact local office to discuss your situation');
    } else {
      steps.push('Consider contacting 211 for other resources');
    }

    return steps;
  }
}

/**
 * Create an eligibility predictor instance
 */
export function createEligibilityPredictor(): EligibilityPredictor {
  return new EligibilityPredictor();
}
