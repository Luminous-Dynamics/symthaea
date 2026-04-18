// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Novel-position framework for Mycelix-specific tax questions.
 *
 * Four questions have no existing case law or published IRS / SARS /
 * HMRC guidance. Rather than pick a silent default (which would expose
 * the user to audit risk without their knowledge), this module surfaces
 * the question, enumerates the defensible stances, and embeds the
 * cite-able legal reasoning directly into the exported tax summary.
 *
 * The user picks a stance. The generator footnotes it in the output.
 * A tax professional reading the export can see both the number AND
 * the reasoning chain that produced it.
 *
 * This is the distinction between *silent compliance* (which looks like
 * evasion under audit) and *openly-positioned compliance* (which looks
 * like good-faith interpretation of an unsettled question).
 *
 * Part of the Mycelix × Nation-State coexistence extensions. See
 * MYCELIX_STATE_COEXISTENCE.md at the repo root.
 */

// ============================================================================
// Question 1: TEND as barter income
// ============================================================================

/**
 * TEND is a mutual-credit currency denominated in hours. When Alice
 * provides a service to Bob in exchange for TEND hours, is that event
 * (a) barter income at fair market value, (b) a zero-sum non-income
 * accounting entry, or (c) a gift?
 *
 * US: IRS Topic 420 treats barter as taxable at FMV. But the IRS has
 * never ruled on equal-hour time-bank exchanges where no market price
 * is set. Some time banks (TimeBanks USA) take the position that
 * equal-hour exchanges are not taxable barter because no dollar value
 * is negotiated.
 *
 * SA: Income Tax Act Section 1 "gross income" includes barter at FMV
 * (SARS Practice Note). Same ambiguity on equal-hour framing.
 */
export type TendBarterStance =
  /** Treat every TEND hour received as barter income at FMV. Conservative. */
  | 'TreatAsBarterFMV'
  /** Treat as mutual-credit zero-sum; no taxable event. Aggressive but defensible. */
  | 'TreatAsZeroSumNonIncome'
  /** Treat as gift (not income in most jurisdictions under gift threshold). */
  | 'TreatAsGift';

const TEND_BARTER_REASONING: Record<TendBarterStance, string> = {
  TreatAsBarterFMV:
    'Per IRS Topic 420 / SARS Practice Note on barter: services exchanged are taxable ' +
    'at fair market value. TEND hours are valued using the community labor_hour_value from ' +
    'community-config.json, sourced from a published wage reference.',
  TreatAsZeroSumNonIncome:
    'Per TimeBanks USA 1985 position letter and analogous equal-hour mutual-credit schemes: ' +
    'equal-hour exchanges with no negotiated dollar value are not barter for tax purposes ' +
    'because no bargained-for consideration exists. The mutual credit nets to zero across ' +
    'members and does not constitute "gross income" under IRC §61 or ZA Income Tax Act §1.',
  TreatAsGift:
    'Per IRC §102 and SA Estate Duty Act analog: gifts received are not income. TEND hours ' +
    'may be characterized as mutual gifts if no enforceable obligation binds the recipient. ' +
    'Caveat: annual gift exclusions may apply (USD 18,000 / 2026; ZAR 100,000 / SA).',
};

// ============================================================================
// Question 2: Demurrage treatment
// ============================================================================

/**
 * SAP carries a 2% annual demurrage that redistributes to commons
 * pools. Is the demurrage amount (a) a charitable contribution
 * deduction, (b) an involuntary property loss, (c) a constructive tax
 * imposed by the community, or (d) not a recognized tax event at all
 * (because no realization occurred)?
 */
export type DemurrageStance =
  | 'CharitableContribution'
  | 'InvoluntaryLoss'
  | 'ConstructiveTax'
  | 'NotTaxable';

const DEMURRAGE_REASONING: Record<DemurrageStance, string> = {
  CharitableContribution:
    'Per IRC §170 / SA ITA §18A: holder consents to demurrage by opting into SAP. ' +
    'Redistribution flows to commons pools that function as qualifying nonprofits. ' +
    'Requires the receiving commons pool to meet the 501(c)(3)/PBO equivalent test. ' +
    'Deduction claimed at FMV of SAP at the moment of redistribution.',
  InvoluntaryLoss:
    'Per IRC §165(c)(3) casualty-loss analog: demurrage is a programmatic reduction ' +
    'the holder cannot prevent once SAP is held. Characterized as ordinary loss at ' +
    'basis. Position may face pushback because the holder opted into SAP voluntarily.',
  ConstructiveTax:
    'Per treatise on "private currency" (e.g., Stark & Sullivan 2019): demurrage is a ' +
    'community-imposed levy on SAP holdings, functionally a wealth tax denominated in ' +
    'SAP. Reports as paid-tax line item, not a deduction. Characterization most ' +
    'robust if the DAO is formally recognized as a taxing authority.',
  NotTaxable:
    'Per IRC §1001 realization requirement: demurrage reduces SAP balance but no ' +
    'conversion to cash or other property occurs. No disposition, no gain-or-loss ' +
    'calculation. Ignored for tax purposes; basis carries forward in remaining SAP. ' +
    'May be challenged by auditor as constructive receipt by the commons pool.',
};

// ============================================================================
// Question 3: MYCEL accrual recognition
// ============================================================================

/**
 * MYCEL is a non-transferable reputation score. When it increases (via
 * recognition events, governance participation, etc.), is that (a) not
 * property and therefore not income, (b) compensation income at the
 * moment of accrual, or (c) deferred until realization (which for
 * non-transferable reputation may never occur)?
 */
export type MycelStance =
  | 'NonPropertyNotIncome'
  | 'CompensationAtAccrual'
  | 'DeferredUntilRealized';

const MYCEL_REASONING: Record<MycelStance, string> = {
  NonPropertyNotIncome:
    'Per Commissioner v. Glenshaw Glass 348 U.S. 426 (1955): income is "undeniable ' +
    'accession to wealth, clearly realized, and over which the taxpayers have ' +
    'complete dominion." MYCEL is non-transferable and not convertible to property; ' +
    'therefore fails the "complete dominion" prong. Not income, no reporting required.',
  CompensationAtAccrual:
    'Per IRC §83 property-rights test: if MYCEL confers a substantial right (e.g., ' +
    'governance vote weight that translates to economic benefit), accrual is income at ' +
    'the FMV of that right. Practically unmeasurable without a liquid market for ' +
    'governance-weighted positions; conservative filer may report at zero FMV citing ' +
    'the lack of a willing buyer.',
  DeferredUntilRealized:
    'Per Rev. Rul. 83-46 and similar deferred-property rulings: non-transferable ' +
    'reputation is unrealized until converted to transferable form. Since MYCEL ' +
    'cannot be transferred by design, realization is permanently deferred. ' +
    'Equivalent outcome to NonPropertyNotIncome, but frames the absence as ' +
    'deferral rather than non-recognition.',
};

// ============================================================================
// Question 4: Compost redistribution as constructive receipt
// ============================================================================

/**
 * When SAP demurrage flows to a commons pool the user has a stake in,
 * has the user constructively received that redistribution? The
 * redistribution is automatic and governed by pre-agreed rules.
 */
export type CompostStance =
  /** User disclaims via non-control over the commons pool. No receipt. */
  | 'DisclaimedViaNonControl'
  /** User realizes pro-rata at redistribution time; basis adjusts. */
  | 'RealizedOnRedistribution';

const COMPOST_REASONING: Record<CompostStance, string> = {
  DisclaimedViaNonControl:
    'Per Helvering v. Horst 311 U.S. 112 (1940) and the constructive-receipt doctrine: ' +
    'income is received when the taxpayer has an unrestricted right to demand it. The ' +
    'commons pool governs redistribution collectively (DAO vote / algorithmic rule); ' +
    'the individual user cannot unilaterally draw funds. Not constructively received.',
  RealizedOnRedistribution:
    'Per IRC §671 grantor-trust analog: if the commons pool treats pro-rata stakeholders ' +
    'as beneficial owners, redistribution flows through to the user at FMV as of the ' +
    'redistribution date. User increases basis in the commons pool share by that amount. ' +
    'Conservative position; required if stakeholders control pool spending.',
};

// ============================================================================
// User's chosen positions
// ============================================================================

/**
 * Four stances, one chosen per question. The tax-export generator
 * embeds the corresponding reasoning string into its output footnotes.
 */
export interface NovelPositionSet {
  tendBarter: TendBarterStance;
  demurrage: DemurrageStance;
  mycel: MycelStance;
  compost: CompostStance;
}

/**
 * A conservative default position set — picks the stance most likely
 * to survive an audit, even at the cost of reporting more income.
 *
 * The system does NOT ship this as a silent default. Callers MUST
 * surface all four questions to the user and get an explicit choice.
 * This function exists only so tests and demos can construct a
 * consistent position set.
 */
export function conservativeDefaults(): NovelPositionSet {
  return {
    tendBarter: 'TreatAsBarterFMV',
    demurrage: 'NotTaxable',
    mycel: 'NonPropertyNotIncome',
    compost: 'DisclaimedViaNonControl',
  };
}

/**
 * Human-readable question labels for the UI.
 */
export const QUESTION_LABELS = {
  tendBarter: 'How should TEND hour exchanges be treated?',
  demurrage: 'How should SAP demurrage (the 2% circulation fee) be treated?',
  mycel: 'How should MYCEL reputation score increases be treated?',
  compost: 'How should commons-pool redistributions to your stake be treated?',
} as const;

// ============================================================================
// Reasoning extraction for export footnotes
// ============================================================================

/**
 * A single footnote entry ready for embedding in a tax export summary.
 * The generator appends these to the CSV/JSON footer so a tax
 * professional or auditor can see the user's position chain.
 */
export interface PositionFootnote {
  question: keyof NovelPositionSet;
  questionLabel: string;
  chosenStance: string;
  reasoning: string;
}

/**
 * Produce the four footnotes for a chosen position set. Ordered by
 * question for consistent presentation across exports.
 */
export function footnotes(positions: NovelPositionSet): PositionFootnote[] {
  return [
    {
      question: 'tendBarter',
      questionLabel: QUESTION_LABELS.tendBarter,
      chosenStance: positions.tendBarter,
      reasoning: TEND_BARTER_REASONING[positions.tendBarter],
    },
    {
      question: 'demurrage',
      questionLabel: QUESTION_LABELS.demurrage,
      chosenStance: positions.demurrage,
      reasoning: DEMURRAGE_REASONING[positions.demurrage],
    },
    {
      question: 'mycel',
      questionLabel: QUESTION_LABELS.mycel,
      chosenStance: positions.mycel,
      reasoning: MYCEL_REASONING[positions.mycel],
    },
    {
      question: 'compost',
      questionLabel: QUESTION_LABELS.compost,
      chosenStance: positions.compost,
      reasoning: COMPOST_REASONING[positions.compost],
    },
  ];
}

/**
 * Render footnotes as a plain-text block suitable for appending to
 * the CSV header or a printed tax return attachment.
 */
export function renderFootnotesText(positions: NovelPositionSet): string {
  const lines: string[] = [];
  lines.push('# ────────────────────────────────────────────────────────────');
  lines.push('# NOVEL TAX POSITIONS — Mycelix primitives have no case law');
  lines.push('# The four questions below have no published IRS / SARS / HMRC');
  lines.push('# guidance. The filer has chosen the following stances in good');
  lines.push('# faith; an auditor may reach different conclusions.');
  lines.push('# ────────────────────────────────────────────────────────────');
  for (const [idx, fn] of footnotes(positions).entries()) {
    lines.push(`#`);
    lines.push(`# Q${idx + 1}: ${fn.questionLabel}`);
    lines.push(`# Position: ${fn.chosenStance}`);
    // Word-wrap reasoning to ~72 cols for CSV compatibility.
    const words = fn.reasoning.split(/\s+/);
    let line = '#   ';
    for (const word of words) {
      if (line.length + word.length + 1 > 76) {
        lines.push(line);
        line = '#   ';
      }
      line += word + ' ';
    }
    if (line.trim() !== '#') lines.push(line.trimEnd());
  }
  lines.push('# ────────────────────────────────────────────────────────────');
  return lines.join('\n');
}

/**
 * Whether the user's chosen TEND-barter stance implies TEND should
 * contribute to the income total in the tax export. Used by the
 * generator to decide whether to include BarterLabor hours.
 */
export function tendContributesToIncome(positions: NovelPositionSet): boolean {
  return positions.tendBarter === 'TreatAsBarterFMV';
}
