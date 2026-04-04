// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS Business Studies Grade 10 — Entrepreneurship, Business Environments, Ethics.

use super::TopicContent;

pub(crate) struct Gr10Entrepreneurship;
impl TopicContent for Gr10Entrepreneurship {
    fn explanation(&self) -> String {
        "An entrepreneur identifies opportunities and takes risks to start a business. \
         South Africa needs entrepreneurs to create jobs and grow the economy. \
         A business plan includes: executive summary, market analysis, financial projections, \
         marketing strategy, and operations plan. \
         Forms of ownership: sole trader (1 person, unlimited liability), partnership (2-20), \
         close corporation (1-10, separate legal entity), company (Pty Ltd, limited liability).".to_string()
    }
    fn worked_example(&self, i: usize) -> String {
        match i {
            0 => "Thandi wants to start a catering business. What form of ownership?\nConsider: risk, capital needed, skills.\nSole trader: simple, but unlimited liability (personal assets at risk).\nClose corporation: separate legal entity, limited liability, more paperwork.\nRecommend: Start as sole trader, register as CC when growing.".to_string(),
            1 => "Calculate break-even: Fixed costs R5,000/month. Selling price R50/meal. Variable cost R30/meal.\nContribution per meal: R50 - R30 = R20.\nBreak-even: R5,000 ÷ R20 = 250 meals/month.\nThandi must sell at least 250 meals to cover costs.".to_string(),
            _ => "SWOT analysis for a spaza shop:\nStrengths: Close to customers, low overhead, personal service.\nWeaknesses: Limited stock variety, no economies of scale.\nOpportunities: Mobile payment (SnapScan), delivery service, community loyalty.\nThreats: Supermarket chains, crime, load shedding.".to_string(),
        }
    }
    fn practice_problem(&self, d: u16) -> String {
        match d {
            0..=300 => "Name 3 qualities of a successful entrepreneur.\nExamples: Risk-taking, creativity, perseverance, leadership, problem-solving, self-discipline.\nOnly money, Only education\nThink about what it takes to START and KEEP a business.\nWhat if customers don't come? What if something goes wrong?".to_string(),
            301..=600 => "Fixed costs: R8,000. Price: R100. Variable cost: R60. Break-even?\n200 units\nContribution = 100-60 = R40. Break-even = 8000/40 = 200.\n80, 133, 800\nContribution = price - variable cost.\n8000 ÷ 40 = ?".to_string(),
            _ => "Explain why a Pty Ltd company has advantages over a sole trader for a growing business.\n1. Limited liability (personal assets protected). 2. Easier to raise capital (sell shares). 3. Separate legal entity (continues if owner leaves). 4. More credible to banks/investors.\nNo difference, Sole trader is better\nWhat happens if the business owes money and fails?\nSole trader: YOU pay all debts. Pty Ltd: company pays, personal assets safe.".to_string(),
        }
    }
    fn hint(&self, l: u8) -> String { match l { 1 => "Think about what a business NEEDS to succeed.".to_string(), 2 => "Break-even = fixed costs ÷ contribution per unit.".to_string(), _ => "SWOT: Strengths/Weaknesses (internal), Opportunities/Threats (external).".to_string() } }
    fn misconception(&self) -> String { "WRONG: You need lots of money to start a business.\nRIGHT: Many successful SA businesses started with very little: a braai stand, a car wash, tutoring. Start small, reinvest profits.\nWHY: The key resource is the entrepreneur's drive and idea, not money.".to_string() }
    fn vocabulary(&self) -> String { "entrepreneur: Person who starts a business, taking financial risk | Elon Musk (SA-born), Patrice Motsepe.\nbreak-even: Point where revenue = costs (no profit, no loss) | Sales needed to cover all expenses.\nSWOT: Analysis of Strengths, Weaknesses, Opportunities, Threats.".to_string() }
    fn flashcard(&self) -> String { "What is limited liability? | Owners' personal assets are protected if the business fails — only lose what they invested".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Create a mini business plan for a student tutoring service. Include: target market, pricing, costs, break-even.\nTarget: Gr8-12 students. Price: R150/hr. Costs: transport R500/month, wifi R200. Contribution: R150. Break-even: R700/R150 ≈ 5 sessions.\n4\nThink practically: who pays, how much, what do you spend?".to_string() }
}
