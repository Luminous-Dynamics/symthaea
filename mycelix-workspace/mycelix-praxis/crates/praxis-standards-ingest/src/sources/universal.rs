// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Universal subjects — essential knowledge everyone should have.
//! Financial literacy, critical thinking, health, civics, systems thinking,
//! emotional intelligence, communication, sustainability, and learning skills.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::*;

pub struct UniversalSource;
impl UniversalSource { pub fn new() -> Self { Self } }

impl CurriculumSource for UniversalSource {
    fn name(&self) -> &str { "Universal Subjects (Everyone Should Know)" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            SourceEntry { id: "financial-literacy".into(), title: "Financial Literacy".into(), subject: "Financial Literacy".into(), level: "Adult".into(), description: "Earning, saving, investing, borrowing, protecting, planning (18 nodes)".into() },
            SourceEntry { id: "critical-thinking".into(), title: "Critical Thinking & Logic".into(), subject: "Critical Thinking".into(), level: "Adult".into(), description: "Fallacies, evidence, reasoning, media literacy (12 nodes)".into() },
            SourceEntry { id: "digital-literacy".into(), title: "Digital Literacy".into(), subject: "Digital Literacy".into(), level: "Adult".into(), description: "Information, safety, creation, citizenship (12 nodes)".into() },
            SourceEntry { id: "health-literacy".into(), title: "Health Literacy".into(), subject: "Health".into(), level: "Adult".into(), description: "Body, nutrition, mental health, first aid (10 nodes)".into() },
            SourceEntry { id: "civic-literacy".into(), title: "Civic Literacy".into(), subject: "Civic Literacy".into(), level: "Adult".into(), description: "Government, rights, voting, participation (10 nodes)".into() },
            SourceEntry { id: "systems-thinking".into(), title: "Systems Thinking".into(), subject: "Systems Thinking".into(), level: "Adult".into(), description: "Feedback loops, emergence, leverage points (8 nodes)".into() },
            SourceEntry { id: "emotional-intelligence".into(), title: "Emotional Intelligence".into(), subject: "Emotional Intelligence".into(), level: "Adult".into(), description: "Self-awareness, empathy, conflict resolution (10 nodes)".into() },
            SourceEntry { id: "communication".into(), title: "Communication Skills".into(), subject: "Communication".into(), level: "Adult".into(), description: "Writing, speaking, listening, presenting (8 nodes)".into() },
            SourceEntry { id: "sustainability".into(), title: "Sustainability".into(), subject: "Sustainability".into(), level: "Adult".into(), description: "Climate, resources, circular economy (8 nodes)".into() },
            SourceEntry { id: "learning-skills".into(), title: "Learning How to Learn".into(), subject: "Learning Science".into(), level: "Adult".into(), description: "Metacognition, spaced repetition, deliberate practice (8 nodes)".into() },
            SourceEntry { id: "economics-basics".into(), title: "Economics Basics".into(), subject: "Economics".into(), level: "Adult".into(), description: "Supply/demand, incentives, markets, policy (8 nodes)".into() },
            SourceEntry { id: "statistics-data".into(), title: "Statistics & Data Literacy".into(), subject: "Statistics".into(), level: "Adult".into(), description: "Probability, distributions, bias, visualization (10 nodes)".into() },
        ])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let (title, subject, nodes_data): (&str, &str, Vec<(&str, &str, &str, Vec<&str>)>) = match id {
            "financial-literacy" => ("Financial Literacy", "Financial Literacy", vec![
                ("FIN.EARN", "Earning & Income", "Career earnings potential, salary negotiation, taxes (income tax, VAT), benefits, side income.", vec!["earning", "taxes", "income"]),
                ("FIN.BUDGET", "Budgeting", "50/30/20 rule, tracking expenses, needs vs wants, emergency fund, opportunity cost.", vec!["budgeting", "expenses", "emergency-fund"]),
                ("FIN.SAVE", "Saving & Compound Interest", "Power of compound interest, savings accounts, tax-free savings (SA), Rule of 72.", vec!["saving", "compound-interest", "rule-of-72"]),
                ("FIN.INVEST", "Investing Basics", "Stocks, bonds, ETFs, index funds, diversification, risk vs return, time in market.", vec!["investing", "stocks", "bonds", "etf", "diversification"]),
                ("FIN.DEBT", "Borrowing & Credit", "Credit scores, interest rates, good vs bad debt, student loans, credit cards, mortgage basics.", vec!["credit", "debt", "loans", "interest-rates"]),
                ("FIN.PROTECT", "Insurance & Protection", "Health, life, car, home insurance. Identity theft protection. Scam recognition.", vec!["insurance", "identity-protection", "scams"]),
                ("FIN.RETIRE", "Retirement Planning", "Pension funds, provident funds (SA), retirement annuities, social security, starting early.", vec!["retirement", "pension", "annuity"]),
                ("FIN.PLAN", "Financial Planning", "Goal setting, life event planning, estate basics, wills, financial advisor role.", vec!["planning", "goals", "estate", "wills"]),
            ]),
            "critical-thinking" => ("Critical Thinking & Logic", "Critical Thinking", vec![
                ("CT.ARGS", "Argument Structure", "Premises, conclusions, validity, soundness. Identifying hidden assumptions.", vec!["arguments", "premises", "conclusions", "validity"]),
                ("CT.FALL", "Logical Fallacies", "Ad hominem, straw man, false dichotomy, appeal to authority, slippery slope, circular reasoning, red herring.", vec!["fallacies", "ad-hominem", "straw-man"]),
                ("CT.EVID", "Evidence Evaluation", "Anecdotal vs statistical evidence, correlation vs causation, sample size, peer review.", vec!["evidence", "correlation", "causation", "peer-review"]),
                ("CT.BIAS", "Cognitive Biases", "Confirmation bias, anchoring, availability heuristic, Dunning-Kruger, sunk cost fallacy.", vec!["cognitive-biases", "confirmation-bias", "anchoring"]),
                ("CT.MEDIA", "Media Literacy", "Source verification, detecting misinformation, understanding bias in reporting, fact-checking methods.", vec!["media-literacy", "misinformation", "fact-checking"]),
                ("CT.STAT", "Statistical Reasoning", "Understanding percentages, risk communication, base rate neglect, Simpson's paradox.", vec!["statistics", "risk", "base-rate", "percentages"]),
            ]),
            "health-literacy" => ("Health Literacy", "Health", vec![
                ("HL.BODY", "How Your Body Works", "Major systems (cardiovascular, respiratory, digestive, nervous). What symptoms mean.", vec!["anatomy", "body-systems", "symptoms"]),
                ("HL.NUTR", "Nutrition & Diet", "Macronutrients, micronutrients, balanced diet, hydration, reading food labels.", vec!["nutrition", "diet", "macronutrients", "food-labels"]),
                ("HL.MENTAL", "Mental Health", "Stress management, anxiety, depression awareness, when to seek help, stigma reduction.", vec!["mental-health", "stress", "anxiety", "depression"]),
                ("HL.FIRST", "First Aid Basics", "CPR, choking response, wound care, burns, when to call emergency services.", vec!["first-aid", "cpr", "emergency"]),
                ("HL.PREVENT", "Disease Prevention", "Vaccines, hygiene, sleep, exercise, screening, chronic disease prevention.", vec!["prevention", "vaccines", "hygiene", "exercise"]),
            ]),
            "systems-thinking" => ("Systems Thinking", "Systems Thinking", vec![
                ("SYS.FEED", "Feedback Loops", "Positive (reinforcing) and negative (balancing) feedback. Examples in nature, economy, technology.", vec!["feedback-loops", "reinforcing", "balancing"]),
                ("SYS.STOCK", "Stocks and Flows", "Accumulation and depletion. Bathtub analogy. Why systems resist change.", vec!["stocks", "flows", "accumulation"]),
                ("SYS.EMERGE", "Emergence", "How simple rules create complex behaviour. Ant colonies, traffic, markets, consciousness.", vec!["emergence", "complexity", "self-organisation"]),
                ("SYS.LEVER", "Leverage Points", "Donella Meadows' 12 leverage points. Where to intervene in a system for maximum effect.", vec!["leverage-points", "meadows", "intervention"]),
                ("SYS.MENTAL", "Mental Models", "How assumptions shape perception. Challenging mental models. Double-loop learning.", vec!["mental-models", "assumptions", "double-loop-learning"]),
                ("SYS.MAP", "System Mapping", "Causal loop diagrams, stock-and-flow diagrams, iceberg model, behaviour-over-time graphs.", vec!["system-mapping", "causal-loops", "iceberg-model"]),
            ]),
            "emotional-intelligence" => ("Emotional Intelligence", "Emotional Intelligence", vec![
                ("EQ.SELF", "Self-Awareness", "Recognising emotions, understanding triggers, emotional vocabulary, mindfulness.", vec!["self-awareness", "emotions", "mindfulness"]),
                ("EQ.MGMT", "Self-Management", "Emotion regulation, impulse control, stress management, adaptability, growth mindset.", vec!["self-management", "regulation", "growth-mindset"]),
                ("EQ.SOCIAL", "Social Awareness", "Empathy, perspective-taking, reading social cues, cultural awareness, compassion.", vec!["empathy", "perspective-taking", "social-awareness"]),
                ("EQ.REL", "Relationship Skills", "Active listening, clear communication, teamwork, conflict resolution, boundary-setting.", vec!["relationships", "listening", "conflict-resolution"]),
                ("EQ.DECIDE", "Responsible Decision-Making", "Ethical considerations, consequence evaluation, help-seeking, reflective practice.", vec!["decision-making", "ethics", "reflection"]),
            ]),
            "learning-skills" => ("Learning How to Learn", "Learning Science", vec![
                ("LEARN.META", "Metacognition", "Thinking about thinking. Planning, monitoring, evaluating your own learning. Self-testing.", vec!["metacognition", "self-testing", "reflection"]),
                ("LEARN.SRS", "Spaced Repetition", "The forgetting curve. How spaced review beats massed practice. Optimal spacing intervals.", vec!["spaced-repetition", "forgetting-curve", "memory"]),
                ("LEARN.DELIB", "Deliberate Practice", "Purposeful practice vs naive practice. Feedback loops. Working at the edge of ability.", vec!["deliberate-practice", "feedback", "skill-development"]),
                ("LEARN.INTER", "Interleaving & Variation", "Mixing topics during study. Why it feels harder but produces better long-term learning.", vec!["interleaving", "variation", "desirable-difficulties"]),
                ("LEARN.RECALL", "Active Recall", "Retrieval practice. Why re-reading is ineffective. Testing effect. Elaborative interrogation.", vec!["active-recall", "testing-effect", "retrieval-practice"]),
                ("LEARN.SLEEP", "Sleep & Learning", "Memory consolidation during sleep. Sleep stages. How sleep deprivation impairs learning.", vec!["sleep", "memory-consolidation", "rest"]),
            ]),
            _ => {
                // Simple stub for other subjects
                let (t, s, n) = match id {
                    "digital-literacy" => ("Digital Literacy", "Digital Literacy", vec![
                        ("DL.INFO", "Information Literacy", "Search effectively, evaluate sources, manage information.", vec!["search", "sources", "information"]),
                        ("DL.SAFE", "Digital Safety", "Strong passwords, phishing recognition, privacy settings, 2FA.", vec!["passwords", "phishing", "privacy", "2fa"]),
                        ("DL.CREATE", "Digital Creation", "Documents, spreadsheets, presentations, basic coding concepts.", vec!["documents", "spreadsheets", "coding"]),
                        ("DL.CITIZEN", "Digital Citizenship", "Online ethics, digital footprint, responsible sharing, copyright.", vec!["ethics", "footprint", "copyright"]),
                    ]),
                    "civic-literacy" => ("Civic Literacy", "Civic Literacy", vec![
                        ("CIV.GOV", "How Government Works", "Branches of government, constitution, separation of powers.", vec!["government", "constitution", "separation-of-powers"]),
                        ("CIV.RIGHTS", "Rights & Responsibilities", "Bill of rights, human rights, civic duties, rule of law.", vec!["rights", "responsibilities", "human-rights"]),
                        ("CIV.VOTE", "Voting & Elections", "Electoral systems, informed voting, political parties, media in politics.", vec!["voting", "elections", "political-parties"]),
                        ("CIV.ENGAGE", "Civic Engagement", "Community participation, activism, volunteering, local government.", vec!["engagement", "volunteering", "community"]),
                    ]),
                    "communication" => ("Communication Skills", "Communication", vec![
                        ("COM.WRITE", "Clear Writing", "Structure, clarity, audience awareness, editing, professional email.", vec!["writing", "clarity", "structure"]),
                        ("COM.SPEAK", "Public Speaking", "Presentation skills, overcoming anxiety, storytelling, body language.", vec!["speaking", "presentations", "storytelling"]),
                        ("COM.LISTEN", "Active Listening", "Reflective listening, asking good questions, non-verbal cues.", vec!["listening", "questions", "non-verbal"]),
                        ("COM.PERSUADE", "Persuasion & Negotiation", "Rhetoric, framing, finding common ground, win-win negotiation.", vec!["persuasion", "negotiation", "rhetoric"]),
                    ]),
                    "sustainability" => ("Sustainability", "Sustainability", vec![
                        ("SUST.CLIMATE", "Climate Science", "Greenhouse effect, carbon cycle, evidence, projections, tipping points.", vec!["climate", "greenhouse", "carbon"]),
                        ("SUST.RESOURCE", "Resource Management", "Renewable vs non-renewable, water, food systems, circular economy.", vec!["resources", "renewable", "circular-economy"]),
                        ("SUST.ACTION", "Sustainable Action", "Individual footprint, collective action, policy, SDGs, greenwashing.", vec!["action", "sdgs", "footprint"]),
                        ("SUST.JUST", "Environmental Justice", "Climate inequality, indigenous rights, pollution burden, just transition.", vec!["justice", "inequality", "just-transition"]),
                    ]),
                    "economics-basics" => ("Economics Basics", "Economics", vec![
                        ("ECON.SUPPLY", "Supply & Demand", "Market equilibrium, price signals, elasticity, surpluses and shortages.", vec!["supply", "demand", "equilibrium"]),
                        ("ECON.INCENT", "Incentives", "How incentives shape behaviour, moral hazard, externalities, Pigouvian taxes.", vec!["incentives", "externalities", "moral-hazard"]),
                        ("ECON.MACRO", "Macroeconomics Basics", "GDP, inflation, unemployment, monetary/fiscal policy, trade.", vec!["gdp", "inflation", "unemployment", "monetary-policy"]),
                        ("ECON.BEHAV", "Behavioural Economics", "Nudges, loss aversion, framing, bounded rationality, choice architecture.", vec!["behavioural-economics", "nudges", "loss-aversion"]),
                    ]),
                    "statistics-data" => ("Statistics & Data Literacy", "Statistics", vec![
                        ("STAT.PROB", "Probability Basics", "Events, conditional probability, Bayes' theorem, expected value.", vec!["probability", "bayes", "expected-value"]),
                        ("STAT.DESC", "Descriptive Statistics", "Mean, median, mode, standard deviation, distributions, visualisation.", vec!["descriptive", "distributions", "visualisation"]),
                        ("STAT.INFER", "Inference & Hypothesis Testing", "Confidence intervals, p-values, significance, Type I/II errors.", vec!["inference", "p-values", "hypothesis-testing"]),
                        ("STAT.BIAS", "Data Bias & Ethics", "Selection bias, survivorship bias, Simpson's paradox, ethical data use.", vec!["bias", "survivorship", "simpsons-paradox", "ethics"]),
                        ("STAT.VIZ", "Data Visualisation", "Charts, graphs, dashboards, misleading visualisations, storytelling with data.", vec!["visualisation", "charts", "dashboards"]),
                    ]),
                    _ => return Err(SourceError::NotFound(id.into())),
                };
                (t, s, n)
            }
        };

        let prefix = id.split('-').next().unwrap_or("UNI").to_uppercase();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for (i, (code, node_title, desc, tags)) in nodes_data.iter().enumerate() {
            let node_id = format!("UNI.{}.{}", prefix, code);
            nodes.push(CurriculumNode {
                id: node_id.clone(), title: node_title.to_string(), description: desc.to_string(),
                node_type: "Concept".into(), difficulty: "Beginner".into(),
                domain: subject.into(), subdomain: code.split('.').next().unwrap_or("General").into(),
                tags: tags.iter().map(|t| t.to_string()).chain([subject.to_lowercase()].into_iter()).collect(),
                estimated_hours: 6, grade_levels: vec!["Adult".into()],
                bloom_level: "Understand".into(), subject_area: subject.into(),
                academic_standards: vec![AcademicStandardRef { framework: "EduNet Universal Curriculum".into(), code: node_id.clone(), description: desc.to_string(), grade_level: "Adult".into() }],
                credit_hours: None, course_level: None, cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![], lab_rubric: None, exam_weight: None,
            });
            if i > 0 {
                let prev_code = nodes_data[i-1].0;
                edges.push(CurriculumEdge { from: format!("UNI.{}.{}", prefix, prev_code), to: node_id, edge_type: "Recommends".into(), strength_permille: 600, rationale: "Sequential learning".into() });
            }
        }

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: title.into(), framework: "EduNet Universal Curriculum".into(),
                source: "https://edunet.luminousdynamics.io".into(), grade_level: "Adult".into(),
                subject_area: subject.into(), domain: subject.into(), version: "2026".into(),
                total_standards: nodes.len(), domains: vec![subject.into()],
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: format!("Essential {} knowledge curated for universal access.", subject),
                academic_level: None, institution: None, cip_code: None, total_credits: None, duration_semesters: None,
            },
            nodes, edges,
        })
    }
}
