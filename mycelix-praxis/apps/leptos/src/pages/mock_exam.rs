// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mock Exam Mode — timed practice exams that simulate NSC conditions.
//!
//! Pulls practice problems from lesson JSONs for Grade 12 topics,
//! groups by Paper 1/2, applies a countdown timer, and scores results
//! with per-topic breakdown.

use leptos::prelude::*;
use serde::Deserialize;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::curriculum::{curriculum_graph, use_progress, use_set_progress};
use crate::katex::render_math_html;

// ============================================================
// Types
// ============================================================

#[derive(Clone, Debug, Deserialize)]
struct ExamProblem {
    question: String,
    answer: String,
    #[serde(default)]
    difficulty_permille: u16,
    #[serde(default)]
    topic_title: String,
    #[serde(default)]
    topic_id: String,
    #[serde(default)]
    marks: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExamState {
    Setup,
    InProgress,
    Reviewing,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExamPaper {
    Paper1,
    Paper2,
}

impl ExamPaper {
    fn label(&self) -> &'static str {
        match self {
            ExamPaper::Paper1 => "Paper 1",
            ExamPaper::Paper2 => "Paper 2",
        }
    }
    fn duration_mins(&self) -> u32 {
        180 // 3 hours per paper
    }
    fn short_duration_mins(&self) -> u32 {
        30 // Quick practice mode
    }
}

// ============================================================
// Component
// ============================================================

#[component]
pub fn MockExamPage() -> impl IntoView {
    let (state, set_state) = signal(ExamState::Setup);
    let (paper, set_paper) = signal(ExamPaper::Paper1);
    let (quick_mode, set_quick_mode) = signal(true); // 30 min quick practice by default
    let (problems, set_problems) = signal(Vec::<ExamProblem>::new());
    let (current_q, set_current_q) = signal(0_usize);
    let (answers, set_answers) = signal(Vec::<Option<bool>>::new()); // None = unanswered, Some(true) = correct
    let (time_left, set_time_left) = signal(30 * 60_i32); // seconds
    let (revealed, set_revealed) = signal(false);
    let set_tracker = crate::study_tracker::use_set_tracker();

    // Store interval ID so we can clear it (prevents timer leak).
    // Using RwSignal<Option<i32>> since gloo Interval isn't Send+Sync.
    let timer_id = RwSignal::new(Option::<i32>::None);

    let clear_timer = move || {
        if let Some(id) = timer_id.get_untracked() {
            if let Some(window) = web_sys::window() {
                window.clear_interval_with_handle(id);
            }
            timer_id.set(None);
        }
    };

    // Clear timer on component unmount
    on_cleanup(move || { clear_timer(); });

    // Real NSC-style exam problems — hand-crafted, not templates
    let generate_problems = move |p: ExamPaper, quick: bool| {
        let max_problems = if quick { 10 } else { 25 };

        let pool: Vec<ExamProblem> = match p {
            ExamPaper::Paper1 => vec![
                // ALGEBRA (25m)
                ExamProblem { question: "Solve for $x$: $x^2 - 5x + 6 = 0$".into(), answer: "$x = 2$ or $x = 3$ (factorise: $(x-2)(x-3) = 0$)".into(), difficulty_permille: 200, topic_title: "Algebra".into(), topic_id: "CAPS.Mathematics.Gr12.P1.ALG".into(), marks: 3 },
                ExamProblem { question: "Solve for $x$: $2x^2 + 3x - 5 = 0$".into(), answer: "$x = 1$ or $x = -\\frac{5}{2}$ (formula: $x = \\frac{-3 \\pm \\sqrt{49}}{4}$)".into(), difficulty_permille: 300, topic_title: "Algebra".into(), topic_id: "CAPS.Mathematics.Gr12.P1.ALG".into(), marks: 4 },
                ExamProblem { question: "Solve simultaneously: $y = 2x - 1$ and $x^2 + y = 7$".into(), answer: "$x = 2, y = 3$ or $x = -4, y = -9$. Substitute: $x^2 + 2x - 1 = 7$, $x^2 + 2x - 8 = 0$".into(), difficulty_permille: 500, topic_title: "Algebra".into(), topic_id: "CAPS.Mathematics.Gr12.P1.ALG".into(), marks: 6 },
                ExamProblem { question: "For which values of $p$ will $2x^2 + px + 8 = 0$ have real roots?".into(), answer: "$p \\leq -8$ or $p \\geq 8$. Discriminant: $p^2 - 64 \\geq 0$".into(), difficulty_permille: 600, topic_title: "Algebra".into(), topic_id: "CAPS.Mathematics.Gr12.P1.ALG".into(), marks: 5 },
                // SEQUENCES (25m)
                ExamProblem { question: "The first three terms of an arithmetic sequence are 2; 5; 8; ... Find T\u{2082}\u{2080}.".into(), answer: "T\u{2082}\u{2080} = 2 + (20\u{2212}1)(3) = 59".into(), difficulty_permille: 200, topic_title: "Sequences".into(), topic_id: "CAPS.Mathematics.Gr12.P1.SEQ".into(), marks: 3 },
                ExamProblem { question: "Calculate: \u{03a3}(k=1 to 20) of (3k + 1)".into(), answer: "= 3\u{00b7}(20)(21)/2 + 20 = 630 + 20 = 650".into(), difficulty_permille: 500, topic_title: "Sequences".into(), topic_id: "CAPS.Mathematics.Gr12.P1.SEQ".into(), marks: 5 },
                ExamProblem { question: "A geometric series has $a = 8$ and $r = \\frac{1}{2}$. Calculate $S_\\infty$.".into(), answer: "$S_\\infty = \\frac{a}{1-r} = \\frac{8}{1-0.5} = 16$".into(), difficulty_permille: 400, topic_title: "Sequences".into(), topic_id: "CAPS.Mathematics.Gr12.P1.SEQ".into(), marks: 4 },
                // FINANCE (25m)
                ExamProblem { question: "Alex invests 45,000 at 9.5% p.a. compounded monthly. How much after 6 years?".into(), answer: "A = 45000(1 + 0.095/12)\u{2077}\u{00b2} = R79 167.34".into(), difficulty_permille: 300, topic_title: "Finance".into(), topic_id: "CAPS.Mathematics.Gr12.P1.FIN".into(), marks: 4 },
                ExamProblem { question: "Calculate the monthly repayment on a R600 000 home loan at 11% p.a. compounded monthly over 20 years.".into(), answer: "R6 191.77. Use PV annuity: x = 600000 \u{00d7} 0.11/12 / (1 \u{2212} (1+0.11/12)\u{207b}\u{00b2}\u{2074}\u{2070})".into(), difficulty_permille: 600, topic_title: "Finance".into(), topic_id: "CAPS.Mathematics.Gr12.P1.FIN".into(), marks: 6 },
                ExamProblem { question: "A sinking fund requires R500 000 in 10 years. Interest: 8% p.a. compounded monthly. Find the monthly payment.".into(), answer: "R2 729.85. FV annuity: x = 500000 \u{00d7} (0.08/12) / ((1+0.08/12)\u{00b9}\u{00b2}\u{2070} \u{2212} 1)".into(), difficulty_permille: 500, topic_title: "Finance".into(), topic_id: "CAPS.Mathematics.Gr12.P1.FIN".into(), marks: 5 },
                // FUNCTIONS (25m)
                ExamProblem { question: "Sketch f(x) = \u{2212}(x\u{2212}2)\u{00b2} + 9. State the turning point, axis of symmetry, and range.".into(), answer: "TP: (2, 9). Axis: x = 2. Range: y \u{2264} 9. Opens downward.".into(), difficulty_permille: 400, topic_title: "Functions".into(), topic_id: "CAPS.Mathematics.Gr12.P1.FN".into(), marks: 5 },
                // CALCULUS (35m)
                ExamProblem { question: "Differentiate: $f(x) = 4x^3 - 6x^2 + 2x - 7$".into(), answer: "$f'(x) = 12x^2 - 12x + 2$".into(), difficulty_permille: 200, topic_title: "Calculus".into(), topic_id: "CAPS.Mathematics.Gr12.P1.CALC".into(), marks: 3 },
                ExamProblem { question: "Find the equation of the tangent to y = x\u{00b2} \u{2212} 3x + 1 at x = 4.".into(), answer: "At x=4: y=5, m=2(4)\u{2212}3=5. Tangent: y\u{2212}5=5(x\u{2212}4), y = 5x \u{2212} 15".into(), difficulty_permille: 400, topic_title: "Calculus".into(), topic_id: "CAPS.Mathematics.Gr12.P1.CALC".into(), marks: 5 },
                ExamProblem { question: "Given f(x) = x\u{00b3} \u{2212} 3x\u{00b2} \u{2212} 9x + 27, find the turning points and point of inflection.".into(), answer: "f'(x) = 3x\u{00b2}\u{2212}6x\u{2212}9 = 3(x\u{2212}3)(x+1). TPs: (\u{2212}1, 32) max, (3, 0) min. POI: f''=6x\u{2212}6=0, x=1, (1, 16)".into(), difficulty_permille: 600, topic_title: "Calculus".into(), topic_id: "CAPS.Mathematics.Gr12.P1.CALC".into(), marks: 8 },
                ExamProblem { question: "A rectangular tank (open top) must hold 32 m\u{00b3}. The base is square with side x. Find x for minimum surface area.".into(), answer: "V=x\u{00b2}h=32, h=32/x\u{00b2}. SA=x\u{00b2}+4xh=x\u{00b2}+128/x. SA'=2x\u{2212}128/x\u{00b2}=0, x\u{00b3}=64, x=4m".into(), difficulty_permille: 800, topic_title: "Calculus".into(), topic_id: "CAPS.Mathematics.Gr12.P1.CALC".into(), marks: 10 },
                // COUNTING (15m)
                ExamProblem { question: "How many 4-digit codes can be made from digits 0\u{2013}9 if no digit repeats and it must be even?".into(), answer: "Case: last digit even (0,2,4,6,8). If last=0: 9\u{00d7}8\u{00d7}7=504. If last\u{2260}0: 4\u{00d7}8\u{00d7}8\u{00d7}7=1792. Total=2296".into(), difficulty_permille: 600, topic_title: "Counting".into(), topic_id: "CAPS.Mathematics.Gr12.P2.CNT".into(), marks: 5 },
                // ELECTRICITY (35m)
                ExamProblem { question: "A battery (\u{03b5}=12V, r=0.5\u{03a9}) connects to R=5.5\u{03a9}. Find the current and terminal voltage.".into(), answer: "I = \u{03b5}/(R+r) = 12/6 = 2A. V\u{209c} = \u{03b5}\u{2212}Ir = 12\u{2212}1 = 11V".into(), difficulty_permille: 300, topic_title: "Circuits".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P1.ELEC1".into(), marks: 4 },
                ExamProblem { question: "Two resistors (6\u{03a9} and 3\u{03a9}) in parallel connect to a 4\u{03a9} resistor in series. Find total resistance.".into(), answer: "R\u{209a} = (6\u{00d7}3)/(6+3) = 2\u{03a9}. R\u{209c} = 2+4 = 6\u{03a9}".into(), difficulty_permille: 400, topic_title: "Circuits".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P1.ELEC1".into(), marks: 4 },
                // MECHANICS (30m)
                ExamProblem { question: "A 5 kg block slides down a 30\u{00b0} frictionless incline from 2m height. Find its speed at the bottom.".into(), answer: "mgh = \u{00bd}mv\u{00b2}. v = \u{221a}(2gh) = \u{221a}(2\u{00d7}9.8\u{00d7}2) = 6.26 m/s".into(), difficulty_permille: 400, topic_title: "Energy".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P1.MECH1".into(), marks: 5 },
                ExamProblem { question: "A 0.02 kg bullet at 400 m/s embeds in a 2 kg block. Find the velocity of the block+bullet.".into(), answer: "Conservation of momentum: 0.02(400) = (0.02+2)v. v = 8/2.02 = 3.96 m/s".into(), difficulty_permille: 500, topic_title: "Momentum".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P1.MECH2".into(), marks: 5 },
                // MODERN PHYSICS (18m)
                ExamProblem { question: "The work function of sodium is 3.7\u{00d7}10\u{207b}\u{00b9}\u{2079} J. Find the threshold frequency.".into(), answer: "f\u{2080} = W\u{2080}/h = 3.7\u{00d7}10\u{207b}\u{00b9}\u{2079} / 6.63\u{00d7}10\u{207b}\u{00b3}\u{2074} = 5.58\u{00d7}10\u{00b9}\u{2074} Hz".into(), difficulty_permille: 400, topic_title: "Photoelectric".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P1.MOD1".into(), marks: 4 },
                // DOPPLER (10m)
                ExamProblem { question: "An ambulance siren (f\u{209b}=500Hz) approaches at 30 m/s. Speed of sound=340 m/s. Find observed frequency.".into(), answer: "f\u{2097} = f\u{209b}(v/(v\u{2212}v\u{209b})) = 500(340/310) = 548.4 Hz".into(), difficulty_permille: 500, topic_title: "Doppler".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P1.DOP".into(), marks: 4 },
            ],
            ExamPaper::Paper2 => vec![
                // EUCLIDEAN GEOMETRY (40m)
                ExamProblem { question: "O is the centre of a circle. AB is a chord and OC \u{22a5} AB. If AB = 24 cm and OC = 5 cm, find the radius.".into(), answer: "AC = 12 cm (perpendicular bisects chord). r = \u{221a}(12\u{00b2}+5\u{00b2}) = \u{221a}169 = 13 cm".into(), difficulty_permille: 300, topic_title: "Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.GEOM".into(), marks: 4 },
                ExamProblem { question: "ABCD is a cyclic quad. \u{00c2} = 70\u{00b0} and \u{0108} = x. D\u{0302}\u{0042}\u{0043} (exterior) = 115\u{00b0}. Find x.".into(), answer: "Opposite angles supplementary: x = 180\u{00b0} \u{2212} 70\u{00b0} = 110\u{00b0}".into(), difficulty_permille: 400, topic_title: "Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.GEOM".into(), marks: 5 },
                ExamProblem { question: "PT is a tangent to circle O at T. PAB passes through centre O. PA = 4, AB = 12. Find PT.".into(), answer: "PT\u{00b2} = PA \u{00d7} PB = 4 \u{00d7} 16 = 64. PT = 8".into(), difficulty_permille: 500, topic_title: "Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.GEOM".into(), marks: 5 },
                ExamProblem { question: "In \u{0394}ABC, D is on AB and E is on AC such that DE \u{2225} BC. AD = 3, DB = 5, AE = 4.5. Find EC.".into(), answer: "AD/DB = AE/EC (proportionality). 3/5 = 4.5/EC. EC = 7.5".into(), difficulty_permille: 400, topic_title: "Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.GEOM".into(), marks: 4 },
                // ANALYTICAL GEOMETRY (40m)
                ExamProblem { question: "Find the equation of the circle with centre $(3, -2)$ and radius $5$.".into(), answer: "$(x-3)^2 + (y+2)^2 = 25$".into(), difficulty_permille: 200, topic_title: "Analytical Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.ANAG".into(), marks: 3 },
                ExamProblem { question: "Determine if (1, 7) lies inside, on, or outside the circle (x\u{2212}3)\u{00b2} + (y\u{2212}4)\u{00b2} = 25.".into(), answer: "(1\u{2212}3)\u{00b2}+(7\u{2212}4)\u{00b2} = 4+9 = 13 < 25. Inside the circle.".into(), difficulty_permille: 400, topic_title: "Analytical Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.ANAG".into(), marks: 4 },
                ExamProblem { question: "Find the equation of the tangent to x\u{00b2}+y\u{00b2}=25 at the point (3, 4).".into(), answer: "Gradient of radius = 4/3. Tangent \u{22a5} radius: m = \u{2212}3/4. y\u{2212}4 = \u{2212}3/4(x\u{2212}3). 3x+4y = 25".into(), difficulty_permille: 600, topic_title: "Analytical Geometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.ANAG".into(), marks: 6 },
                // TRIGONOMETRY (40m)
                ExamProblem { question: "Solve: 2sin\u{00b2}\u{03b8} \u{2212} sin\u{03b8} \u{2212} 1 = 0 for \u{03b8} \u{2208} [0\u{00b0}; 360\u{00b0}].".into(), answer: "(2sin\u{03b8}+1)(sin\u{03b8}\u{2212}1)=0. sin\u{03b8}=\u{2212}\u{00bd} or sin\u{03b8}=1. \u{03b8}=90\u{00b0}, 210\u{00b0}, 330\u{00b0}".into(), difficulty_permille: 400, topic_title: "Trigonometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.TRIG".into(), marks: 5 },
                ExamProblem { question: "Prove: cos2A/(1+sin2A) = (cosA\u{2212}sinA)/(cosA+sinA)".into(), answer: "LHS = (cos\u{00b2}A\u{2212}sin\u{00b2}A)/(1+2sinAcosA) = (cosA\u{2212}sinA)(cosA+sinA)/(cosA+sinA)\u{00b2} = RHS".into(), difficulty_permille: 600, topic_title: "Trigonometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.TRIG".into(), marks: 6 },
                ExamProblem { question: "In \u{0394}PQR, PQ = 8, QR = 5, P\u{0302} = 40\u{00b0}. Find QR using the sine rule.".into(), answer: "sin R/PQ = sin P/QR is wrong — use cosine rule: QR\u{00b2} = PQ\u{00b2}+PR\u{00b2}\u{2212}2(PQ)(PR)cos P. Need more info or use sine rule: QR/sinP = PQ/sinR".into(), difficulty_permille: 500, topic_title: "Trigonometry".into(), topic_id: "CAPS.Mathematics.Gr12.P2.TRIG".into(), marks: 5 },
                // STATISTICS (30m)
                ExamProblem { question: "Data: 3, 5, 7, 7, 8, 9, 12, 15. Find the mean, median, Q1, Q3, and IQR.".into(), answer: "Mean=8.25. Median=(7+8)/2=7.5. Q1=(5+7)/2=6. Q3=(9+12)/2=10.5. IQR=4.5".into(), difficulty_permille: 300, topic_title: "Statistics".into(), topic_id: "CAPS.Mathematics.Gr12.P2.STAT".into(), marks: 5 },
                ExamProblem { question: "The regression equation is \u{0177} = 2.3x + 15. Predict y when x = 10. Is prediction at x = 50 reliable if data range is x \u{2208} [2, 20]?".into(), answer: "At x=10: \u{0177}=38. At x=50: extrapolation outside data range \u{2014} NOT reliable.".into(), difficulty_permille: 500, topic_title: "Statistics".into(), topic_id: "CAPS.Mathematics.Gr12.P2.STAT".into(), marks: 5 },
                ExamProblem { question: "r = 0.92 for a dataset. Comment on the correlation and whether the line is suitable for prediction.".into(), answer: "Strong positive linear correlation (r close to 1). The regression line is suitable for interpolation within data range.".into(), difficulty_permille: 400, topic_title: "Statistics".into(), topic_id: "CAPS.Mathematics.Gr12.P2.STAT".into(), marks: 3 },
                // CHEMISTRY Paper 2
                ExamProblem { question: "Calculate the pH of a 0.01 mol/dm\u{00b3} HCl solution.".into(), answer: "HCl is strong acid, fully ionised. [H\u{2083}O\u{207a}] = 0.01. pH = \u{2212}log(0.01) = 2".into(), difficulty_permille: 200, topic_title: "Acids & Bases".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P2.ACID".into(), marks: 3 },
                ExamProblem { question: "25 cm\u{00b3} of 0.1 mol/dm\u{00b3} NaOH neutralises 20 cm\u{00b3} of HCl. Find the concentration of HCl.".into(), answer: "n(NaOH) = 0.1\u{00d7}0.025 = 0.0025 mol. 1:1 ratio. c(HCl) = 0.0025/0.020 = 0.125 mol/dm\u{00b3}".into(), difficulty_permille: 400, topic_title: "Acids & Bases".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P2.ACID".into(), marks: 5 },
                ExamProblem { question: "State Le Chatelier's principle. If temperature increases for N\u{2082}+3H\u{2082}\u{21cc}2NH\u{2083} (\u{0394}H<0), predict the shift.".into(), answer: "System shifts to oppose change. Exothermic forward \u{2192} increase T favours reverse \u{2192} shift LEFT. Less NH\u{2083} produced.".into(), difficulty_permille: 400, topic_title: "Equilibrium".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P2.EQUIL".into(), marks: 4 },
                ExamProblem { question: "Calculate E\u{00b0}cell for the cell: Zn|Zn\u{00b2}\u{207a}||Cu\u{00b2}\u{207a}|Cu. E\u{00b0}(Zn)=\u{2212}0.76V, E\u{00b0}(Cu)=+0.34V.".into(), answer: "E\u{00b0}cell = E\u{00b0}cathode \u{2212} E\u{00b0}anode = 0.34 \u{2212} (\u{2212}0.76) = +1.10V. Spontaneous.".into(), difficulty_permille: 400, topic_title: "Electrochemistry".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P2.ELCHEM".into(), marks: 4 },
                // ORGANIC CHEMISTRY (25m)
                ExamProblem { question: "Name the organic compound: CH\u{2083}CH\u{2082}CH(CH\u{2083})CH\u{2082}OH".into(), answer: "3-methylbutan-1-ol (longest chain = 4C with OH, methyl branch on C3)".into(), difficulty_permille: 300, topic_title: "Organic Chemistry".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P2.ORG1".into(), marks: 3 },
                ExamProblem { question: "Write the balanced equation for the esterification of ethanol and ethanoic acid. Name the product.".into(), answer: "CH\u{2083}COOH + C\u{2082}H\u{2085}OH \u{21cc} CH\u{2083}COOC\u{2082}H\u{2085} + H\u{2082}O. Product: ethyl ethanoate (an ester).".into(), difficulty_permille: 400, topic_title: "Organic Chemistry".into(), topic_id: "CAPS.PhysicalSciences.Gr12.P2.ORG2".into(), marks: 5 },
            ],
        };

        let mut exam_problems = pool;

        // Shuffle deterministically (daily variety)
        let day_seed = (js_sys::Date::new_0().get_date() as usize) * 31 + (js_sys::Date::new_0().get_month() as usize);
        for i in 0..exam_problems.len() {
            let j = (i + day_seed * 7 + i * 13) % exam_problems.len();
            exam_problems.swap(i, j);
        }

        exam_problems.truncate(max_problems);
        exam_problems
    };

    let start_exam = move |_| {
        let p = paper.get_untracked();
        let quick = quick_mode.get_untracked();
        let probs = generate_problems(p, quick);
        let duration = if quick { p.short_duration_mins() } else { p.duration_mins() };

        set_answers.set(vec![None; probs.len()]);
        set_problems.set(probs);
        set_current_q.set(0);
        set_time_left.set((duration * 60) as i32);
        set_revealed.set(false);
        set_state.set(ExamState::InProgress);

        // Record study activity
        set_tracker.update(|t| t.record_study_day());

        // Clear any existing timer before starting a new one
        clear_timer();

        // Start timer via web_sys (raw i32 handle, avoids non-Send gloo Interval)
        let cb = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(move || {
            // Guard: stop ticking if exam is no longer in progress
            if state.get_untracked() != ExamState::InProgress {
                return;
            }
            set_time_left.update(|t| {
                if *t > 0 {
                    *t -= 1;
                } else {
                    set_state.set(ExamState::Complete);
                }
            });
        });
        if let Some(window) = web_sys::window() {
            if let Ok(id) = window.set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                1_000,
            ) {
                cb.forget(); // intentional: closure must outlive the interval; cleared via clear_timer()
                timer_id.set(Some(id));
            }
        }
    };

    let minutes = move || time_left.get() / 60;
    let secs = move || time_left.get() % 60;

    view! {
        <div class="praxis-skill-map">
            <a href="/exam-prep" class="refuge-back-link">"\u{2190} Back to Exam Prep"</a>
            <h1 style="font-size: 1.5rem; margin: 1rem 0 0.5rem">"Mock Exam"</h1>

            {move || match state.get() {
                ExamState::Setup => {
                    view! {
                        <div style="max-width: 500px">
                            <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.6">
                                "Practice under exam conditions. Problems are drawn from Grade 12 topics weighted by exam marks."
                            </p>

                            // Paper selection
                            <div style="display: flex; gap: 0.75rem; margin-bottom: 1rem">
                                <button
                                    class=move || if paper.get() == ExamPaper::Paper1 { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                    on:click=move |_| set_paper.set(ExamPaper::Paper1)
                                >"Paper 1 (Algebra, Calculus, Finance)"</button>
                                <button
                                    class=move || if paper.get() == ExamPaper::Paper2 { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                    on:click=move |_| set_paper.set(ExamPaper::Paper2)
                                >"Paper 2 (Geometry, Trig, Stats)"</button>
                            </div>

                            // Duration selection
                            <div style="display: flex; gap: 0.75rem; margin-bottom: 1.5rem">
                                <button
                                    class=move || if quick_mode.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                    on:click=move |_| set_quick_mode.set(true)
                                >"Quick (30 min, 10 Qs)"</button>
                                <button
                                    class=move || if !quick_mode.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                    on:click=move |_| set_quick_mode.set(false)
                                >"Full Exam (3 hrs, 25 Qs)"</button>
                            </div>

                            <button class="session-start-btn" on:click=start_exam style="max-width: 300px">
                                <span style="font-size: 1.1rem; font-weight: 700">"Start Exam"</span>
                                <span style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.15rem">
                                    {move || paper.get().label()}" \u{2014} "
                                    {move || if quick_mode.get() { "30 minutes" } else { "3 hours" }}
                                </span>
                            </button>
                        </div>
                    }.into_any()
                }

                ExamState::InProgress => {
                    let probs = problems.get();
                    let total = probs.len();
                    let idx = current_q.get().min(total.saturating_sub(1));

                    if total == 0 {
                        return view! { <p>"No problems available for this paper."</p> }.into_any();
                    }

                    let prob = &probs[idx];
                    let question = prob.question.clone();
                    let answer = prob.answer.clone();
                    let topic = prob.topic_title.clone();
                    let marks = prob.marks;
                    let topic_id = prob.topic_id.clone();

                    view! {
                        // Timer bar
                        <div class="exam-timer-bar">
                            <span style="font-family: monospace; font-size: 1rem; font-weight: 600">
                                {move || format!("{:02}:{:02}", minutes(), secs())}
                            </span>
                            <span style="font-size: 0.8rem; color: var(--text-secondary)">
                                "Q "{idx + 1}"/" {total}
                            </span>
                            <button
                                style="font-size: 0.75rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border); border-radius: 4px; background: var(--surface); color: var(--text-secondary); cursor: pointer; font-family: inherit"
                                on:click=move |_| set_state.set(ExamState::Complete)
                            >"End Exam"</button>
                        </div>

                        // Question card
                        <div style="padding: 1.25rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; margin: 1rem 0">
                            <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.5rem">
                                {topic}" \u{2014} "{marks}"m"
                            </div>
                            <div style="font-size: 0.95rem; line-height: 1.7; white-space: pre-line"
                                inner_html=render_math_html(&question)
                            ></div>

                            // Reveal answer
                            {move || if revealed.get() {
                                let ans = answer.clone();
                                let ans_html = render_math_html(&ans);
                                let tid = topic_id.clone();
                                view! {
                                    <div class="praxis-problem-a" style="margin-top: 1rem"
                                        inner_html=ans_html
                                    ></div>
                                    <div style="display: flex; gap: 0.5rem; margin-top: 0.75rem">
                                        <button
                                            class="praxis-filter-btn"
                                            style="font-size: 0.8rem; border-color: var(--mastery-green); color: var(--mastery-green)"
                                            on:click=move |_| {
                                                set_answers.update(|a| { if idx < a.len() { a[idx] = Some(true); } });
                                                set_revealed.set(false);
                                                if idx + 1 < total { set_current_q.set(idx + 1); }
                                                else { set_state.set(ExamState::Complete); }
                                            }
                                        >"Correct"</button>
                                        <button
                                            class="praxis-filter-btn"
                                            style="font-size: 0.8rem"
                                            on:click=move |_| {
                                                set_answers.update(|a| { if idx < a.len() { a[idx] = Some(false); } });
                                                set_revealed.set(false);
                                                if idx + 1 < total { set_current_q.set(idx + 1); }
                                                else { set_state.set(ExamState::Complete); }
                                            }
                                        >"Incorrect"</button>
                                        <a href=format!("/study/{}", tid) style="font-size: 0.8rem; color: var(--info); text-decoration: none; padding: 0.3rem 0.5rem">
                                            "Study this topic \u{2192}"
                                        </a>
                                    </div>
                                }.into_any()
                            } else {
                                view! {
                                    <button
                                        class="praxis-filter-btn active"
                                        style="margin-top: 1rem"
                                        on:click=move |_| set_revealed.set(true)
                                    >"Show Answer"</button>
                                }.into_any()
                            }}
                        </div>

                        // Question navigation dots
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; justify-content: center">
                            {(0..total).map(|i| {
                                let cls = move || {
                                    let a = answers.get();
                                    let is_current = i == current_q.get();
                                    let mut c = String::from("exam-dot");
                                    match a.get(i) {
                                        Some(Some(true)) => c.push_str(" exam-dot-correct"),
                                        Some(Some(false)) => c.push_str(" exam-dot-incorrect"),
                                        _ => {}
                                    }
                                    if is_current { c.push_str(" exam-dot-current"); }
                                    c
                                };
                                view! {
                                    <button
                                        class=cls
                                        on:click=move |_| { set_current_q.set(i); set_revealed.set(false); }
                                    >
                                        {i + 1}
                                    </button>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }

                ExamState::Complete | ExamState::Reviewing => {
                    let a = answers.get();
                    let total = a.len();
                    let correct = a.iter().filter(|x| **x == Some(true)).count();
                    let incorrect = a.iter().filter(|x| **x == Some(false)).count();
                    let unanswered = a.iter().filter(|x| x.is_none()).count();
                    let pct = if total > 0 { correct * 100 / total } else { 0 };

                    let probs = problems.get();
                    let total_marks: u16 = probs.iter().map(|p| p.marks).sum();
                    let earned_marks: u16 = probs.iter().enumerate()
                        .filter(|(i, _)| a.get(*i) == Some(&Some(true)))
                        .map(|(_, p)| p.marks)
                        .sum();

                    let grade = match pct {
                        80..=100 => ("A", "var(--mastery-green)"),
                        70..=79 => ("B", "var(--info)"),
                        60..=69 => ("C", "var(--mastery-yellow)"),
                        50..=59 => ("D", "var(--warning)"),
                        40..=49 => ("E", "var(--warning)"),
                        _ => ("F", "var(--error)"),
                    };

                    view! {
                        <div style="text-align: center; padding: 1rem 0">
                            <div style=format!("font-size: 3rem; font-weight: 700; color: {}", grade.1)>
                                {grade.0}
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 600; margin-top: 0.25rem">
                                {earned_marks}"/"{total_marks}" marks ("{pct}"%)"
                            </div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem">
                                {correct}" correct, "{incorrect}" incorrect, "{unanswered}" skipped"
                            </div>

                            // Per-topic breakdown
                            <div style="margin-top: 1.5rem; text-align: left">
                                <h3 style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem">"Topic Breakdown"</h3>
                                {probs.iter().enumerate().map(|(i, p)| {
                                    let got_it = a.get(i) == Some(&Some(true));
                                    let color = if got_it { "var(--mastery-green)" } else { "var(--text-tertiary)" };
                                    let icon = if got_it { "\u{2714}" } else if a.get(i) == Some(&Some(false)) { "\u{2718}" } else { "\u{2014}" };
                                    let title = p.topic_title.clone();
                                    let marks = p.marks;
                                    view! {
                                        <div style=format!("display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0; font-size: 0.8rem; color: {}", color)>
                                            <span style="width: 1rem; text-align: center">{icon}</span>
                                            <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">{title}</span>
                                            <span style="color: var(--text-tertiary)">{marks}"m"</span>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>

                            <div style="display: flex; gap: 0.75rem; justify-content: center; margin-top: 1.5rem">
                                <button class="praxis-filter-btn active" on:click=move |_| set_state.set(ExamState::Setup)>
                                    "Try Another"
                                </button>
                                <a href="/exam-prep" class="praxis-filter-btn" style="text-decoration: none">"Back to Exam Prep"</a>
                            </div>
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
