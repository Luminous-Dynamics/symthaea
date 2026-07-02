// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::TopicContent;

macro_rules! stub_topic {
    ($name:ident, $title:expr, $expl:expr, $vocab_term:expr) => {
        pub(crate) struct $name;
        impl TopicContent for $name {
            fn explanation(&self) -> String { $expl.to_string() }
            fn worked_example(&self, _index: usize) -> String {
                format!("Solve a {} problem.\nStep 1: Identify the given information -> read carefully\nStep 2: Apply the relevant formula or method -> show working\nStep 3: State the answer with units -> verify\nAnswer: See worked solution", $title)
            }
            fn practice_problem(&self, _difficulty: u16) -> String {
                format!("Practice: {}\nAnswer\nThe answer follows from applying the correct method.\nWrong answer 1,Wrong answer 2,Wrong answer 3\nThink about what method applies here.,Review the key formula for this topic.", $title)
            }
            fn hint(&self, level: u8) -> String {
                match level { 1 => format!("What concept from {} applies here?", $title), 2 => format!("Apply the method for {} step by step.", $title), _ => format!("Start by writing down what you know about {}.", $title) }
            }
            fn misconception(&self) -> String {
                format!("WRONG: A common error in {}.\nRIGHT: The correct understanding.\nWHY: This mistake is common because students often skip a key step.", $title)
            }
            fn vocabulary(&self) -> String { format!("{}: A key term | Used in this topic.\nconcept: An important idea | Fundamental to understanding.", $vocab_term) }
            fn flashcard(&self) -> String { format!("What is the key idea in {}? | {}", $title, $expl.split('.').next().unwrap_or($title)) }
            fn assessment_item(&self, _context: &str) -> String { format!("Explain {}.\nKey concept answer.\n3\nThink about what you learned.", $title) }
        }
    };
}

stub_topic!(Gr10FactorisationTopic, "Factorisation",
    "In Grade 10, you master factorisation — the reverse of expanding. Methods: (1) Common factor: take out the greatest common factor. (2) Grouping: group terms in pairs and factor each pair. (3) Trinomials: find two numbers that multiply to ac and add to b in ax² + bx + c. (4) Difference of squares: a² - b² = (a+b)(a-b). (5) Sum/difference of cubes: a³ + b³ = (a+b)(a²-ab+b²).",
    "factorise");

stub_topic!(Gr10EquationsTopic, "Equations and Inequalities",
    "Grade 10 equations include linear equations, quadratic equations (solved by factorisation), literal equations (solving for a specific variable in a formula), and simultaneous linear equations (two equations, two unknowns — solved by substitution or elimination). Linear inequalities are solved like equations but the inequality sign flips when multiplying or dividing by a negative number.",
    "inequality");

stub_topic!(Gr10PatternsTopic, "Number Patterns",
    "In Grade 10, you investigate linear sequences where the first difference is constant. The general term is Tn = a + (n-1)d where a is the first term and d is the common difference. You determine the general term from a given sequence, find specific terms, and solve problems where Tn equals a given value.",
    "sequence");

stub_topic!(Gr10FinanceTopic, "Finance and Growth",
    "Simple interest: A = P(1 + in) where P is the principal, i is the interest rate per year, and n is the number of years. Compound interest: A = P(1 + i)^n where interest is calculated on the accumulated amount. Hire purchase uses simple interest on the full cash price minus deposit. Inflation measures how prices increase over time.",
    "interest");

stub_topic!(Gr10FunctionsTopic, "Functions — Linear and Quadratic",
    "A function assigns exactly one output to each input. In Grade 10: linear function y = ax + q (straight line, a = gradient, q = y-intercept), quadratic function y = ax² + q (parabola, a determines shape and direction, q shifts up/down). Domain is the set of allowed x-values, range is the set of y-values the function produces.",
    "function");

stub_topic!(Gr10FunctionsGraphsTopic, "Functions — Hyperbola and Exponential",
    "The hyperbola y = a/x + q has two branches, with asymptotes at x = 0 and y = q. The exponential function y = ab^x + q shows rapid growth (b > 1) or decay (0 < b < 1), with asymptote y = q. For all functions: find intercepts by setting x = 0 or y = 0, determine domain and range, and understand how a and q affect the graph.",
    "asymptote");

stub_topic!(Gr10TrigGraphsTopic, "Trigonometric Functions",
    "In Grade 10, you sketch and interpret graphs of y = sin θ, y = cos θ, and y = tan θ for θ ∈ [0°, 360°]. The sine and cosine graphs oscillate between -1 and 1 with period 360°. The tangent graph has period 180° and vertical asymptotes at 90° and 270°. The parameter a stretches vertically (amplitude) and q shifts the graph up or down.",
    "amplitude");

stub_topic!(Gr10ProbabilityTopic, "Probability",
    "Grade 10 probability introduces Venn diagrams for representing two events. Mutually exclusive events cannot happen at the same time: P(A and B) = 0. Complementary events: P(not A) = 1 - P(A). The addition rule: P(A or B) = P(A) + P(B) - P(A and B). For mutually exclusive events, the last term is zero.",
    "Venn diagram");

stub_topic!(Gr10MeasurementTopic, "Measurement",
    "In Grade 10, you calculate surface area and volume of right prisms (triangular, rectangular) and cylinders. Volume of a prism = area of cross-section × height. Volume of a cylinder = πr²h. Surface area = sum of all face areas. Important: if you scale all dimensions by factor k, area scales by k² and volume scales by k³.",
    "prism");
