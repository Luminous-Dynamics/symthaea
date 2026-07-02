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

stub_topic!(Gr11SurdsTopic, "Surds and Simplification",
    "A surd is an irrational number expressed as a root, like √2 or ∛5. Simplify surds by finding perfect square factors: √72 = √(36 × 2) = 6√2. To add/subtract surds, they must have the same radicand: 3√2 + 5√2 = 8√2. Rationalise denominators by multiplying by the conjugate. Exponent laws extend to rational exponents: a^(m/n) = ⁿ√(a^m).",
    "surd");

stub_topic!(Gr11QuadraticsTopic, "Quadratic and Simultaneous Equations",
    "Completing the square converts ax² + bx + c to a(x + p)² + q. The quadratic formula x = (-b ± √(b²-4ac))/(2a) solves any quadratic. The discriminant Δ = b² - 4ac determines the nature of roots: Δ > 0 gives two distinct real roots, Δ = 0 gives two equal roots, Δ < 0 gives no real roots. Quadratic inequalities use sign diagrams. Simultaneous equations (one linear, one quadratic) are solved by substitution.",
    "discriminant");

stub_topic!(Gr11SequencesTopic, "Patterns, Sequences, and Series",
    "Arithmetic sequences have a constant difference d: Tn = a + (n-1)d. Sum: Sn = n/2[2a + (n-1)d]. Geometric sequences have a constant ratio r: Tn = ar^(n-1). Sum: Sn = a(r^n - 1)/(r - 1). Sigma notation (Σ) compactly expresses sums. A geometric series converges when |r| < 1, with sum to infinity S∞ = a/(1 - r).",
    "series");

stub_topic!(Gr11FinanceTopic, "Finance, Growth, and Decay",
    "Compound growth: A = P(1+i)^n. Compound decay: A = P(1-i)^n. Nominal rate is the stated annual rate; effective rate accounts for compounding: 1 + i_eff = (1 + i_nom/m)^m where m is compounding periods per year. Depreciation: straight-line (fixed amount per year) vs reducing balance (fixed percentage of current value). Always convert rates to match the compounding period.",
    "depreciation");

stub_topic!(Gr11FunctionsTopic, "Functions — Transformations",
    "In Grade 11, you study functions in shifted form: quadratic y = a(x-p)² + q (turning point at (p,q)), hyperbola y = a/(x-p) + q (asymptotes at x = p and y = q), exponential y = ab^(x-p) + q (asymptote at y = q). The parameter p shifts the graph horizontally, q shifts vertically, and a controls stretch and reflection.",
    "transformation");

stub_topic!(Gr11InverseFunctionsTopic, "Inverse Functions",
    "The inverse function reverses the original: if f(x) = y, then f⁻¹(y) = x. To find the inverse, swap x and y and solve for y. The graph of the inverse is a reflection of the original in the line y = x. The inverse of y = ax + q is y = (x - q)/a. The inverse of y = ax² is y = ±√(x/a) — you must restrict the domain. The inverse of y = b^x is y = log_b(x).",
    "inverse");

stub_topic!(Gr11ProbabilityTopic, "Dependent and Independent Events",
    "Independent events: P(A and B) = P(A) × P(B). The occurrence of one does not affect the other. Dependent events: P(A and B) = P(A) × P(B|A), where P(B|A) is the probability of B given A has occurred. Tree diagrams show all outcomes for multi-step experiments. Without replacement changes probabilities (dependent). Contingency tables organise two-way data to test independence.",
    "independent");

stub_topic!(Gr11MeasurementTopic, "Pyramids, Cones, and Spheres",
    "Volume of a pyramid = ⅓ × base area × height. Volume of a cone = ⅓πr²h. Volume of a sphere = ⁴⁄₃πr³. Surface area of a sphere = 4πr². Surface area of a cone = πr² + πrl (where l is slant height). For combination shapes (e.g., cone + hemisphere), calculate each part separately and add. The ⅓ factor for pyramids and cones is the key difference from prisms and cylinders.",
    "pyramid");
