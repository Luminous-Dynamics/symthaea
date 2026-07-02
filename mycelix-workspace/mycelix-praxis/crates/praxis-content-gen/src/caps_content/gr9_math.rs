// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 9 Mathematics topic content — foundational skills for FET Phase.

use super::TopicContent;

macro_rules! stub_topic {
    ($name:ident, $title:expr, $expl:expr, $vocab_term:expr) => {
        pub(crate) struct $name;
        impl TopicContent for $name {
            fn explanation(&self) -> String { $expl.to_string() }
            fn worked_example(&self, _index: usize) -> String {
                format!("Solve a basic {} problem.\nStep 1: Identify what is given -> read the question\nStep 2: Apply the method -> work through\nStep 3: Check your answer -> verify\nAnswer: See worked solution above", $title)
            }
            fn practice_problem(&self, _difficulty: u16) -> String {
                format!("Practice {} problem.\nAnswer\nExplanation of the answer.\nDistractor 1,Distractor 2,Distractor 3\nHint: Think about the key concept.,Hint: Apply the method step by step.", $title)
            }
            fn hint(&self, level: u8) -> String {
                match level {
                    1 => format!("What is the key concept in {}?", $title),
                    2 => format!("Remember the method for {}. Apply it step by step.", $title),
                    _ => format!("For {}: start by identifying what you know, then apply the formula.", $title),
                }
            }
            fn misconception(&self) -> String {
                format!("WRONG: A common mistake in {}.\nRIGHT: The correct approach.\nWHY: Understanding why students make this error helps avoid it.", $title)
            }
            fn vocabulary(&self) -> String { format!("{}: A key term in this topic | Used in Grade 9 mathematics.", $vocab_term) }
            fn flashcard(&self) -> String { format!("What is the key concept in {}? | {}", $title, $expl) }
            fn assessment_item(&self, _context: &str) -> String {
                format!("Explain the key idea behind {}.\nThe main concept is...\n3\nThink about what you learned.", $title)
            }
        }
    };
}

stub_topic!(Gr9WholeNumbersTopic, "Whole Numbers and Integers",
    "In Grade 9, you work with integers — positive and negative whole numbers. BODMAS (Brackets, Orders, Division, Multiplication, Addition, Subtraction) tells you the order to do calculations. When multiplying integers: positive × positive = positive, negative × negative = positive, positive × negative = negative. You also learn prime factorisation, HCF (highest common factor) and LCM (lowest common multiple).",
    "integer");

stub_topic!(Gr9FractionsDecimalsTopic, "Fractions, Decimals, and Percentages",
    "Grade 9 covers converting between fractions, decimals, and percentages. To convert a fraction to a decimal, divide the numerator by the denominator. To convert to a percentage, multiply by 100. Ratio and proportion are used to compare quantities and solve problems like sharing amounts or scaling recipes.",
    "ratio");

stub_topic!(Gr9ExponentsTopic, "Exponents",
    "Exponent laws let you simplify expressions with powers. Key rules: a^m × a^n = a^(m+n), a^m ÷ a^n = a^(m-n), (a^m)^n = a^(mn), (ab)^n = a^n × b^n, a^0 = 1 (for a ≠ 0). Scientific notation writes very large or small numbers as a number between 1 and 10 multiplied by a power of 10.",
    "exponent");

stub_topic!(Gr9AlgebraExpressionsTopic, "Algebraic Expressions",
    "In Grade 9 algebra, you learn to expand brackets using the distributive law: a(b + c) = ab + ac. You simplify by collecting like terms (terms with the same variable and exponent). Factorising is the reverse — finding common factors. For binomials: (x + a)(x + b) = x² + (a+b)x + ab. Difference of two squares: a² - b² = (a + b)(a - b).",
    "expand");

stub_topic!(Gr9AlgebraEquationsTopic, "Algebraic Equations",
    "Solving equations means finding the value of the variable that makes the equation true. For linear equations, use inverse operations to isolate the variable: if 3x + 7 = 22, subtract 7 from both sides (3x = 15), then divide by 3 (x = 5). Always do the same operation to both sides. Word problems: translate the English into an equation, then solve.",
    "equation");

stub_topic!(Gr9PatternsTopic, "Numeric and Geometric Patterns",
    "A pattern is a sequence of numbers or shapes that follows a rule. In a linear pattern, the difference between consecutive terms is constant. The general term (nth term) of a linear sequence is Tn = a + (n-1)d, where a is the first term and d is the common difference. You can use the general term to find any term without listing them all.",
    "sequence");

stub_topic!(Gr9FunctionsTopic, "Functions and Relationships",
    "A function is a rule that assigns exactly one output to each input. You can represent functions as equations (y = 2x + 3), tables of values, or graphs on the Cartesian plane. The Cartesian plane has an x-axis (horizontal) and y-axis (vertical). Each point is written as (x, y). Linear functions produce straight-line graphs.",
    "function");

stub_topic!(Gr9GeomStraightLinesToopic, "Geometry of Straight Lines",
    "Angles on a straight line add up to 180° (supplementary). Vertically opposite angles are equal. When a transversal crosses parallel lines: corresponding angles are equal (F-pattern), alternate angles are equal (Z-pattern), and co-interior angles add to 180° (C-pattern or U-pattern).",
    "transversal");

stub_topic!(Gr9Geom2DShapesTopic, "Geometry of 2D Shapes",
    "The angles of any triangle add up to 180°. The Theorem of Pythagoras states that in a right-angled triangle, a² + b² = c² where c is the hypotenuse (longest side, opposite the right angle). Triangles are congruent (identical) if SSS, SAS, ASA, or RHS conditions are met. Similar triangles have the same shape but different sizes.",
    "Pythagoras");

stub_topic!(Gr9MeasurementTopic, "Area, Perimeter, and Volume",
    "Perimeter is the distance around a shape. Area is the space inside a 2D shape: rectangle = l × w, triangle = ½ × b × h, circle = πr². Volume is the space inside a 3D shape: rectangular prism = l × w × h, cylinder = πr²h. Surface area is the total area of all faces. Always include the correct units: cm for length, cm² for area, cm³ for volume.",
    "volume");

stub_topic!(Gr9DataHandlingTopic, "Data Handling",
    "The mean is the sum of all values divided by the number of values. The median is the middle value when data is arranged in order. The mode is the most frequent value. The range is the difference between the highest and lowest values. Data can be displayed using bar graphs, histograms, pie charts, and line graphs.",
    "mean");

stub_topic!(Gr9ProbabilityTopic, "Probability",
    "Probability measures how likely an event is to occur, on a scale from 0 (impossible) to 1 (certain). P(event) = number of favourable outcomes / total number of outcomes. The probability of an event NOT happening is 1 - P(event). Tree diagrams show all possible outcomes for two or more events and help calculate combined probabilities.",
    "probability");
