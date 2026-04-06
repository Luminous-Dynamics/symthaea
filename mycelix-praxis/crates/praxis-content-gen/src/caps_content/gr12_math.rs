// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 12 Mathematics topic content.

use super::TopicContent;

pub(crate) struct CalculusTopic;

impl TopicContent for CalculusTopic {
    fn explanation(&self) -> String {
        "Differential calculus is about finding the rate of change of a function. \
         The derivative f'(x) tells us the gradient of the tangent to the curve at any point x. \
         We define the derivative from first principles as: f'(x) = lim[h→0] (f(x+h) − f(x))/h. \
         For practical differentiation, we use the power rule: if f(x) = xⁿ, then f'(x) = nxⁿ⁻¹. \
         The derivative helps us find turning points (where f'(x) = 0), determine whether a function is \
         increasing or decreasing, and solve optimisation problems.".to_string()
    }

    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Find f'(x) from first principles if f(x) = 3x².\n\
                  Step 1: Write f(x+h) = 3(x+h)² = 3(x² + 2xh + h²) = 3x² + 6xh + 3h² -> f(x+h) expanded\n\
                  Step 2: f(x+h) − f(x) = 3x² + 6xh + 3h² − 3x² = 6xh + 3h² -> difference\n\
                  Step 3: [f(x+h) − f(x)]/h = (6xh + 3h²)/h = 6x + 3h -> divided by h\n\
                  Step 4: lim[h→0] (6x + 3h) = 6x -> take the limit\n\
                  Answer: f'(x) = 6x".to_string(),
            1 => "Given f(x) = x³ − 6x² + 9x + 2, find the turning points and sketch.\n\
                  Step 1: f'(x) = 3x² − 12x + 9 -> differentiate using power rule\n\
                  Step 2: Set f'(x) = 0: 3x² − 12x + 9 = 0, so x² − 4x + 3 = 0 -> solve for x\n\
                  Step 3: (x − 1)(x − 3) = 0, so x = 1 or x = 3 -> factorise\n\
                  Step 4: f(1) = 1 − 6 + 9 + 2 = 6 and f(3) = 27 − 54 + 27 + 2 = 2 -> find y-values\n\
                  Step 5: f''(x) = 6x − 12. f''(1) = −6 < 0 (local max), f''(3) = 6 > 0 (local min) -> classify\n\
                  Answer: Local maximum at (1, 6) and local minimum at (3, 2)".to_string(),
            _ => "A farmer wants to fence a rectangular camp against a river (no fence needed on river side). \
                  He has 120 m of fencing. Find the dimensions that maximise the area.\n\
                  Step 1: Let width = x, then length = 120 − 2x (two widths + one length = 120) -> set up variables\n\
                  Step 2: Area A = x(120 − 2x) = 120x − 2x² -> express area in terms of x\n\
                  Step 3: A'(x) = 120 − 4x -> differentiate\n\
                  Step 4: Set A'(x) = 0: 120 − 4x = 0, so x = 30 -> find critical point\n\
                  Step 5: Length = 120 − 2(30) = 60. A''(x) = −4 < 0, confirming maximum -> verify\n\
                  Answer: Width = 30 m, Length = 60 m, Maximum area = 1800 m²".to_string(),
        }
    }

    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Differentiate f(x) = 5x³ − 2x + 7.\n\
                        15x² − 2\n\
                        Apply the power rule to each term: d/dx(5x³) = 15x², d/dx(−2x) = −2, d/dx(7) = 0.\n\
                        15x³ − 2,15x² + 2,15x² − 2x\n\
                        What rule do we use for each term?,What happens to constants when differentiated?".to_string(),
            301..=600 => "Find the equation of the tangent to y = x² − 4x + 3 at the point where x = 1.\n\
                          y = −2x + 2\n\
                          At x = 1: y = 1 − 4 + 3 = 0. y' = 2x − 4, so m = 2(1) − 4 = −2. Tangent: y − 0 = −2(x − 1), y = −2x + 2.\n\
                          y = −2x + 1,y = 2x − 2,y = −2x\n\
                          First find the y-value at x = 1.,Find the gradient using the derivative. Then use y − y₁ = m(x − x₁).".to_string(),
            601..=800 => "For f(x) = 2x³ − 9x² + 12x − 4, determine the x-values where f is increasing.\n\
                          x < 1 or x > 2\n\
                          f'(x) = 6x² − 18x + 12 = 6(x² − 3x + 2) = 6(x−1)(x−2). f'(x) > 0 when x < 1 or x > 2.\n\
                          x > 2 only,1 < x < 2,x < 2\n\
                          Find f'(x) first.,Set f'(x) = 0 to find critical points. Then test intervals.".to_string(),
            _ => "A closed cylindrical can must hold 500π cm³. Find the radius that minimises the total surface area.\n\
                  r = 5∛2 ≈ 6.3 cm\n\
                  V = πr²h = 500π, so h = 500/r². SA = 2πr² + 2πrh = 2πr² + 1000π/r. \
                  dSA/dr = 4πr − 1000π/r² = 0. 4r³ = 1000, r³ = 250, r = ∛250 ≈ 6.3 cm (or exactly 5∛2). \
                  d²SA/dr² = 4π + 2000π/r³ > 0 ✓\n\
                  r = 10 cm,r = 250 cm,r = √500 cm\n\
                  Express h in terms of r using the volume constraint.,Write SA in terms of r only, then differentiate and set equal to zero.".to_string(),
        }
    }

    fn hint(&self, level: u8) -> String {
        match level {
            1 => "What rule do you use when the exponent is a constant? Think about bringing the power down.".to_string(),
            2 => "Use the power rule: multiply by the exponent, then reduce the exponent by 1. For f(x) = xⁿ, f'(x) = nxⁿ⁻¹.".to_string(),
            _ => "For f(x) = 3x², apply the power rule: bring down the 2, multiply by the coefficient: f'(x) = 2 × 3x¹ = 6x.".to_string(),
        }
    }

    fn misconception(&self) -> String {
        "WRONG: Students often think the derivative of x² is 2x², keeping the original exponent.\n\
         RIGHT: The power rule says d/dx(xⁿ) = nxⁿ⁻¹. So d/dx(x²) = 2x¹ = 2x. The exponent decreases by 1.\n\
         WHY: Students remember to \"bring down the power\" but forget to reduce the exponent. \
         Practice: write xⁿ → nxⁿ⁻¹ as two separate actions.".to_string()
    }

    fn vocabulary(&self) -> String {
        "derivative: The rate of change of a function; the gradient of the tangent at any point | The derivative of f(x) = x² is f'(x) = 2x.\n\
         first principles: Calculating the derivative using the limit definition f'(x) = lim[h→0] (f(x+h)−f(x))/h | We proved f'(x) = 2x from first principles.\n\
         turning point: A point where the function changes from increasing to decreasing (or vice versa); where f'(x) = 0 | The parabola y = x² − 4x has a turning point at x = 2.\n\
         tangent: A straight line that touches the curve at exactly one point and has the same gradient as the curve at that point | The tangent to y = x² at x = 3 has gradient 6.\n\
         optimisation: Using calculus to find the maximum or minimum value of a quantity | We used optimisation to find the dimensions that maximise area.".to_string()
    }

    fn flashcard(&self) -> String {
        "What is the derivative of f(x) = axⁿ? | f'(x) = naxⁿ⁻¹ (multiply by the exponent, reduce exponent by 1)".to_string()
    }

    fn assessment_item(&self, context: &str) -> String {
        if context.contains("Remember") {
            "State the power rule for differentiation.\nf'(x) = nxⁿ⁻¹ for f(x) = xⁿ\n2\nRecall the formula from class.".to_string()
        } else if context.contains("Evaluate") || context.contains("Create") {
            "A box with no lid is made from a 20 cm × 20 cm sheet by cutting squares of side x from each corner. \
             Show that the volume is V = 4x³ − 80x² + 400x, and find x for maximum volume.\n\
             V'(x) = 12x² − 160x + 400 = 0, x = 10/3 ≈ 3.33 cm\n5\n\
             Draw a diagram first. What are the dimensions after folding?".to_string()
        } else {
            "Find the gradient of f(x) = x³ − 3x + 1 at the point where x = 2.\n\
             f'(x) = 3x² − 3, f'(2) = 3(4) − 3 = 9\n3\n\
             Differentiate first, then substitute x = 2.".to_string()
        }
    }
}

pub(crate) struct AlgebraTopic {
    pub(crate) grade: u8,
}

impl TopicContent for AlgebraTopic {
    fn explanation(&self) -> String {
        match self.grade {
            12 => "In Grade 12, you must solve quadratic equations using factorisation, completing the square, \
                   and the quadratic formula x = (−b ± √(b²−4ac)) / 2a. The discriminant Δ = b² − 4ac tells us about \
                   the nature of roots: Δ > 0 means two distinct real roots, Δ = 0 means two equal real roots, \
                   and Δ < 0 means no real roots. You also solve simultaneous equations where one is linear and one is quadratic, \
                   and quadratic inequalities using a sign diagram or parabola sketch.".to_string(),
            11 => "Grade 11 algebra introduces surds (√a), the quadratic formula, and the nature of roots. \
                   You simplify expressions with surds, solve quadratic equations by completing the square, \
                   and use the discriminant to determine how many roots an equation has. \
                   Simultaneous equations now include one linear and one quadratic equation.".to_string(),
            _ => "Grade 10 algebra covers factorisation (common factor, grouping, trinomials, difference of squares, \
                  sum/difference of cubes) and solving linear, quadratic, literal, and simultaneous equations. \
                  You also solve linear inequalities and represent solutions on a number line.".to_string(),
        }
    }

    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Solve: 2x² − 5x − 3 = 0\n\
                  Step 1: Try factorisation: find factors of 2 × (−3) = −6 that add to −5 → these are −6 and +1 -> identify factors\n\
                  Step 2: Rewrite: 2x² − 6x + x − 3 = 0 -> split middle term\n\
                  Step 3: Group: 2x(x − 3) + 1(x − 3) = 0 -> factorise by grouping\n\
                  Step 4: (2x + 1)(x − 3) = 0 -> common bracket\n\
                  Step 5: x = −½ or x = 3 -> solve each factor\n\
                  Answer: x = −½ or x = 3".to_string(),
            1 => "Solve simultaneously: y = x + 1 and x² + y² = 5\n\
                  Step 1: Substitute y = x + 1 into x² + y² = 5 -> substitution method\n\
                  Step 2: x² + (x + 1)² = 5 → x² + x² + 2x + 1 = 5 -> expand\n\
                  Step 3: 2x² + 2x − 4 = 0 → x² + x − 2 = 0 -> simplify\n\
                  Step 4: (x + 2)(x − 1) = 0, so x = −2 or x = 1 -> factorise\n\
                  Step 5: When x = −2, y = −1. When x = 1, y = 2 -> find y-values\n\
                  Answer: (−2, −1) and (1, 2)".to_string(),
            _ => "Determine the nature of the roots of 3x² + 2x + 1 = 0.\n\
                  Step 1: Identify a = 3, b = 2, c = 1 -> read coefficients\n\
                  Step 2: Δ = b² − 4ac = (2)² − 4(3)(1) = 4 − 12 = −8 -> calculate discriminant\n\
                  Step 3: Since Δ < 0, the equation has no real roots -> interpret\n\
                  Answer: Non-real roots (Δ = −8 < 0)".to_string(),
        }
    }

    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Solve: x² − 7x + 12 = 0\n\
                        x = 3 or x = 4\n\
                        Factorise: (x − 3)(x − 4) = 0. Each factor gives a root.\n\
                        x = 3 or x = −4,x = −3 or x = −4,x = 6 or x = 2\n\
                        Find two numbers that multiply to 12 and add to −7.,Try (x − ?)(x − ?) = 0.".to_string(),
            301..=600 => "Solve for x: 3x² − x = 4\n\
                          x = 4/3 or x = −1\n\
                          Rearrange: 3x² − x − 4 = 0. (3x − 4)(x + 1) = 0.\n\
                          x = 4/3 or x = 1,x = −4/3 or x = −1,x = 4 or x = −1\n\
                          First move everything to one side.,Factor or use the quadratic formula.".to_string(),
            601..=800 => "For which values of k does x² + kx + 9 = 0 have equal roots?\n\
                          k = 6 or k = −6\n\
                          Equal roots when Δ = 0: k² − 4(1)(9) = 0, k² = 36, k = ±6.\n\
                          k = 6 only,k = 3 or k = −3,k = 36\n\
                          Equal roots means the discriminant equals zero.,Set b² − 4ac = 0 and solve for k.".to_string(),
            _ => "Solve: 2/(x−1) + 3/(x+2) = 1 (x ≠ 1, x ≠ −2)\n\
                  x = 2 + √7 ≈ 4.65 or x = 2 − √7 ≈ −0.65\n\
                  Multiply through by (x−1)(x+2): 2(x+2) + 3(x−1) = (x−1)(x+2). \
                  Expand: 2x+4+3x−3 = x²+x−2. Simplify: 5x+1 = x²+x−2 → x²−4x−3 = 0. \
                  Using the quadratic formula: x = (4±√(16+12))/2 = (4±√28)/2 = 2±√7. Both values satisfy the restrictions.\n\
                  x = 4 only,x = 1,no solution\n\
                  Multiply every term by the LCD.,Use the quadratic formula on x²−4x−3 = 0. Check restrictions.".to_string(),
        }
    }

    fn hint(&self, level: u8) -> String {
        match level {
            1 => "What type of equation is this? Can you rearrange it to standard form ax² + bx + c = 0?".to_string(),
            2 => "Try factorising first. If that doesn't work, use the quadratic formula: x = (−b ± √(b²−4ac))/(2a).".to_string(),
            _ => "For x² − 7x + 12 = 0: find two numbers that multiply to +12 and add to −7. Those are −3 and −4.".to_string(),
        }
    }

    fn misconception(&self) -> String {
        "WRONG: Students often forget to set the equation equal to zero before factorising. \
         They factorise x² − 5x = 6 as x(x−5) = 6 and then say x = 6 or x − 5 = 6.\n\
         RIGHT: First rearrange: x² − 5x − 6 = 0. Then factorise: (x − 6)(x + 1) = 0. So x = 6 or x = −1.\n\
         WHY: Factorisation only works to find roots when the product equals zero (zero product property).".to_string()
    }

    fn vocabulary(&self) -> String {
        "discriminant: The expression Δ = b² − 4ac that determines the nature of roots | If Δ > 0, the equation has two distinct real roots.\n\
         quadratic formula: x = (−b ± √(b²−4ac))/(2a) for solving ax² + bx + c = 0 | Use the quadratic formula when you can't factorise.\n\
         nature of roots: Whether roots are real/non-real, equal/unequal, rational/irrational | The discriminant tells us the nature of roots.\n\
         surd: An irrational number expressed as a root, e.g., √2 or ∛5 | √8 simplifies to 2√2.\n\
         simultaneous equations: Two or more equations solved together to find common solutions | Solve y = x + 1 and x² + y = 5 simultaneously.".to_string()
    }

    fn flashcard(&self) -> String {
        "What does the discriminant Δ = b² − 4ac tell us? | Δ > 0: two distinct real roots. Δ = 0: two equal real roots. Δ < 0: no real roots.".to_string()
    }

    fn assessment_item(&self, context: &str) -> String {
        if context.contains("Remember") {
            "Write down the quadratic formula.\nx = (−b ± √(b²−4ac))/(2a)\n2\nRecall the formula used when factorisation is not possible.".to_string()
        } else {
            "Solve 2x² + 3x − 2 = 0 by completing the square.\n\
             x = ½ or x = −2\n4\nDivide by the coefficient of x² first.".to_string()
        }
    }
}

pub(crate) struct SequencesTopic;

impl TopicContent for SequencesTopic {
    fn explanation(&self) -> String {
        "Arithmetic sequences have a common difference d: Tₙ = a + (n−1)d. \
         Geometric sequences have a common ratio r: Tₙ = arⁿ⁻¹. \
         For sums: arithmetic Sₙ = n/2[2a + (n−1)d], geometric Sₙ = a(rⁿ − 1)/(r − 1). \
         A convergent geometric series (|r| < 1) has sum to infinity S∞ = a/(1 − r). \
         Sigma notation Σ compactly expresses sums.".to_string()
    }

    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Find S₂₀ for the arithmetic series 3 + 7 + 11 + 15 + ...\n\
                  Step 1: Identify a = 3, d = 7 − 3 = 4 -> find first term and common difference\n\
                  Step 2: Sₙ = n/2[2a + (n−1)d] -> write the formula\n\
                  Step 3: S₂₀ = 20/2[2(3) + (19)(4)] = 10[6 + 76] = 10 × 82 -> substitute\n\
                  Answer: S₂₀ = 820".to_string(),
            1 => "Find the sum to infinity: 18 + 6 + 2 + ⅔ + ...\n\
                  Step 1: r = 6/18 = ⅓, and |⅓| < 1 so series converges -> check convergence\n\
                  Step 2: S∞ = a/(1 − r) = 18/(1 − ⅓) = 18/(⅔) -> apply formula\n\
                  Step 3: S∞ = 18 × 3/2 = 27 -> simplify\n\
                  Answer: S∞ = 27".to_string(),
            _ => "The 3rd term of a geometric sequence is 12 and the 6th term is 96. Find a and r.\n\
                  Step 1: T₃ = ar² = 12 and T₆ = ar⁵ = 96 -> write equations\n\
                  Step 2: Divide: ar⁵/ar² = 96/12, so r³ = 8, r = 2 -> find ratio\n\
                  Step 3: ar² = 12, so a(4) = 12, a = 3 -> find first term\n\
                  Answer: a = 3, r = 2".to_string(),
        }
    }

    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Find T₁₅ for the arithmetic sequence 5, 8, 11, 14, ...\n\
                        47\n\
                        a = 5, d = 3. T₁₅ = 5 + (14)(3) = 5 + 42 = 47.\n\
                        42,50,45\n\
                        What is the common difference?,Use Tₙ = a + (n−1)d with n = 15.".to_string(),
            _ => "Evaluate: Σ(k=1 to ∞) of 3·(½)ᵏ\n\
                  3\n\
                  This is a GP with a = 3/2, r = 1/2. S∞ = (3/2)/(1/2) = 3.\n\
                  3/2,6,1\n\
                  Write out the first few terms.,Is |r| < 1? Use S∞ = a/(1−r).".to_string(),
        }
    }

    fn hint(&self, level: u8) -> String {
        match level {
            1 => "Is this arithmetic (constant difference) or geometric (constant ratio)?".to_string(),
            2 => "For arithmetic: Tₙ = a + (n−1)d. For geometric: Tₙ = arⁿ⁻¹. Identify a and d or r first.".to_string(),
            _ => "You have a = 3 and d = 4. Plug into Sₙ = n/2[2a + (n−1)d].".to_string(),
        }
    }

    fn misconception(&self) -> String {
        "WRONG: Students confuse Tₙ (the nth term) with Sₙ (the sum of n terms).\n\
         RIGHT: Tₙ gives one specific term. Sₙ gives the total of all terms from T₁ to Tₙ.\n\
         WHY: Both use n, but they answer different questions: \"what is the 10th term?\" vs \"what is the sum of the first 10 terms?\"".to_string()
    }

    fn vocabulary(&self) -> String {
        "common difference: The constant value d added between consecutive terms of an arithmetic sequence | In 3, 7, 11, 15, the common difference is 4.\n\
         common ratio: The constant multiplier r between consecutive terms of a geometric sequence | In 2, 6, 18, 54, the common ratio is 3.\n\
         convergent series: A geometric series where |r| < 1, so the sum approaches a finite value | The series 1 + ½ + ¼ + ... converges to 2.\n\
         sigma notation: The symbol Σ used to express the sum of a series compactly | Σ(k=1 to n) k = 1 + 2 + 3 + ... + n.\n\
         general term: The formula Tₙ that gives any term in the sequence | The general term of 5, 8, 11, ... is Tₙ = 3n + 2.".to_string()
    }

    fn flashcard(&self) -> String {
        "When does a geometric series converge, and what is its sum? | Converges when |r| < 1. Sum to infinity S∞ = a/(1 − r).".to_string()
    }

    fn assessment_item(&self, _context: &str) -> String {
        "Find the sum of the first 30 terms of the arithmetic series where T₅ = 15 and T₁₂ = 36.\n\
         S₃₀ = 1365\n5\nFind d first using T₁₂ − T₅ = 7d.".to_string()
    }
}

pub(crate) struct FinanceTopic;
impl TopicContent for FinanceTopic {
    fn explanation(&self) -> String {
        "Grade 12 finance introduces annuities — regular equal payments over time. \
         Future value annuity: F = x[(1+i)ⁿ − 1]/i (saving towards a target). \
         Present value annuity: P = x[1 − (1+i)⁻ⁿ]/i (paying off a loan). \
         Key distinction: future value is for sinking funds (saving), present value is for loans (borrowing). \
         Always convert the interest rate to match the payment period (monthly payments → monthly rate).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "You take out a loan of R200 000 at 12% p.a. compounded monthly. Calculate the monthly repayment over 5 years.\n\
                  Step 1: i = 0.12/12 = 0.01 per month, n = 5 × 12 = 60 months -> convert rate and period\n\
                  Step 2: P = x[1 − (1+i)⁻ⁿ]/i → 200000 = x[1 − (1.01)⁻⁶⁰]/0.01 -> apply PV formula\n\
                  Step 3: (1.01)⁻⁶⁰ = 0.55045, so [1 − 0.55045]/0.01 = 44.955 -> calculate bracket\n\
                  Step 4: x = 200000/44.955 = R4449.09 -> solve for x\n\
                  Answer: Monthly repayment = R4 449.09".to_string(),
            _ => "You save R500 per month at 9% p.a. compounded monthly. How much do you have after 3 years?\n\
                  Step 1: i = 0.09/12 = 0.0075, n = 36 -> convert\n\
                  Step 2: F = 500[(1.0075)³⁶ − 1]/0.0075 -> FV annuity formula\n\
                  Step 3: (1.0075)³⁶ = 1.30865, so [1.30865 − 1]/0.0075 = 41.153 -> calculate\n\
                  Step 4: F = 500 × 41.153 = R20 576.39 -> multiply\n\
                  Answer: R20 576.39".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=400 => "Calculate the monthly payment on a R150 000 loan at 10.5% p.a. compounded monthly over 4 years.\n\
                        R3 835.67\n\
                        i = 0.105/12 = 0.00875, n = 48. P = x[1−(1.00875)⁻⁴⁸]/0.00875. x = 150000/39.108 = R3835.67.\n\
                        R3125.00,R4201.50,R3500.00\n\
                        Convert the annual rate to monthly first.,Use the present value annuity formula.".to_string(),
            _ => "A car costs R350 000. You pay a 15% deposit and finance the rest at 11% p.a. compounded monthly over 6 years. \
                  Find the monthly instalment and the total amount paid.\n\
                  R5 777.83 per month; total R416 403.76\n\
                  Loan = 350000 × 0.85 = R297500. i = 0.11/12, n = 72. x = 297500/[1−(1+i)⁻⁷²]/i.\n\
                  R4861.11,R6500.00,R5200.00\n\
                  Calculate the deposit first. The loan amount is 85% of R350 000.,Remember: total paid = monthly payment × number of months.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level {
            1 => "Are you saving (future value) or paying off a loan (present value)?".to_string(),
            2 => "For loans, use P = x[1 − (1+i)⁻ⁿ]/i. Convert annual rate to monthly.".to_string(),
            _ => "i = annual rate ÷ 12, n = years × 12. Substitute into the PV formula and solve for x.".to_string(),
        }
    }
    fn misconception(&self) -> String {
        "WRONG: Students use the annual interest rate directly instead of converting to monthly.\n\
         RIGHT: If payments are monthly, divide the annual rate by 12: i = r/12.\n\
         WHY: The payment period must match the compounding period in the formula.".to_string()
    }
    fn vocabulary(&self) -> String {
        "annuity: A series of regular, equal payments made at fixed intervals | A home loan with monthly payments is an annuity.\n\
         sinking fund: Money saved regularly to replace an asset or pay a future expense | The company set up a sinking fund for equipment replacement.\n\
         present value: The current worth of a future sum or series of payments | The present value of the loan determines your monthly payment.\n\
         nominal rate: The annual interest rate before adjusting for compounding frequency | A nominal rate of 12% p.a. compounded monthly means 1% per month.\n\
         effective rate: The actual annual rate after accounting for compounding | A nominal 12% monthly gives an effective rate of 12.68%.".to_string()
    }
    fn flashcard(&self) -> String {
        "What is the present value annuity formula? | P = x[1 − (1+i)⁻ⁿ]/i, where x = payment, i = rate per period, n = number of periods".to_string()
    }
    fn assessment_item(&self, _context: &str) -> String {
        "Explain the difference between a future value annuity and a present value annuity. Give an example of each.\n\
         FV: saving towards a goal (e.g., retirement fund). PV: paying off a loan (e.g., home bond).\n4\n\
         Think: which starts with money and which ends with money?".to_string()
    }
}

// Shorter implementations for remaining topics — each has real Matric content

pub(crate) struct FunctionsTopic;
impl TopicContent for FunctionsTopic {
    fn explanation(&self) -> String { "Functions describe relationships between variables. In Grade 12, you revise all function types: \
         linear (y = ax + q), quadratic (y = a(x−p)² + q with turning point (p,q)), hyperbolic (y = a/(x−p) + q with asymptotes), \
         and exponential (y = ab^(x−p) + q). The logarithmic function y = log_b(x) is the inverse of y = bˣ. \
         You must determine equations from graphs, find intersection points, and interpret real-world graphs.".to_string() }
    fn worked_example(&self, index: usize) -> String { match index {
        0 => "Sketch f(x) = −(x−2)² + 9 showing intercepts, turning point, and axis of symmetry.\n\
              Step 1: Shape: a = −1 < 0 so opens down -> identify shape\n\
              Step 2: Turning point at (2, 9) -> read from equation\n\
              Step 3: y-intercept: f(0) = −4 + 9 = 5, so (0, 5) -> set x = 0\n\
              Step 4: x-intercepts: (x−2)² = 9, x = 5 or x = −1 -> set y = 0\n\
              Answer: TP(2,9), y-int(0,5), x-ints(−1,0) and (5,0), axis x = 2".to_string(),
        1 => "Determine the equation of the hyperbola y = a/(x−p) + q with asymptotes x = 1 and y = −2, passing through (3, 0).\n\
              Step 1: From asymptotes: p = 1, q = −2 -> read from asymptotes\n\
              Step 2: y = a/(x−1) − 2. Substitute (3,0): 0 = a/(3−1) − 2 -> substitute point\n\
              Step 3: 2 = a/2, so a = 4 -> solve for a\n\
              Answer: y = 4/(x−1) − 2".to_string(),
        _ => "If f(x) = 2ˣ, determine the equation of its inverse.\n\
              Step 1: Write y = 2ˣ -> start with function\n\
              Step 2: Swap x and y: x = 2ʸ -> reflect in y = x\n\
              Step 3: Take log₂ of both sides: y = log₂(x) -> solve for y\n\
              Answer: f⁻¹(x) = log₂(x)".to_string(),
    } }
    fn practice_problem(&self, _difficulty: u16) -> String { "Determine the equation of the parabola with turning point (3, −4) passing through (5, 0).\n\
         y = (x − 3)² − 4\n\
         y = a(x−3)² − 4. At (5,0): 0 = a(4) − 4, a = 1. So y = (x−3)² − 4.\n\
         y = (x−3)² + 4,y = −(x−3)² − 4,y = 2(x−3)² − 4\n\
         Start with y = a(x−p)² + q where (p,q) is the turning point.,Substitute the given point to find a.".to_string() }
    fn hint(&self, level: u8) -> String { match level { 1 => "What is the turning point form of a parabola?".to_string(), 2 => "y = a(x − p)² + q. Substitute the TP, then use another point to find a.".to_string(), _ => "TP(3,−4) gives y = a(x−3)² − 4. Put in (5,0): 0 = a(4) − 4, so a = 1.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Students read the turning point of y = −2(x−3)² + 5 as (−3, 5).\n\
         RIGHT: The turning point is (3, 5). The formula is y = a(x−p)² + q, so the sign in the bracket is opposite.\n\
         WHY: (x − 3)² means p = +3, not −3. The negative is part of the formula structure.".to_string() }
    fn vocabulary(&self) -> String { "turning point: The minimum or maximum point of a parabola | y = 2(x−1)² + 3 has turning point (1, 3).\nasymptote: A line that a graph approaches but never touches | y = 1/(x−2) + 3 has asymptotes at x = 2 and y = 3.".to_string() }
    fn flashcard(&self) -> String { "What is the turning point of y = a(x − p)² + q? | The turning point is (p, q). If a > 0 it's a minimum; if a < 0 it's a maximum.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Given f(x) = 2/(x+1) − 3, state the equations of the asymptotes and sketch the graph.\n\
         Vertical: x = −1, Horizontal: y = −3\n4\nWhat values make the denominator zero?".to_string() }
}

pub(crate) struct CountingTopic;
impl TopicContent for CountingTopic {
    fn explanation(&self) -> String { "The fundamental counting principle: if event A can happen in m ways and event B in n ways, \
         then A and B together can happen in m × n ways. Factorial: n! = n × (n−1) × ... × 1. \
         Permutations (order matters): nPr = n!/(n−r)!. Combinations (order doesn't matter): nCr = n!/[r!(n−r)!]. \
         Use permutations for arrangements, combinations for selections.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "How many 4-letter codes can be formed from the word SUNDAY if no letter is repeated?\n\
         Step 1: SUNDAY has 6 distinct letters -> count available letters\n\
         Step 2: First position: 6 choices. Second: 5. Third: 4. Fourth: 3 -> apply counting principle\n\
         Step 3: Total = 6 × 5 × 4 × 3 = 360 -> multiply\n\
         Answer: 360 codes".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "A committee of 3 must be chosen from 5 men and 4 women. How many committees have at least 1 woman?\n\
         74\n\
         Total committees = 9C3 = 84. All-male = 5C3 = 10. At least 1 woman = 84 − 10 = 74.\n\
         84,80,64\n\
         Try the complement: total minus all-male.,Total ways to choose 3 from 9 people is 9C3.".to_string() }
    fn hint(&self, level: u8) -> String { match level { 1 => "Does order matter here? That determines permutation vs combination.".to_string(), _ => "\"At least 1\" problems are easier using the complement: total − none.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Students confuse permutations and combinations.\n\
         RIGHT: If order matters (arrangements), use permutations. If order doesn't matter (selections), use combinations.\n\
         WHY: Choosing a committee of {A,B,C} is the same as {C,A,B} — combination. But the code ABC ≠ CAB — permutation.".to_string() }
    fn vocabulary(&self) -> String { "factorial: n! = n × (n−1) × ... × 1, with 0! = 1 | 5! = 120.\npermutation: An ordered arrangement | The permutations of AB are AB and BA.\ncombination: An unordered selection | The combination {A,B} = {B,A} — just one selection.".to_string() }
    fn flashcard(&self) -> String { "What is the formula for combinations (nCr)? | nCr = n! / [r!(n−r)!]".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "In how many ways can the letters of the word MATHEMATICS be arranged?\n11!/(2!2!2!) = 4 989 600\n4\nWhich letters repeat?".to_string() }
}

pub(crate) struct TrigonometryTopic;
impl TopicContent for TrigonometryTopic {
    fn explanation(&self) -> String { "Grade 12 trigonometry combines compound angles, identities, equations, and applications. \
         Key identities: sin(α±β) = sinα·cosβ ± cosα·sinβ, cos(α±β) = cosα·cosβ ∓ sinα·sinβ. \
         Double angles: sin2α = 2sinα·cosα, cos2α = cos²α − sin²α = 1 − 2sin²α = 2cos²α − 1. \
         For triangles: sine rule a/sinA = b/sinB, cosine rule a² = b² + c² − 2bc·cosA, area = ½ab·sinC.".to_string() }
    fn worked_example(&self, index: usize) -> String { match index {
        0 => "Prove that cos2x/(1 + sin2x) = (cosx − sinx)/(cosx + sinx).\n\
              Step 1: LHS numerator = cos²x − sin²x = (cosx − sinx)(cosx + sinx) -> factorise\n\
              Step 2: LHS denominator = 1 + 2sinx·cosx = (sinx + cosx)² -> complete square\n\
              Step 3: Cancel (cosx + sinx) -> simplify\n\
              Answer: LHS = RHS. QED".to_string(),
        1 => "In triangle PQR, PQ = 7, QR = 9, angle Q = 120°. Find PR using the cosine rule.\n\
              Step 1: PR² = PQ² + QR² − 2(PQ)(QR)cosQ = 49 + 81 − 2(7)(9)cos120° -> apply cosine rule\n\
              Step 2: cos120° = −0.5, so PR² = 130 − 126(−0.5) = 130 + 63 = 193 -> substitute\n\
              Step 3: PR = √193 ≈ 13.89 -> take square root\n\
              Answer: PR ≈ 13.89 units".to_string(),
        _ => "Solve for x ∈ [0°, 360°]: sin(x + 30°) = 0.5\n\
              Step 1: Reference angle: sin⁻¹(0.5) = 30° -> find reference\n\
              Step 2: x + 30° = 30° or x + 30° = 150° (sine positive in Q1 and Q2) -> general solution\n\
              Step 3: x = 0° or x = 120° -> solve for x\n\
              Answer: x = 0° or x = 120°".to_string(),
    } }
    fn practice_problem(&self, _difficulty: u16) -> String { "Solve for θ ∈ [0°, 360°]: 2cos²θ − cosθ − 1 = 0\n\
         θ = 0°, 120°, 240°, 360°\n\
         Let c = cosθ: 2c² − c − 1 = 0 → (2c + 1)(c − 1) = 0 → c = −½ or c = 1. \
         cosθ = 1 → θ = 0°, 360°. cosθ = −½ → θ = 120°, 240°.\n\
         θ = 60° and 300°,θ = 120° only,θ = 0° and 180°\n\
         Treat this as a quadratic in cosθ.,Once you find cosθ, think about which quadrants give that value.".to_string() }
    fn hint(&self, level: u8) -> String { match level { 1 => "Can you rewrite this trig equation as a quadratic?".to_string(), 2 => "Let u = cosθ (or sinθ). Solve the quadratic, then find all angles.".to_string(), _ => "2cos²θ − cosθ − 1 = 0: let c = cosθ. Then 2c² − c − 1 = (2c+1)(c−1) = 0.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Students think sin(A + B) = sinA + sinB.\n\
         RIGHT: sin(A + B) = sinA·cosB + cosA·sinB. You cannot distribute sin over addition.\n\
         WHY: Sine is a function, not a multiplier. Test it: sin(30° + 60°) = sin90° = 1, but sin30° + sin60° = 0.5 + 0.866 = 1.366 ≠ 1.".to_string() }
    fn vocabulary(&self) -> String { "compound angle: An angle formed by adding or subtracting two angles | sin(α + β) is a compound angle formula.\nidentity: An equation true for all values of the variable | sin²θ + cos²θ = 1 is a trigonometric identity.\ngeneral solution: The full set of angles satisfying a trig equation | General solution of sinθ = ½ is θ = 30° + k·360° or θ = 150° + k·360°.".to_string() }
    fn flashcard(&self) -> String { "Write the double angle formula for cos2α (three forms). | cos2α = cos²α − sin²α = 1 − 2sin²α = 2cos²α − 1".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "If sinA = 3/5 (A acute) and cosB = −12/13 (B obtuse), find cos(A + B) without a calculator.\n\
         cos(A+B) = cosA·cosB − sinA·sinB = (4/5)(−12/13) − (3/5)(5/13) = −48/65 − 15/65 = −63/65\n5\n\
         Draw right triangles to find the missing ratios.".to_string() }
}

pub(crate) struct GeometryTopic;
impl TopicContent for GeometryTopic {
    fn explanation(&self) -> String { "Grade 12 Euclidean geometry focuses on circle theorems and proofs. \
         You must prove and apply: angle at centre = 2 × angle at circumference, angles in the same segment are equal, \
         opposite angles of a cyclic quad are supplementary, tangent ⊥ radius, \
         tan-chord angle = angle in the alternate segment, and the proportionality theorem.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "In the diagram, O is the centre. AB is a chord and OC ⊥ AB. If AB = 10 and OC = 12, find the radius.\n\
         Step 1: OC ⊥ AB, so C is the midpoint of AB → AC = 5 -> theorem: perpendicular from centre bisects chord\n\
         Step 2: Triangle OCA is right-angled at C -> identify right triangle\n\
         Step 3: OA² = OC² + AC² = 12² + 5² = 144 + 25 = 169 -> Pythagoras\n\
         Step 4: OA = √169 = 13 -> take square root\n\
         Answer: Radius = 13 units".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "ABCD is a cyclic quadrilateral. Â = 75° and Ĉ = 105°. Is ABCD a cyclic quad? Explain.\n\
         Yes, because Â + Ĉ = 75° + 105° = 180° (opposite angles supplementary).\n\
         The sum confirms the converse of the cyclic quad theorem.\n\
         No because angles aren't equal,Yes because all angles = 90°,Cannot determine\n\
         What is the property of opposite angles in a cyclic quad?,Check if opposite angles add to 180°.".to_string() }
    fn hint(&self, level: u8) -> String { match level { 1 => "Which circle theorem applies here? Look at the relationship between the angles.".to_string(), _ => "Opposite angles of a cyclic quad sum to 180°. Check: 75° + 105° = ?".to_string() } }
    fn misconception(&self) -> String { "WRONG: Students confuse 'angle at centre' with 'angle in a semicircle'.\n\
         RIGHT: Angle at centre = 2× angle at circumference (for the same arc). Angle in a semicircle = 90° (special case where the arc is a diameter).\n\
         WHY: The semicircle theorem IS the angle-at-centre theorem when the central angle is 180°: half of 180° = 90°.".to_string() }
    fn vocabulary(&self) -> String { "cyclic quadrilateral: A quadrilateral whose vertices all lie on a circle | ABCD is cyclic if a circle passes through A, B, C, and D.\ntangent: A line that touches a circle at exactly one point | A tangent is perpendicular to the radius at the point of contact.\nsubtend: To form an angle by connecting endpoints of an arc to a point | Arc AB subtends angle AOB at the centre.".to_string() }
    fn flashcard(&self) -> String { "What is the relationship between opposite angles of a cyclic quadrilateral? | They are supplementary (add up to 180°).".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Prove that the angle subtended by an arc at the centre is twice the angle at the circumference.\nConstruction + proof using isosceles triangles and exterior angle theorem.\n6\nDraw a diameter from the circumference point.".to_string() }
}

pub(crate) struct AnalyticalGeomTopic;
impl TopicContent for AnalyticalGeomTopic {
    fn explanation(&self) -> String { "Analytical geometry uses algebra to solve geometric problems on the Cartesian plane. \
         Key tools: distance formula d = √[(x₂−x₁)² + (y₂−y₁)²], midpoint M = ((x₁+x₂)/2, (y₁+y₂)/2), \
         gradient m = (y₂−y₁)/(x₂−x₁). Circle with centre (a,b) and radius r: (x−a)² + (y−b)² = r². \
         Tangent to circle at point P is perpendicular to the radius OP: m_tangent × m_radius = −1.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "Find the equation of the tangent to x² + y² = 25 at the point (3, 4).\n\
         Step 1: Centre is O(0,0), radius = 5 -> identify circle\n\
         Step 2: Gradient of radius OP: m = (4−0)/(3−0) = 4/3 -> find gradient to point\n\
         Step 3: Tangent ⊥ radius: m_tangent = −3/4 -> negative reciprocal\n\
         Step 4: y − 4 = −3/4(x − 3) → y = −3x/4 + 9/4 + 4 → y = −3x/4 + 25/4 -> point-gradient form\n\
         Answer: 3x + 4y = 25 (or y = −¾x + 25/4)".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "Determine whether the point (1, 7) lies inside, on, or outside the circle (x−3)² + (y−4)² = 16.\n\
         Inside\n\
         Substitute: (1−3)² + (7−4)² = 4 + 9 = 13. Since 13 < 16 (LHS < r²), the point lies inside the circle.\n\
         Outside,On the circle,Cannot determine\n\
         Substitute the point into the LHS of the circle equation.,Compare LHS with r². If LHS < r² → inside, = r² → on, > r² → outside.".to_string() }
    fn hint(&self, _level: u8) -> String { "To test if a point is inside/on/outside, substitute into the circle equation and compare with r².".to_string() }
    fn misconception(&self) -> String { "WRONG: Students forget that the tangent-radius relationship is m₁ × m₂ = −1 (perpendicular).\n\
         RIGHT: The tangent at any point on a circle is perpendicular to the radius at that point.\n\
         WHY: This is a theorem — the tangent just 'touches' the circle, meeting the radius at 90°.".to_string() }
    fn vocabulary(&self) -> String { "inclination: The angle θ that a line makes with the positive x-axis | If m = 1, the inclination is 45°.\ncollinear: Points that lie on the same straight line | A, B, C are collinear if m_AB = m_BC.".to_string() }
    fn flashcard(&self) -> String { "What is the equation of a circle with centre (a, b) and radius r? | (x − a)² + (y − b)² = r²".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Find the equation of the circle with diameter endpoints A(−2, 3) and B(4, −1).\n\
         Centre (1, 1), r² = 13: (x−1)² + (y−1)² = 13\n4\nThe centre is the midpoint of the diameter.".to_string() }
}

pub(crate) struct StatisticsTopic;
impl TopicContent for StatisticsTopic {
    fn explanation(&self) -> String { "Grade 12 statistics covers measures of dispersion and regression. \
         Standard deviation σ = √[Σ(xᵢ − x̄)²/n] measures spread around the mean. \
         For grouped data, use the class midpoints. The regression line ŷ = a + bx (least squares) \
         is the line of best fit for bivariate data. The correlation coefficient r (−1 ≤ r ≤ 1) \
         measures the strength and direction of the linear relationship.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "Calculate the standard deviation of: 4, 7, 8, 9, 12.\n\
         Step 1: Mean x̄ = (4+7+8+9+12)/5 = 40/5 = 8 -> find mean\n\
         Step 2: Deviations: (4−8)²=16, (7−8)²=1, (8−8)²=0, (9−8)²=1, (12−8)²=16 -> square differences\n\
         Step 3: Sum = 16+1+0+1+16 = 34 -> add squared deviations\n\
         Step 4: Variance = 34/5 = 6.8 -> divide by n\n\
         Step 5: σ = √6.8 ≈ 2.61 -> take square root\n\
         Answer: σ ≈ 2.61".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "A dataset has mean 45 and standard deviation 5. What percentage of data falls within one standard deviation of the mean for a normal distribution?\n\
         68%\n\
         For normal distributions: ~68% within 1 SD (40 to 50), ~95% within 2 SD, ~99.7% within 3 SD.\n\
         50%,95%,34%\n\
         Think about the empirical rule (68-95-99.7 rule).,One SD from the mean covers the interval (x̄ − σ, x̄ + σ).".to_string() }
    fn hint(&self, _level: u8) -> String { "Remember the empirical rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students divide by (n−1) instead of n when calculating population standard deviation.\n\
         RIGHT: In Matric, we use σ = √[Σ(x−x̄)²/n] (divide by n). The (n−1) version is for sample standard deviation, which is not in the CAPS syllabus.\n\
         WHY: Textbooks and calculators may use both. In the exam, always divide by n.".to_string() }
    fn vocabulary(&self) -> String { "standard deviation: A measure of how spread out data is from the mean | A small σ means data is clustered near the mean.\nvariance: The square of the standard deviation, σ² | Variance = Σ(x−x̄)²/n.\ncorrelation coefficient: r measures linear association (−1 to 1) | r = 0.95 indicates a strong positive correlation.\nregression line: The line of best fit ŷ = a + bx | Use the regression equation to predict values.".to_string() }
    fn flashcard(&self) -> String { "What is the formula for standard deviation? | σ = √[Σ(xᵢ − x̄)²/n] where x̄ is the mean and n is the number of data values.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Given r = −0.92 for a dataset, describe the correlation and state whether a linear model is appropriate.\n\
         Strong negative correlation. A linear model is appropriate because |r| is close to 1.\n3\nWhat does the sign tell you? What does the magnitude tell you?".to_string() }
}

// ---- Physics Topics ----
