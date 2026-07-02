// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mock content generator for development.
//!
//! Returns mathematically correct, grade-appropriate content for Common Core
//! Grade 3 Math standards. This lets developers build the full pipeline and
//! UI without requiring a Symthaea runtime.

use crate::channels::{ContentChannels, ContentIntent};
use crate::pipeline::{ContentGenerator, GenerationOutput};

/// Mock content generator using pre-written templates.
///
/// Returns realistic educational content for Grade 3 Math standards.
/// For unknown standards, returns generic but still mathematically sound content.
pub struct MockGenerator;

impl ContentGenerator for MockGenerator {
    fn generate(&self, channels: &ContentChannels, context: &str, _max_tokens: usize) -> GenerationOutput {
        let text = match channels.intent {
            ContentIntent::TeachConcept => {
                if context.contains("vocabulary") || context.contains("Vocabulary") {
                    mock_vocabulary(context)
                } else if context.contains("misconception") || context.contains("Misconception") {
                    mock_misconception(context)
                } else {
                    mock_explanation(context)
                }
            }
            ContentIntent::GiveExample => mock_worked_example(context),
            ContentIntent::AskQuestion => {
                if context.contains("flashcard") || context.contains("Flashcard") {
                    mock_flashcard(context)
                } else if context.contains("assessment") || context.contains("Assessment") {
                    mock_assessment_item(context)
                } else {
                    mock_practice_problem(context)
                }
            }
            ContentIntent::ProvideHint => mock_hint(context),
            ContentIntent::ExplainMisconception => mock_misconception(context),
            ContentIntent::Encourage => {
                "Great effort! Making mistakes is how we learn. \
                 Let's look at the problem together and try a different approach."
                    .to_string()
            }
            ContentIntent::ReflectOnLearning => {
                "Think about what strategy you used. Could you explain it to a friend? \
                 What other problems could you solve the same way?"
                    .to_string()
            }
        };

        GenerationOutput {
            text,
            coherence: 0.85,
            hallucination_flag: false,
            veto_count: 0,
        }
    }
}

/// Generate a teaching explanation for a standard.
fn mock_explanation(context: &str) -> String {
    if context.contains("3.OA.A.1") || context.contains("Interpret products") {
        "Multiplication is a way to add equal groups quickly. \
         When we say 4 x 3, we mean 4 groups with 3 in each group. \
         Think of 4 bags, each with 3 apples \u{2014} that's 12 apples total! \
         We can also see this as an array: 4 rows with 3 in each row. \
         The answer to a multiplication problem is called the product."
            .to_string()
    } else if context.contains("3.OA.A.2") || context.contains("quotients") {
        "Division is about sharing equally or making equal groups. \
         When we write 12 / 3, we can think of it two ways: \
         splitting 12 objects into 3 equal groups (each group gets 4), \
         or making groups of 3 from 12 objects (we get 4 groups). \
         The answer to a division problem is called the quotient."
            .to_string()
    } else if context.contains("3.OA.A.3") || context.contains("word problems") {
        "We can use multiplication and division to solve real-world problems. \
         Look for clue words: 'each', 'every', 'per', and 'groups of' often mean multiply. \
         Words like 'share equally', 'split', and 'how many in each' often mean divide. \
         Draw a picture or write an equation with a ? for the unknown number."
            .to_string()
    } else if context.contains("3.OA.A.4") || context.contains("unknown") {
        "Sometimes we know the answer but need to find a missing number. \
         In 8 x ? = 48, we need to find what times 8 gives us 48. \
         We can use division to help: 48 / 8 = 6, so the missing number is 6. \
         Multiplication and division are inverse operations \u{2014} they undo each other."
            .to_string()
    } else if context.contains("3.OA.B.5") || context.contains("properties") {
        "Multiplication has special properties that make it easier to use. \
         The commutative property means 3 x 5 = 5 x 3 (order doesn't matter). \
         The associative property means (2 x 3) x 4 = 2 x (3 x 4) (grouping doesn't matter). \
         The distributive property lets us break apart: 6 x 7 = 6 x (5 + 2) = 30 + 12 = 42."
            .to_string()
    } else if context.contains("3.OA.B.6") || context.contains("unknown-fact") {
        "Division and multiplication are related. If you know that 6 x 7 = 42, \
         then you also know that 42 / 7 = 6 and 42 / 6 = 7. \
         This is called a fact family. Knowing one fact helps you find three others!"
            .to_string()
    } else if context.contains("3.OA.C.7") || context.contains("fluently") {
        "To multiply fluently means you can find the answer quickly and accurately. \
         Start with the facts you know (like 2x, 5x, 10x) and build from there. \
         For example, to find 7 x 8, you could think: \
         7 x 8 = 7 x (5 + 3) = 35 + 21 = 56. \
         Practice makes these facts automatic!"
            .to_string()
    } else if context.contains("3.OA.D.8") || context.contains("two-step") {
        "Some problems need two steps to solve. Read the whole problem first. \
         Figure out what you need to find. Then plan your steps. \
         Example: Sam has 3 bags of 5 marbles and finds 4 more. \
         Step 1: 3 x 5 = 15 marbles in bags. Step 2: 15 + 4 = 19 marbles total."
            .to_string()
    } else if context.contains("3.OA.D.9") || context.contains("patterns") {
        "Patterns are everywhere in math! Look at the multiplication table: \
         all products of 2 are even, all products of 5 end in 0 or 5, \
         and all products of 9 have digits that add up to 9 (like 9, 18, 27). \
         Finding patterns helps you check your work and predict answers."
            .to_string()
    } else if context.contains("3.NF.A.1") || context.contains("fraction 1/b") {
        "A fraction tells us about equal parts of a whole. \
         When we write 1/4, the bottom number (4) tells us how many equal parts, \
         and the top number (1) tells us how many parts we're talking about. \
         Imagine cutting a pizza into 4 equal slices \u{2014} each slice is 1/4 of the pizza!"
            .to_string()
    } else if context.contains("3.NF.A.2") || context.contains("number line") {
        "We can show fractions on a number line! \
         First, divide the space between 0 and 1 into equal parts. \
         For fourths, make 4 equal spaces. \
         The fraction 3/4 is at the third mark from 0. \
         Each mark represents one-fourth of the way from 0 to 1."
            .to_string()
    } else if context.contains("3.NF.A.3") || context.contains("equivalent") {
        "Two fractions are equivalent if they represent the same amount. \
         For example, 1/2 = 2/4 = 3/6 \u{2014} they all mean 'half'. \
         You can see this by folding paper: fold in half, then fold again. \
         The half you marked is now covered by 2 of the 4 equal pieces."
            .to_string()
    } else if context.contains("3.NBT") || context.contains("place value") || context.contains("round") {
        "Place value tells us what each digit in a number is worth. \
         In 346, the 3 means 300, the 4 means 40, and the 6 means 6. \
         When we round, we find the nearest ten or hundred. \
         346 rounds to 350 (nearest ten) or 300 (nearest hundred)."
            .to_string()
    } else if context.contains("3.MD") || context.contains("Measurement") {
        "Measurement helps us describe the world with numbers. \
         We measure time in hours and minutes, liquid in liters, \
         and weight in grams and kilograms. \
         Choosing the right unit makes our measurements useful!"
            .to_string()
    } else if context.contains("3.G") || context.contains("Geometry") {
        "Shapes have special properties we can describe. \
         A quadrilateral has 4 sides. A rhombus has 4 equal sides. \
         A rectangle has 4 right angles. A square is both! \
         We can sort shapes into categories by their properties."
            .to_string()
    } else {
        "Let's explore this concept together! \
         Mathematics is about finding patterns and solving problems. \
         We'll start with something you already know and build from there. \
         Remember: every expert was once a beginner."
            .to_string()
    }
}

/// Generate a worked example with step-by-step solution.
/// Format: problem line, then step -> result lines, then answer line.
fn mock_worked_example(context: &str) -> String {
    if context.contains("3.OA.A.1") || context.contains("products") {
        if context.contains("#1") || context.contains("concrete") {
            "There are 5 baskets. Each basket has 3 oranges. How many oranges in total?\n\
             Draw 5 groups, each with 3 circles -> 5 groups of 3\n\
             Count all the circles: 3 + 3 + 3 + 3 + 3 -> 15\n\
             Write the equation: 5 x 3 = 15 -> 5 x 3 = 15\n\
             There are 15 oranges in total."
                .to_string()
        } else if context.contains("#2") || context.contains("array") || context.contains("representation") {
            "Show 4 x 6 using an array.\n\
             Draw 4 rows -> 4 rows drawn\n\
             Put 6 dots in each row -> 4 rows of 6 dots\n\
             Count the total: 6 + 6 + 6 + 6 = 24 -> 24 dots total\n\
             4 x 6 = 24"
                .to_string()
        } else {
            "A farmer plants 7 rows of tomato plants with 4 plants in each row. How many plants?\n\
             Identify the groups: 7 rows -> 7 groups\n\
             Identify how many in each: 4 plants per row -> 4 in each group\n\
             Multiply: 7 x 4 = 28 -> 28 plants\n\
             The farmer planted 28 tomato plants."
                .to_string()
        }
    } else if context.contains("3.NF.A.1") || context.contains("fraction") {
        if context.contains("#1") || context.contains("concrete") {
            "Show 3/4 of a pizza.\n\
             Draw a circle for the whole pizza -> one whole pizza\n\
             Divide it into 4 equal slices -> 4 equal parts\n\
             Shade 3 of the slices -> 3 parts shaded\n\
             3/4 of the pizza is shaded."
                .to_string()
        } else if context.contains("#2") || context.contains("representation") {
            "Show 2/6 on a number line.\n\
             Draw a line from 0 to 1 -> number line ready\n\
             Divide into 6 equal spaces -> 6 equal segments\n\
             Count 2 spaces from 0 and mark -> point at 2/6\n\
             2/6 is located at the second mark."
                .to_string()
        } else {
            "Sarah ate 2/8 of a chocolate bar. What fraction did she eat?\n\
             The bar is divided into 8 equal pieces -> 8 equal parts\n\
             Sarah ate 2 pieces -> 2 out of 8\n\
             Write the fraction: 2/8 -> 2/8 of the bar\n\
             Sarah ate 2/8 of the chocolate bar."
                .to_string()
        }
    } else if context.contains("3.OA.C.7") || context.contains("fluently") {
        "Find 7 x 8 using a strategy.\n\
         Break 8 into 5 + 3 -> 7 x (5 + 3)\n\
         Multiply each part: 7 x 5 = 35, 7 x 3 = 21 -> two partial products\n\
         Add the parts: 35 + 21 = 56 -> combined result\n\
         7 x 8 = 56"
            .to_string()
    } else {
        "Solve: There are 6 boxes with 4 crayons each. How many crayons?\n\
         Identify: 6 groups of 4 -> 6 x 4\n\
         Use skip counting: 4, 8, 12, 16, 20, 24 -> counted 6 fours\n\
         Write the equation: 6 x 4 = 24 -> 24 crayons\n\
         There are 24 crayons in total."
            .to_string()
    }
}

/// Generate a practice problem.
/// Format: question, answer, explanation, distractors (comma-separated), hints.
fn mock_practice_problem(context: &str) -> String {
    if context.contains("3.OA.A.1") || context.contains("products") {
        if context.contains("easy") {
            "How many stars in total? 3 groups of 2 stars.\n\
             6\n\
             3 groups of 2 means 3 x 2 = 6 stars.\n\
             5, 8, 32\n\
             Think about how many groups there are.\n\
             Draw the groups and count."
                .to_string()
        } else if context.contains("challenging") || context.contains("advanced") {
            "A bookshelf has 8 shelves. Each shelf holds 7 books. How many books fit on the bookshelf?\n\
             56\n\
             8 shelves with 7 books each: 8 x 7 = 56 books.\n\
             15, 48, 63\n\
             How many groups? How many in each group?\n\
             Write it as a multiplication: 8 x 7 = ?"
                .to_string()
        } else {
            "A classroom has 5 tables. Each table has 4 chairs. How many chairs are there?\n\
             20\n\
             5 tables with 4 chairs each: 5 x 4 = 20 chairs.\n\
             9, 15, 24\n\
             What is being counted in equal groups?\n\
             5 groups of 4: try skip counting by 4."
                .to_string()
        }
    } else if context.contains("3.NF.A.1") || context.contains("fraction") {
        "A pie is cut into 6 equal slices. Maria eats 2 slices. What fraction of the pie did she eat?\n\
         2/6\n\
         There are 6 equal parts (denominator) and Maria ate 2 (numerator): 2/6.\n\
         2/3, 6/2, 1/3\n\
         How many total equal pieces is the pie cut into?\n\
         The bottom number is the total pieces, the top is how many she ate."
            .to_string()
    } else if context.contains("3.OA.C.7") || context.contains("fluently") {
        "What is 9 x 6?\n\
         54\n\
         9 x 6 = 54. Strategy: 10 x 6 = 60, subtract one 6: 60 - 6 = 54.\n\
         45, 56, 63\n\
         Can you use a fact you already know to figure this out?\n\
         Try: 9 x 6 = (10 x 6) - (1 x 6)."
            .to_string()
    } else {
        "There are 4 vases. Each vase has 6 flowers. How many flowers are there in all?\n\
         24\n\
         4 groups of 6: 4 x 6 = 24 flowers.\n\
         10, 18, 26\n\
         Think about equal groups.\n\
         4 x 6: try skip counting by 6 four times."
            .to_string()
    }
}

/// Generate a hint at the given level.
fn mock_hint(context: &str) -> String {
    if context.contains("level-1") || context.contains("vague") {
        "What operation do you think this problem is asking you to use? \
         Look for clue words in the problem."
            .to_string()
    } else if context.contains("level-2") || context.contains("moderate") {
        "Try drawing a picture with equal groups. \
         How many groups are there? How many are in each group?"
            .to_string()
    } else {
        "Set up a multiplication equation: ___ x ___ = ? \
         The number of groups goes first, then the number in each group. \
         Now multiply to find the total."
            .to_string()
    }
}

/// Generate vocabulary in "term: definition | example" format.
fn mock_vocabulary(context: &str) -> String {
    if context.contains("3.OA") || context.contains("products") || context.contains("Algebraic") {
        "product: The answer when two numbers are multiplied | The product of 3 and 4 is 12.\n\
         factor: A number being multiplied | In 3 x 4 = 12, the factors are 3 and 4.\n\
         array: Objects arranged in equal rows and columns | An array of 3 rows and 4 columns shows 3 x 4.\n\
         equation: A math sentence with an equals sign | 5 x 3 = 15 is an equation.\n\
         equal groups: Groups that all have the same number of objects | 4 bags of 5 apples are equal groups."
            .to_string()
    } else if context.contains("3.NF") || context.contains("fraction") {
        "fraction: A number that names part of a whole or part of a group | 1/2 of a pizza is a fraction.\n\
         numerator: The top number in a fraction; tells how many parts | In 3/4, the numerator is 3.\n\
         denominator: The bottom number in a fraction; tells how many equal parts in all | In 3/4, the denominator is 4.\n\
         equivalent fractions: Fractions that name the same amount | 1/2 and 2/4 are equivalent fractions.\n\
         unit fraction: A fraction with 1 as the numerator | 1/3 and 1/8 are unit fractions."
            .to_string()
    } else {
        "operation: A mathematical process like addition or multiplication | Multiplication is an operation.\n\
         strategy: A plan for solving a problem | Skip counting is a strategy for multiplication.\n\
         unknown: A number we need to find | In 3 x ? = 12, the unknown is 4."
            .to_string()
    }
}

/// Generate a misconception block in "WRONG: / RIGHT: / WHY:" format.
fn mock_misconception(context: &str) -> String {
    if context.contains("3.OA.A.1") || context.contains("products") {
        "WRONG: Multiplication always makes numbers bigger.\n\
         RIGHT: Multiplication by 1 gives the same number, and by 0 gives 0. For example, 5 x 1 = 5 and 5 x 0 = 0.\n\
         WHY: Students mostly see examples with factors greater than 1 early on, so they overgeneralize.\n\n\
         WRONG: 3 x 4 and 4 x 3 mean different things because the groups are different.\n\
         RIGHT: While the stories may differ (3 groups of 4 vs. 4 groups of 3), the product is the same: 12.\n\
         WHY: Students confuse the meaning of the factors with the result of the operation."
            .to_string()
    } else if context.contains("3.NF") || context.contains("fraction") {
        "WRONG: A bigger denominator means a bigger fraction.\n\
         RIGHT: A bigger denominator means more parts, so each part is smaller. 1/8 < 1/4.\n\
         WHY: Students assume bigger numbers always mean bigger amounts, not realizing that more parts means smaller pieces."
            .to_string()
    } else if context.contains("3.OA.C.7") || context.contains("fluently") {
        "WRONG: You have to memorize every multiplication fact individually.\n\
         RIGHT: You can use strategies and known facts to figure out new ones. For example, 7 x 8 = 7 x (5 + 3) = 35 + 21 = 56.\n\
         WHY: Students are often told to 'just memorize' without being shown that multiplication facts are connected."
            .to_string()
    } else {
        "WRONG: Math problems only have one way to solve them.\n\
         RIGHT: Most problems can be solved multiple ways. Drawing pictures, using number lines, or writing equations are all valid.\n\
         WHY: Students are often shown only one method and assume alternatives are wrong."
            .to_string()
    }
}

/// Generate a flashcard in "front | back" format.
fn mock_flashcard(context: &str) -> String {
    if context.contains("3.OA.A.1") || context.contains("products") {
        "What does 5 x 3 mean? | 5 groups of 3, which equals 15.".to_string()
    } else if context.contains("3.NF.A.1") || context.contains("fraction") {
        "What does 1/4 mean? | 1 part out of 4 equal parts of a whole.".to_string()
    } else if context.contains("3.OA.C.7") || context.contains("fluently") {
        "What is 8 x 7? | 56. Strategy: 8 x 7 = (8 x 5) + (8 x 2) = 40 + 16 = 56.".to_string()
    } else {
        "What is a product in multiplication? | The answer when two numbers are multiplied together."
            .to_string()
    }
}

/// Generate an assessment item.
fn mock_assessment_item(context: &str) -> String {
    if context.contains("Remember") {
        "What is the product of 6 and 3?\n\
         18\n\
         Think: 6 groups of 3."
            .to_string()
    } else if context.contains("Understand") {
        "Explain what 4 x 5 means using words or a picture.\n\
         4 x 5 means 4 groups of 5 objects, which is 20 objects total.\n\
         Draw a picture to help you explain."
            .to_string()
    } else if context.contains("Apply") {
        "A baker makes 6 trays of cookies. Each tray has 8 cookies. How many cookies did the baker make?\n\
         48\n\
         What equation represents this problem?"
            .to_string()
    } else if context.contains("Analyze") {
        "Jake says 5 x 4 = 20 and 4 x 5 = 20. Is he correct? Why do both give the same answer?\n\
         Yes, both equal 20 because multiplication is commutative \u{2014} order does not change the product.\n\
         What property of multiplication does this show?"
            .to_string()
    } else if context.contains("Evaluate") {
        "Lisa solved 7 x 3 by adding 7 + 7 + 7. Did she use a correct strategy? Is there a better way?\n\
         Yes, 7 + 7 + 7 = 21 = 7 x 3 is correct. An alternative: 7 x 3 = (5 x 3) + (2 x 3) = 15 + 6 = 21.\n\
         Compare the two strategies. Which is faster for bigger numbers?"
            .to_string()
    } else if context.contains("Create") {
        "Write your own word problem that can be solved with 3 x 9 = 27.\n\
         Example: A store has 3 shelves with 9 toys on each shelf. How many toys are there?\n\
         Make sure your problem uses equal groups."
            .to_string()
    } else {
        "How many pencils are in 4 boxes if each box has 7 pencils?\n\
         28\n\
         Write a multiplication equation to solve."
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_returns_nonempty_for_known_standards() {
        let gen = MockGenerator;
        let standards = [
            "3.OA.A.1", "3.OA.A.2", "3.OA.A.3", "3.OA.A.4",
            "3.OA.B.5", "3.OA.B.6", "3.OA.C.7", "3.OA.D.8",
            "3.OA.D.9", "3.NF.A.1", "3.NF.A.2", "3.NF.A.3",
        ];

        for code in &standards {
            let context = format!("Teaching {code} to Grade3 student. Standard: {code}");
            let output = gen.generate(
                &ContentChannels::teaching_factual(),
                &context,
                512,
            );
            assert!(
                !output.text.is_empty(),
                "Empty output for standard {}",
                code
            );
            assert!(output.coherence > 0.0);
            assert!(!output.hallucination_flag);
        }
    }

    #[test]
    fn test_mock_returns_content_for_unknown_standard() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::teaching_factual(),
            "Teaching 5.NBT.A.1 to Grade5 student.",
            512,
        );
        assert!(!output.text.is_empty());
    }

    #[test]
    fn test_mock_hint_levels_differ() {
        let gen = MockGenerator;
        let h1 = gen.generate(
            &ContentChannels::hint(1),
            "level-1 hint for 3.OA.A.1",
            128,
        );
        let h3 = gen.generate(
            &ContentChannels::hint(3),
            "level-3 hint for 3.OA.A.1",
            128,
        );
        assert_ne!(h1.text, h3.text);
    }

    #[test]
    fn test_mock_vocabulary_has_multiple_terms() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::teaching_factual(),
            "vocabulary for 3.OA Algebraic",
            256,
        );
        let line_count = output.text.lines().count();
        assert!(line_count >= 3, "Expected at least 3 vocab terms, got {}", line_count);
    }

    #[test]
    fn test_mock_misconception_has_structure() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::misconception_correction(),
            "misconception about 3.OA.A.1 products",
            256,
        );
        assert!(output.text.contains("WRONG:"));
        assert!(output.text.contains("RIGHT:"));
        assert!(output.text.contains("WHY:"));
    }

    #[test]
    fn test_mock_worked_example_has_steps() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::worked_example(),
            "Example #1 for 3.OA.A.1 products concrete",
            384,
        );
        // Should have multiple lines with -> separator
        let step_count = output.text.lines().filter(|l| l.contains("->")).count();
        assert!(step_count >= 2, "Expected at least 2 steps, got {}", step_count);
    }

    #[test]
    fn test_mock_flashcard_format() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::practice_problem(),
            "flashcard for 3.OA.A.1 products",
            128,
        );
        assert!(output.text.contains(" | "), "Flashcard should have front | back format");
    }

    #[test]
    fn test_mock_encouragement() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::encouragement(),
            "student got wrong answer",
            128,
        );
        assert!(output.text.contains("effort") || output.text.contains("learn"));
    }

    #[test]
    fn test_mock_reflection() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::reflection(),
            "after completing multiplication lesson",
            128,
        );
        assert!(!output.text.is_empty());
    }

    #[test]
    fn test_mock_practice_difficulty_varies() {
        let gen = MockGenerator;
        let easy = gen.generate(
            &ContentChannels::practice_problem(),
            "easy practice problem for 3.OA.A.1 products",
            256,
        );
        let hard = gen.generate(
            &ContentChannels::practice_problem(),
            "challenging practice problem for 3.OA.A.1 products",
            256,
        );
        // Different difficulty levels should produce different content
        assert_ne!(easy.text, hard.text);
    }

    #[test]
    fn test_mock_assessment_bloom_levels() {
        let gen = MockGenerator;
        let blooms = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];
        for bloom in &blooms {
            let output = gen.generate(
                &ContentChannels::practice_problem(),
                &format!("assessment {} for 3.OA.A.1", bloom),
                256,
            );
            assert!(!output.text.is_empty(), "Empty output for Bloom's level {}", bloom);
        }
    }

    #[test]
    fn test_mock_fractions_content_correct() {
        let gen = MockGenerator;
        let output = gen.generate(
            &ContentChannels::teaching_factual(),
            "Teaching 3.NF.A.1 fraction 1/b",
            512,
        );
        // Should mention parts and whole
        assert!(output.text.contains("parts") || output.text.contains("equal"));
    }
}
