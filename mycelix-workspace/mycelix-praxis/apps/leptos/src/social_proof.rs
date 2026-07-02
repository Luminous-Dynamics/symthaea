// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Social proof and wise feedback — belonging without peers.
//!
//! NSC examiner report data normalized struggle messages, and
//! sovereignty-aware error feedback.

/// NSC exam insight for a curriculum topic.
pub struct ExamInsight {
    pub node_id: &'static str,
    pub year: u16,
    pub percentage_struggled: u8,
    pub common_error: &'static str,
    pub examiner_tip: &'static str,
}

/// Get NSC exam insight for a node (if available).
pub fn exam_insight(node_id: &str) -> Option<&'static ExamInsight> {
    NSC_INSIGHTS.iter().find(|i| i.node_id == node_id)
}

/// Average number of attempts students need (for "normalized struggle" messages).
pub fn expected_attempts(node_id: &str) -> u32 {
    match node_id {
        n if n.contains("CALC") => 6,
        n if n.contains("GEOM") => 8,
        n if n.contains("TRIG") => 5,
        n if n.contains("ELEC") => 5,
        n if n.contains("EQUIL") => 4,
        n if n.contains("ACID") => 5,
        n if n.contains("ORG") => 4,
        _ => 4,
    }
}

/// Wise feedback message based on sovereignty mode.
/// Inspired by Yeager & Walton (2011) — normalizes struggle, conveys high expectations.
pub fn wise_error_message(concept: &str, attempt: u32, is_guardian: bool) -> String {
    let expected = 5; // average
    if is_guardian {
        if attempt <= 2 {
            format!("Not quite — but that's exactly how understanding builds. {} usually takes a few tries.", concept)
        } else {
            format!("Keep going. Most students need {}-{} attempts for {}. You're getting closer.", expected - 1, expected + 1, concept)
        }
    } else {
        if attempt <= 2 {
            format!("Attempt {}. Most students need {}-{} tries for this concept.", attempt, expected - 1, expected + 1)
        } else {
            format!("Attempt {}. This is one of the harder concepts — persistence pays off.", attempt)
        }
    }
}

// NSC examiner report data (from publicly available DBE diagnostic reports)
static NSC_INSIGHTS: &[ExamInsight] = &[
    ExamInsight {
        node_id: "CAPS.Mathematics.Gr12.P1.CALC",
        year: 2025,
        percentage_struggled: 62,
        common_error: "Students confuse the gradient of the tangent with the average gradient between two points.",
        examiner_tip: "Always use f'(x) for the gradient at a point, not the two-point formula.",
    },
    ExamInsight {
        node_id: "CAPS.Mathematics.Gr12.P1.FN",
        year: 2025,
        percentage_struggled: 55,
        common_error: "Students cannot identify how parameters a, p, q affect the shape and position of graphs.",
        examiner_tip: "Practice sketching with different values of a, p, q. The interactive explorer builds this intuition.",
    },
    ExamInsight {
        node_id: "CAPS.Mathematics.Gr12.P2.GEOM",
        year: 2025,
        percentage_struggled: 71,
        common_error: "Students cannot construct logical proof chains with valid reasons for each step.",
        examiner_tip: "Every statement needs a reason. Write the theorem name next to each step.",
    },
    ExamInsight {
        node_id: "CAPS.Mathematics.Gr12.P2.TRIG",
        year: 2025,
        percentage_struggled: 48,
        common_error: "Students distribute sin over addition: sin(A+B) = sinA + sinB (wrong!).",
        examiner_tip: "Use the compound angle formulae. Test with numbers: sin(30+60) \u{2260} sin30 + sin60.",
    },
    ExamInsight {
        node_id: "CAPS.Mathematics.Gr12.P1.ALG",
        year: 2025,
        percentage_struggled: 35,
        common_error: "Students factorise before setting the equation equal to zero.",
        examiner_tip: "Always rearrange to standard form (= 0) before factorising.",
    },
    ExamInsight {
        node_id: "CAPS.PhysicalSciences.Gr12.P1.ELEC1",
        year: 2025,
        percentage_struggled: 68,
        common_error: "Students confuse series and parallel resistance calculations, especially with internal resistance.",
        examiner_tip: "Draw the circuit. Label every V, I, R. Use \u{03b5} = I(R + r) as your starting point.",
    },
    ExamInsight {
        node_id: "CAPS.PhysicalSciences.Gr12.P1.MECH2",
        year: 2025,
        percentage_struggled: 52,
        common_error: "Sign convention errors — students mix up positive direction for velocity and acceleration.",
        examiner_tip: "Choose upward as positive, stick with it. Gravity is always -9.8 m/s\u{00b2}.",
    },
    ExamInsight {
        node_id: "CAPS.PhysicalSciences.Gr12.P2.EQUIL",
        year: 2025,
        percentage_struggled: 58,
        common_error: "Students think a catalyst shifts equilibrium position.",
        examiner_tip: "A catalyst speeds up both directions equally. It does NOT change Kc.",
    },
    ExamInsight {
        node_id: "CAPS.PhysicalSciences.Gr12.P2.ACID",
        year: 2025,
        percentage_struggled: 45,
        common_error: "Confusing strong/weak with concentrated/dilute.",
        examiner_tip: "Strong = fully ionised. Weak = partially ionised. This is about degree of ionisation, not amount.",
    },
    ExamInsight {
        node_id: "CAPS.PhysicalSciences.Gr12.P2.ORG1",
        year: 2025,
        percentage_struggled: 42,
        common_error: "IUPAC naming errors — wrong longest chain or wrong numbering direction.",
        examiner_tip: "Number from the end nearest the functional group. The longest chain must contain the functional group.",
    },
];
