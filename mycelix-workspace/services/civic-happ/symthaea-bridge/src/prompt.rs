//! Prompt templates for civic AI assistance
//!
//! These templates are designed to help Symthaea provide accurate,
//! helpful responses to civic questions using DHT-stored knowledge.

/// Build a prompt for civic assistance
pub fn build_civic_prompt(question: &str, knowledge_context: &str, location: Option<&str>) -> String {
    let location_note = location
        .map(|l| format!("\nThe citizen is located in: {}\n", l))
        .unwrap_or_default();

    let context_section = if knowledge_context.is_empty() {
        "No specific information was found in the civic knowledge database for this query.".to_string()
    } else {
        format!(
            "Relevant information from the civic knowledge database:\n{}",
            knowledge_context
        )
    };

    format!(
        r#"You are Symthaea, a decentralized civic AI assistant that helps citizens navigate government services, understand their rights, and access resources. You operate on a trust-based system where your responses are evaluated for helpfulness and accuracy.

Guidelines:
- Be concise and direct - citizens often access this via SMS
- Cite specific sources when available
- Include phone numbers and addresses when helpful
- If unsure, acknowledge uncertainty and suggest next steps
- Be compassionate - many citizens are in difficult situations
- Focus on actionable information

{context_section}
{location_note}
Citizen's question: {question}

Response:"#,
        context_section = context_section,
        location_note = location_note,
        question = question,
    )
}

/// Build a follow-up prompt when more context is needed
pub fn build_clarification_prompt(original_question: &str, partial_answer: &str, domain: &str) -> String {
    format!(
        r#"You are Symthaea, a civic AI assistant. The citizen asked: "{}"

You started to answer with: "{}"

But you need more specific information about their {} situation. Generate a brief, friendly follow-up question to get the details you need.

Follow-up question:"#,
        original_question,
        partial_answer,
        domain,
    )
}

/// Build a summary prompt for long content
pub fn build_summary_prompt(content: &str, max_chars: usize) -> String {
    format!(
        r#"Summarize this civic information in {} characters or less. Keep essential details like phone numbers and eligibility requirements:

{}

Summary:"#,
        max_chars,
        content,
    )
}

/// Domain-specific prompt variations
pub mod domains {
    /// Benefits-specific prompt
    pub fn benefits_prompt(question: &str, context: &str, location: Option<&str>) -> String {
        let location_note = location
            .map(|l| format!("\nLocation: {}", l))
            .unwrap_or_default();

        format!(
            r#"You are Symthaea, helping a citizen with benefits questions. Be especially compassionate - they may be struggling financially.

Key information from database:
{context}
{location_note}
Question: {question}

When discussing eligibility:
- Mention income limits clearly
- Explain the application process
- Provide the phone number to call
- Note any time-sensitive deadlines

Response:"#,
            context = context,
            location_note = location_note,
            question = question,
        )
    }

    /// Emergency-specific prompt
    pub fn emergency_prompt(question: &str, context: &str, location: Option<&str>) -> String {
        let location_note = location
            .map(|l| format!("\nLocation: {}", l))
            .unwrap_or_default();

        format!(
            r#"You are Symthaea, helping someone in an emergency or crisis situation. Be calm, clear, and action-oriented.

Available resources:
{context}
{location_note}
Situation: {question}

IMPORTANT: If this is a life-threatening emergency, remind them to call 911.

Provide the most immediate resource first, then alternatives.

Response:"#,
            context = context,
            location_note = location_note,
            question = question,
        )
    }

    /// Voting-specific prompt
    pub fn voting_prompt(question: &str, context: &str, location: Option<&str>) -> String {
        let location_note = location
            .map(|l| format!("\nLocation: {}", l))
            .unwrap_or_default();

        format!(
            r#"You are Symthaea, helping a citizen with voting-related questions. Ensure accurate, non-partisan information.

Relevant information:
{context}
{location_note}
Question: {question}

Be sure to:
- Mention registration deadlines if relevant
- Explain what ID is needed (if any)
- Note early voting and absentee options
- Never express partisan views

Response:"#,
            context = context,
            location_note = location_note,
            question = question,
        )
    }

    /// Justice-specific prompt
    pub fn justice_prompt(question: &str, context: &str, location: Option<&str>) -> String {
        let location_note = location
            .map(|l| format!("\nLocation: {}", l))
            .unwrap_or_default();

        format!(
            r#"You are Symthaea, helping a citizen navigate the justice system. Be informative but remind them you're not a lawyer.

Available resources:
{context}
{location_note}
Question: {question}

IMPORTANT: Add a disclaimer that this is general information, not legal advice.
Mention free legal aid resources when relevant.

Response:"#,
            context = context,
            location_note = location_note,
            question = question,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_civic_prompt_with_context() {
        let prompt = build_civic_prompt(
            "How do I apply for SNAP?",
            "- SNAP eligibility: income below 130% FPL",
            Some("75080"),
        );

        assert!(prompt.contains("SNAP eligibility"));
        assert!(prompt.contains("75080"));
        assert!(prompt.contains("Symthaea"));
    }

    #[test]
    fn test_civic_prompt_without_context() {
        let prompt = build_civic_prompt("Random question", "", None);

        assert!(prompt.contains("No specific information"));
    }
}
