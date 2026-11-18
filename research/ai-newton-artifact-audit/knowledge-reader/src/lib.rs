//! Minimal, provenance-preserving reader for AI-Newton native knowledge text.
//!
//! This module classifies source syntax only. An upstream "is conserved" or
//! "is zero" suffix is a native label, not a formal proof or verification result.
//! Raw source lines and line numbers are always retained.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// The syntactic class of a native knowledge entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    /// Ordinary primitive/quantity/object symbol, such as posx or t.
    SymbolDefinition,
    /// A source identifier in the C_<digits> namespace.
    Concept,
    /// A source identifier in the R_<digits> namespace.
    NativeRule,
}

/// A textual suffix reported by AI-Newton for a native rule entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeRuleLabel {
    Conserved,
    Zero,
    Other,
}

/// One parsed line. The original line and line number are kept verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeRecord {
    pub source_line_number: usize,
    pub raw_line: String,
    pub identifier: String,
    pub body: String,
    pub kind: RecordKind,
    /// Present only for R_* entries; never represents a proof status.
    pub native_rule_label: Option<NativeRuleLabel>,
}

/// Counts that can be computed without interpreting expressions as physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnowledgeSummary {
    pub symbol_definitions: usize,
    pub concept_entries: usize,
    pub native_rule_entries: usize,
    pub conserved_labels: usize,
    pub zero_labels: usize,
    pub other_rule_labels: usize,
    /// Duplicate identifiers are reported, not discarded or merged.
    pub duplicate_identifier_entries: usize,
}

/// Parsed content inside the [Knowledge] ... [end] section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeDocument {
    pub records: Vec<KnowledgeRecord>,
    pub summary: KnowledgeSummary,
}

/// Fail-closed syntax errors. No malformed line is silently skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorKind {
    MissingKnowledgeHeader,
    MissingEndMarker,
    MalformedRecord,
    MalformedConceptIdentifier,
    MalformedRuleIdentifier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line_number: usize,
    pub kind: ParseErrorKind,
    pub raw_line: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "AI-Newton knowledge parse error at line {}: {:?}: {}",
            self.line_number, self.kind, self.raw_line
        )
    }
}

impl Error for ParseError {}

fn has_numeric_suffix(identifier: &str, prefix: &str) -> bool {
    identifier
        .strip_prefix(prefix)
        .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn classify_identifier(
    identifier: &str,
    line_number: usize,
    raw_line: &str,
) -> Result<RecordKind, ParseError> {
    if identifier.starts_with("C_") {
        if has_numeric_suffix(identifier, "C_") {
            return Ok(RecordKind::Concept);
        }
        return Err(ParseError {
            line_number,
            kind: ParseErrorKind::MalformedConceptIdentifier,
            raw_line: raw_line.to_owned(),
        });
    }

    if identifier.starts_with("R_") {
        if has_numeric_suffix(identifier, "R_") {
            return Ok(RecordKind::NativeRule);
        }
        return Err(ParseError {
            line_number,
            kind: ParseErrorKind::MalformedRuleIdentifier,
            raw_line: raw_line.to_owned(),
        });
    }

    if identifier.is_empty() || identifier.chars().any(char::is_whitespace) {
        return Err(ParseError {
            line_number,
            kind: ParseErrorKind::MalformedRecord,
            raw_line: raw_line.to_owned(),
        });
    }

    Ok(RecordKind::SymbolDefinition)
}

fn native_rule_label(body: &str) -> NativeRuleLabel {
    let normalized = body.trim();
    if normalized.ends_with(" is conserved") {
        NativeRuleLabel::Conserved
    } else if normalized.ends_with(" is zero") {
        NativeRuleLabel::Zero
    } else {
        NativeRuleLabel::Other
    }
}

/// Parse only the native [Knowledge] section.
///
/// The parser stops at [end]; later sections such as [Experiments] are outside
/// this reader's scope. Every non-empty line inside the section must contain
/// one := separator. Expressions are retained as strings; they are not
/// evaluated, normalized, or treated as verified physical laws.
pub fn parse_knowledge(input: &str) -> Result<KnowledgeDocument, ParseError> {
    let mut header_seen = false;
    let mut end_seen = false;
    let mut records = Vec::new();

    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();

        if !header_seen {
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "[Knowledge]" {
                header_seen = true;
                continue;
            }
            return Err(ParseError {
                line_number,
                kind: ParseErrorKind::MissingKnowledgeHeader,
                raw_line: line.to_owned(),
            });
        }

        if trimmed == "[end]" {
            end_seen = true;
            break;
        }

        if trimmed.is_empty() {
            continue;
        }

        let Some((left, right)) = line.split_once(":=") else {
            return Err(ParseError {
                line_number,
                kind: ParseErrorKind::MalformedRecord,
                raw_line: line.to_owned(),
            });
        };

        let identifier = left.trim();
        let body = right.trim();
        let kind = classify_identifier(identifier, line_number, line)?;
        let native_rule_label = match kind {
            RecordKind::NativeRule => Some(native_rule_label(body)),
            RecordKind::SymbolDefinition | RecordKind::Concept => None,
        };

        records.push(KnowledgeRecord {
            source_line_number: line_number,
            raw_line: line.to_owned(),
            identifier: identifier.to_owned(),
            body: body.to_owned(),
            kind,
            native_rule_label,
        });
    }

    if !header_seen {
        return Err(ParseError {
            line_number: 1,
            kind: ParseErrorKind::MissingKnowledgeHeader,
            raw_line: String::new(),
        });
    }

    if !end_seen {
        return Err(ParseError {
            line_number: input.lines().count().max(1),
            kind: ParseErrorKind::MissingEndMarker,
            raw_line: String::new(),
        });
    }

    let mut seen = HashSet::new();
    let mut summary = KnowledgeSummary {
        symbol_definitions: 0,
        concept_entries: 0,
        native_rule_entries: 0,
        conserved_labels: 0,
        zero_labels: 0,
        other_rule_labels: 0,
        duplicate_identifier_entries: 0,
    };

    for record in &records {
        if !seen.insert(record.identifier.as_str()) {
            summary.duplicate_identifier_entries += 1;
        }

        match record.kind {
            RecordKind::SymbolDefinition => summary.symbol_definitions += 1,
            RecordKind::Concept => summary.concept_entries += 1,
            RecordKind::NativeRule => {
                summary.native_rule_entries += 1;
                match record.native_rule_label {
                    Some(NativeRuleLabel::Conserved) => summary.conserved_labels += 1,
                    Some(NativeRuleLabel::Zero) => summary.zero_labels += 1,
                    Some(NativeRuleLabel::Other) | None => summary.other_rule_labels += 1,
                }
            }
        }
    }

    Ok(KnowledgeDocument { records, summary })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = concat!(
        "[Knowledge]\n",
        "cy := (1->Slope) |- cy[1]\n",
        "C_01 := (1->Particle) |- D[posz[1]]/D[t[0]]\n",
        "R_10 := |- C_51 is conserved\n",
        "R_11 := (1->Particle) |- C_58[1] is zero\n",
        "R_12 := |- candidate_status_unknown\n",
        "[end]\n",
        "[Experiments]\n",
        "this section is intentionally ignored\n",
    );

    #[test]
    fn parses_native_symbols_concepts_and_rule_labels() {
        let parsed = parse_knowledge(SAMPLE).unwrap();
        assert_eq!(parsed.records.len(), 5);
        assert_eq!(parsed.records[0].kind, RecordKind::SymbolDefinition);
        assert_eq!(parsed.records[1].kind, RecordKind::Concept);
        assert_eq!(parsed.records[2].kind, RecordKind::NativeRule);
        assert_eq!(parsed.records[2].native_rule_label, Some(NativeRuleLabel::Conserved));
        assert_eq!(parsed.records[3].native_rule_label, Some(NativeRuleLabel::Zero));
        assert_eq!(parsed.records[4].native_rule_label, Some(NativeRuleLabel::Other));
    }

    #[test]
    fn retains_exact_raw_line_and_original_line_number() {
        let parsed = parse_knowledge(SAMPLE).unwrap();
        assert_eq!(
            parsed.records[1].raw_line,
            "C_01 := (1->Particle) |- D[posz[1]]/D[t[0]]"
        );
        assert_eq!(parsed.records[1].source_line_number, 3);
    }

    #[test]
    fn reports_counts_without_claiming_proof() {
        let parsed = parse_knowledge(SAMPLE).unwrap();
        assert_eq!(
            parsed.summary,
            KnowledgeSummary {
                symbol_definitions: 1,
                concept_entries: 1,
                native_rule_entries: 3,
                conserved_labels: 1,
                zero_labels: 1,
                other_rule_labels: 1,
                duplicate_identifier_entries: 0,
            }
        );
    }

    #[test]
    fn retains_duplicate_identifiers_instead_of_merging_them() {
        let input = "[Knowledge]\nC_01 := x\nC_01 := y\n[end]\n";
        let parsed = parse_knowledge(input).unwrap();
        assert_eq!(parsed.records.len(), 2);
        assert_eq!(parsed.summary.duplicate_identifier_entries, 1);
    }

    #[test]
    fn malformed_records_fail_closed() {
        let input = "[Knowledge]\nthis is not a record\n[end]\n";
        let error = parse_knowledge(input).unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::MalformedRecord);
        assert_eq!(error.line_number, 2);
    }

    #[test]
    fn malformed_native_identifiers_fail_closed() {
        let input = "[Knowledge]\nC_bad := x\n[end]\n";
        let error = parse_knowledge(input).unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::MalformedConceptIdentifier);
    }

    #[test]
    fn missing_header_and_end_marker_fail_closed() {
        let missing_header = parse_knowledge("x := y\n[end]\n").unwrap_err();
        assert_eq!(missing_header.kind, ParseErrorKind::MissingKnowledgeHeader);

        let missing_end = parse_knowledge("[Knowledge]\nx := y\n").unwrap_err();
        assert_eq!(missing_end.kind, ParseErrorKind::MissingEndMarker);
    }

    #[test]
    fn empty_knowledge_section_is_valid() {
        let parsed = parse_knowledge("[Knowledge]\n[end]\n").unwrap();
        assert!(parsed.records.is_empty());
    }
}
