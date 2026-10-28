//! Read-only, evidence-preserving fixed-format COBOL source indexer.
//!
//! This is intentionally NOT a COBOL parser, compiler, or behavior verifier. It
//! extracts source anchors and declarations without expanding COPY books or
//! claiming dialect-specific semantic meaning. Ambiguous fixed-column inputs
//! fail closed with diagnostics rather than guessing at column boundaries.

use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexOptions {
    /// Include lines marked with the fixed-format debug indicator 'D'.
    pub include_debug_lines: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            include_debug_lines: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub line: usize,
    /// One-based physical source column.
    pub column_start: usize,
    /// One-based inclusive physical source column.
    pub column_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedSpan {
    pub name: String,
    pub location: SourceLocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyReference {
    pub name: String,
    pub location: SourceLocation,
    pub has_replacing_clause: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataItem {
    pub level: u8,
    pub name: String,
    pub picture: Option<String>,
    pub usage: Option<String>,
    pub redefines: Option<String>,
    pub location: SourceLocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedBlock {
    pub kind: String,
    pub location: SourceLocation,
    pub has_end_exec: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CobolIndex {
    pub schema_version: String,
    pub source_format: String,
    pub physical_line_count: usize,
    pub program_id: Option<NamedSpan>,
    pub divisions: Vec<NamedSpan>,
    pub copy_references: Vec<CopyReference>,
    pub file_descriptions: Vec<NamedSpan>,
    pub data_items: Vec<DataItem>,
    pub embedded_blocks: Vec<EmbeddedBlock>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Word,
    StringLiteral,
    Punctuation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    value: String,
    kind: TokenKind,
    location: SourceLocation,
}

#[derive(Debug, Clone)]
struct FixedRecord {
    line: usize,
    indicator: u8,
    text: String,
    text_start_column: usize,
    continuation: bool,
}

fn location(line: usize, start: usize, end: usize) -> SourceLocation {
    SourceLocation {
        line,
        column_start: start,
        column_end: end.max(start),
    }
}

fn diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    code: &str,
    message: impl Into<String>,
    loc: Option<SourceLocation>,
) {
    diagnostics.push(Diagnostic {
        code: code.to_owned(),
        message: message.into(),
        location: loc,
    });
}

fn physical_records(input: &str, options: IndexOptions, diagnostics: &mut Vec<Diagnostic>) -> Vec<FixedRecord> {
    let physical_lines: Vec<&str> = input.split('\n').collect();
    let mut records = Vec::new();
    let mut opaque_continued_literal = false;

    for (index, original) in physical_lines.iter().enumerate() {
        let line_number = index + 1;
        let line = original.strip_suffix('\r').unwrap_or(original);

        if line.contains('\r') {
            diagnostic(
                diagnostics,
                "bare_carriage_return",
                "bare carriage return in source record; fixed-format columns are not trusted",
                Some(location(line_number, 1, line.len().max(1))),
            );
            continue;
        }
        if !line.is_ascii() {
            diagnostic(
                diagnostics,
                "non_ascii_fixed_format_record",
                "source record is not ASCII; decode the source encoding before column-sensitive indexing",
                Some(location(line_number, 1, line.len().max(1))),
            );
            continue;
        }
        if line.as_bytes().contains(&b'\t') {
            diagnostic(
                diagnostics,
                "tab_in_fixed_format_record",
                "tab expansion is implementation-dependent; fixed-format columns are not guessed",
                Some(location(line_number, 1, line.len().max(1))),
            );
            continue;
        }

        let bytes = line.as_bytes();
        let indicator = bytes.get(6).copied().unwrap_or(b' ');
        if matches!(indicator, b'*' | b'/') {
            continue;
        }
        if indicator == b'D' || indicator == b'd' {
            if !options.include_debug_lines {
                continue;
            }
        } else if !matches!(indicator, b' ' | b'-') {
            diagnostic(
                diagnostics,
                "unknown_fixed_format_indicator",
                format!("unrecognized indicator column byte {:?}", indicator as char),
                Some(location(line_number, 7, 7)),
            );
            continue;
        }

        let continuation = indicator == b'-';
        if opaque_continued_literal {
            if continuation {
                diagnostic(
                    diagnostics,
                    "continued_literal_opaque",
                    "continued literal continuation record was excluded from indexing to avoid treating literal text as source code",
                    Some(location(line_number, 7, 7)),
                );
                continue;
            }
            opaque_continued_literal = false;
        }

        // Fixed source area is physical columns 8-72 inclusive. Columns 73+
        // are identification-area bytes and are deliberately ignored.
        let start = 7usize;
        let end = bytes.len().min(72);
        let source_text = if bytes.len() > start {
            &line[start..end]
        } else {
            ""
        };
        let record = FixedRecord {
            line: line_number,
            indicator,
            text: source_text.to_owned(),
            text_start_column: 8,
            continuation,
        };
        if tokenizes_to_unterminated_literal(&record) {
            opaque_continued_literal = true;
        }
        records.push(record);
    }
    records
}

fn tokenizes_to_unterminated_literal(record: &FixedRecord) -> bool {
    let bytes = record.text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte != b'\'' && byte != b'"' {
            i += 1;
            continue;
        }
        let quote = byte;
        i += 1;
        let mut closed = false;
        while i < bytes.len() {
            if bytes[i] == quote {
                if bytes.get(i + 1) == Some(&quote) {
                    i += 2;
                } else {
                    i += 1;
                    closed = true;
                    break;
                }
            } else {
                i += 1;
            }
        }
        if !closed {
            return true;
        }
    }
    false
}

fn tokenize_record(record: &FixedRecord, diagnostics: &mut Vec<Diagnostic>) -> Vec<Token> {
    let bytes = record.text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        let byte = bytes[i];
        if byte.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        if byte == b'\'' || byte == b'"' {
            let quote = byte;
            i += 1;
            let content_start = i;
            let mut output = String::new();
            let mut closed = false;
            while i < bytes.len() {
                if bytes[i] == quote {
                    if bytes.get(i + 1) == Some(&quote) {
                        output.push(quote as char);
                        i += 2;
                    } else {
                        output.push_str(&record.text[content_start..i]);
                        i += 1;
                        closed = true;
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            if !closed {
                diagnostic(
                    diagnostics,
                    "unterminated_literal",
                    "literal is not closed within this physical source record; continuation is treated as opaque",
                    Some(location(
                        record.line,
                        record.text_start_column + start,
                        record.text_start_column + bytes.len().saturating_sub(1),
                    )),
                );
                break;
            }
            tokens.push(Token {
                value: output,
                kind: TokenKind::StringLiteral,
                location: location(
                    record.line,
                    record.text_start_column + start,
                    record.text_start_column + i.saturating_sub(1),
                ),
            });
            continue;
        }

        if byte.is_ascii_alphanumeric() || byte == b'_' {
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-' || bytes[i] == b'_')
            {
                i += 1;
            }
            tokens.push(Token {
                value: record.text[start..i].to_ascii_uppercase(),
                kind: TokenKind::Word,
                location: location(
                    record.line,
                    record.text_start_column + start,
                    record.text_start_column + i.saturating_sub(1),
                ),
            });
            continue;
        }

        i += 1;
        tokens.push(Token {
            value: (byte as char).to_string(),
            kind: TokenKind::Punctuation,
            location: location(
                record.line,
                record.text_start_column + start,
                record.text_start_column + start,
            ),
        });
    }
    tokens
}

fn is_word(token: &Token, word: &str) -> bool {
    token.kind == TokenKind::Word && token.value == word
}

fn next_non_period(tokens: &[Token], mut index: usize) -> Option<usize> {
    while index < tokens.len() && tokens[index].value == "." {
        index += 1;
    }
    (index < tokens.len()).then_some(index)
}

fn statement_end(tokens: &[Token], start: usize) -> usize {
    let mut index = start;
    while index < tokens.len() {
        if tokens[index].value == "." && tokens[index].kind == TokenKind::Punctuation {
            return index;
        }
        index += 1;
    }
    tokens.len()
}

fn joined_tokens(tokens: &[Token]) -> String {
    let mut result = String::new();
    for token in tokens {
        match token.value.as_str() {
            "(" | ")" | "." | "," | ":" => result.push_str(&token.value),
            _ => {
                if !result.is_empty()
                    && !result.ends_with('(')
                    && !result.ends_with(':')
                    && !result.ends_with('.')
                {
                    result.push(' ');
                }
                result.push_str(&token.value);
            }
        }
    }
    result
}

pub fn index_fixed_format(input: &str, options: IndexOptions) -> CobolIndex {
    let physical_line_count = input.split('\n').count();
    let mut diagnostics = Vec::new();
    let records = physical_records(input, options, &mut diagnostics);
    let mut tokens = Vec::new();
    let mut previous_last_token: Option<Token> = None;

    for record in &records {
        let record_tokens = tokenize_record(record, &mut diagnostics);
        if record.continuation {
            if let (Some(previous), Some(first)) = (previous_last_token.as_ref(), record_tokens.first()) {
                if previous.kind == TokenKind::Word
                    && first.kind == TokenKind::Word
                    && previous.location.line + 1 == first.location.line
                {
                    diagnostic(
                        &mut diagnostics,
                        "continuation_token_boundary_ambiguous",
                        "word appears at the start of a continuation record; the indexer does not concatenate tokens across physical lines",
                        Some(first.location.clone()),
                    );
                }
            }
        }
        if let Some(last) = record_tokens.last() {
            previous_last_token = Some(last.clone());
        }
        tokens.extend(record_tokens);
    }

    let mut program_id: Option<NamedSpan> = None;
    let mut divisions = Vec::new();
    let mut copy_references = Vec::new();
    let mut file_descriptions = Vec::new();
    let mut data_items = Vec::new();
    let mut embedded_blocks = Vec::new();
    let mut active_division: Option<String> = None;

    for i in 0..tokens.len() {
        let token = &tokens[i];

        if is_word(token, "IDENTIFICATION")
            && tokens.get(i + 1).is_some_and(|next| is_word(next, "DIVISION"))
            || is_word(token, "ID") && tokens.get(i + 1).is_some_and(|next| is_word(next, "DIVISION"))
        {
            let name = "IDENTIFICATION".to_owned();
            active_division = Some(name.clone());
            divisions.push(NamedSpan {
                name,
                location: token.location.clone(),
            });
        } else if is_word(token, "ENVIRONMENT")
            && tokens.get(i + 1).is_some_and(|next| is_word(next, "DIVISION"))
        {
            active_division = Some("ENVIRONMENT".to_owned());
            divisions.push(NamedSpan {
                name: "ENVIRONMENT".to_owned(),
                location: token.location.clone(),
            });
        } else if is_word(token, "DATA")
            && tokens.get(i + 1).is_some_and(|next| is_word(next, "DIVISION"))
        {
            active_division = Some("DATA".to_owned());
            divisions.push(NamedSpan {
                name: "DATA".to_owned(),
                location: token.location.clone(),
            });
        } else if is_word(token, "PROCEDURE")
            && tokens.get(i + 1).is_some_and(|next| is_word(next, "DIVISION"))
        {
            active_division = Some("PROCEDURE".to_owned());
            divisions.push(NamedSpan {
                name: "PROCEDURE".to_owned(),
                location: token.location.clone(),
            });
        }

        if is_word(token, "PROGRAM-ID") {
            if let Some(name_index) = next_non_period(&tokens, i + 1) {
                let name_token = &tokens[name_index];
                if name_token.kind == TokenKind::Word || name_token.kind == TokenKind::StringLiteral {
                    if program_id.is_some() {
                        diagnostic(
                            &mut diagnostics,
                            "duplicate_program_id",
                            "multiple PROGRAM-ID declarations were found",
                            Some(token.location.clone()),
                        );
                    } else {
                        program_id = Some(NamedSpan {
                            name: name_token.value.clone(),
                            location: name_token.location.clone(),
                        });
                    }
                } else {
                    diagnostic(
                        &mut diagnostics,
                        "malformed_program_id",
                        "PROGRAM-ID is not followed by a name token",
                        Some(token.location.clone()),
                    );
                }
            } else {
                diagnostic(
                    &mut diagnostics,
                    "malformed_program_id",
                    "PROGRAM-ID has no name token",
                    Some(token.location.clone()),
                );
            }
        }

        if is_word(token, "COPY") {
            let Some(name_index) = next_non_period(&tokens, i + 1) else {
                diagnostic(
                    &mut diagnostics,
                    "missing_copybook_name",
                    "COPY statement does not name a copybook",
                    Some(token.location.clone()),
                );
                continue;
            };
            let name_token = &tokens[name_index];
            if !matches!(name_token.kind, TokenKind::Word | TokenKind::StringLiteral) {
                diagnostic(
                    &mut diagnostics,
                    "malformed_copybook_name",
                    "COPY statement does not contain a word or string copybook name",
                    Some(name_token.location.clone()),
                );
                continue;
            }
            let end = statement_end(&tokens, name_index + 1);
            let has_replacing_clause = tokens[name_index + 1..end]
                .iter()
                .any(|candidate| is_word(candidate, "REPLACING"));
            copy_references.push(CopyReference {
                name: name_token.value.clone(),
                location: name_token.location.clone(),
                has_replacing_clause,
            });
        }

        if (is_word(token, "FD") || is_word(token, "SD")) && active_division.as_deref() == Some("DATA") {
            if let Some(name_index) = next_non_period(&tokens, i + 1) {
                if tokens[name_index].kind == TokenKind::Word {
                    file_descriptions.push(NamedSpan {
                        name: format!("{}:{}", token.value, tokens[name_index].value),
                        location: tokens[name_index].location.clone(),
                    });
                }
            }
        }

        if active_division.as_deref() == Some("DATA")
            && token.kind == TokenKind::Word
            && token.value.len() == 2
            && token.value.bytes().all(|byte| byte.is_ascii_digit())
        {
            let level = token.value.parse::<u8>().unwrap_or(0);
            if matches!(level, 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 49 | 66 | 77 | 78 | 88) {
                let Some(name_index) = next_non_period(&tokens, i + 1) else {
                    diagnostic(
                        &mut diagnostics,
                        "malformed_data_entry",
                        "data level has no following data-name token",
                        Some(token.location.clone()),
                    );
                    continue;
                };
                if tokens[name_index].kind != TokenKind::Word {
                    continue;
                }

                let end = statement_end(&tokens, name_index + 1);
                let entry_tokens = &tokens[name_index + 1..end];
                let clause_starts = [
                    "USAGE", "VALUE", "VALUES", "REDEFINES", "OCCURS", "SIGN",
                    "SYNC", "SYNCHRONIZED", "JUSTIFIED", "BLANK", "GLOBAL",
                    "EXTERNAL", "COMP", "COMP-3", "COMP-5", "BINARY", "DISPLAY",
                    "PACKED-DECIMAL", "INDEX", "POINTER",
                ];

                let mut picture = None;
                for j in 0..entry_tokens.len() {
                    if is_word(&entry_tokens[j], "PIC") || is_word(&entry_tokens[j], "PICTURE") {
                        let mut k = j + 1;
                        if k < entry_tokens.len() && is_word(&entry_tokens[k], "IS") {
                            k += 1;
                        }
                        let picture_start = k;
                        while k < entry_tokens.len()
                            && !clause_starts.iter().any(|clause| is_word(&entry_tokens[k], clause))
                        {
                            k += 1;
                        }
                        if picture_start < k {
                            picture = Some(joined_tokens(&entry_tokens[picture_start..k]));
                        }
                        break;
                    }
                }

                let usage_words = [
                    "DISPLAY", "COMP", "COMP-1", "COMP-2", "COMP-3", "COMP-4",
                    "COMP-5", "COMPUTATIONAL", "COMPUTATIONAL-3", "BINARY",
                    "PACKED-DECIMAL", "INDEX", "POINTER",
                ];
                let mut usage = None;
                for j in 0..entry_tokens.len() {
                    if is_word(&entry_tokens[j], "USAGE") {
                        if let Some(next) = entry_tokens.get(j + 1) {
                            let value = if is_word(next, "IS") {
                                entry_tokens.get(j + 2)
                            } else {
                                Some(next)
                            };
                            if let Some(value) = value {
                                usage = Some(value.value.clone());
                            }
                        }
                        break;
                    }
                    if usage_words.iter().any(|word| is_word(&entry_tokens[j], word)) {
                        usage = Some(entry_tokens[j].value.clone());
                        break;
                    }
                }

                let mut redefines = None;
                for j in 0..entry_tokens.len() {
                    if is_word(&entry_tokens[j], "REDEFINES") {
                        redefines = entry_tokens.get(j + 1).map(|value| value.value.clone());
                        break;
                    }
                }

                data_items.push(DataItem {
                    level,
                    name: tokens[name_index].value.clone(),
                    picture,
                    usage,
                    redefines,
                    location: location(
                        token.location.line,
                        token.location.column_start,
                        tokens[end.saturating_sub(1)].location.column_end,
                    ),
                });
            }
        }

        if is_word(token, "EXEC") {
            if let Some(kind_token) = tokens.get(i + 1) {
                let kind = if is_word(kind_token, "SQL") {
                    Some("SQL")
                } else if is_word(kind_token, "CICS") {
                    Some("CICS")
                } else {
                    None
                };
                if let Some(kind) = kind {
                    let end = tokens[i..]
                        .iter()
                        .position(|candidate| is_word(candidate, "END-EXEC"))
                        .map(|offset| i + offset);
                    embedded_blocks.push(EmbeddedBlock {
                        kind: kind.to_owned(),
                        location: token.location.clone(),
                        has_end_exec: end.is_some(),
                    });
                    if end.is_none() {
                        diagnostic(
                            &mut diagnostics,
                            "unterminated_embedded_block",
                            format!("EXEC {kind} has no indexed END-EXEC token"),
                            Some(token.location.clone()),
                        );
                    }
                }
            }
        }
    }

    if program_id.is_none() {
        diagnostic(
            &mut diagnostics,
            "program_id_not_found",
            "no PROGRAM-ID declaration was indexed; the source may be a fragment or use unsupported syntax",
            None,
        );
    }

    CobolIndex {
        schema_version: "vleb-cobol-index-v1".to_owned(),
        source_format: "fixed".to_owned(),
        physical_line_count,
        program_id,
        divisions,
        copy_references,
        file_descriptions,
        data_items,
        embedded_blocks,
        diagnostics,
    }
}

fn escape_json(value: &str) -> String {
    let mut output = String::new();
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c <= '\u{1f}' => {
                let _ = write!(&mut output, "\\u{:04x}", c as u32);
            }
            c => output.push(c),
        }
    }
    output
}

fn q(value: &str) -> String {
    format!("\"{}\"", escape_json(value))
}

fn loc_json(value: &SourceLocation) -> String {
    format!(
        "{{\"line\":{},\"column_start\":{},\"column_end\":{}}}",
        value.line, value.column_start, value.column_end
    )
}

fn optional_string_json(value: &Option<String>) -> String {
    match value {
        Some(value) => q(value),
        None => "null".to_owned(),
    }
}

impl CobolIndex {
    /// Deterministic JSON. Output is designed for source indexing, not an AST.
    pub fn to_json(&self) -> String {
        let mut output = String::new();
        let _ = write!(
            output,
            "{{\n  \"schema_version\": {},\n  \"source_format\": {},\n  \"physical_line_count\": {},\n",
            q(&self.schema_version),
            q(&self.source_format),
            self.physical_line_count
        );
        match &self.program_id {
            Some(program) => {
                let _ = write!(
                    output,
                    "  \"program_id\": {{\"name\":{},\"location\":{}}},\n",
                    q(&program.name),
                    loc_json(&program.location)
                );
            }
            None => output.push_str("  \"program_id\": null,\n"),
        }

        output.push_str("  \"divisions\": [");
        for (i, item) in self.divisions.iter().enumerate() {
            if i > 0 { output.push(','); }
            let _ = write!(output, "{{\"name\":{},\"location\":{}}}", q(&item.name), loc_json(&item.location));
        }
        output.push_str("],\n  \"copy_references\": [");
        for (i, item) in self.copy_references.iter().enumerate() {
            if i > 0 { output.push(','); }
            let _ = write!(
                output,
                "{{\"name\":{},\"location\":{},\"has_replacing_clause\":{}}}",
                q(&item.name), loc_json(&item.location), item.has_replacing_clause
            );
        }
        output.push_str("],\n  \"file_descriptions\": [");
        for (i, item) in self.file_descriptions.iter().enumerate() {
            if i > 0 { output.push(','); }
            let _ = write!(output, "{{\"name\":{},\"location\":{}}}", q(&item.name), loc_json(&item.location));
        }
        output.push_str("],\n  \"data_items\": [");
        for (i, item) in self.data_items.iter().enumerate() {
            if i > 0 { output.push(','); }
            let _ = write!(
                output,
                "{{\"level\":{},\"name\":{},\"picture\":{},\"usage\":{},\"redefines\":{},\"location\":{}}}",
                item.level, q(&item.name), optional_string_json(&item.picture),
                optional_string_json(&item.usage), optional_string_json(&item.redefines),
                loc_json(&item.location)
            );
        }
        output.push_str("],\n  \"embedded_blocks\": [");
        for (i, item) in self.embedded_blocks.iter().enumerate() {
            if i > 0 { output.push(','); }
            let _ = write!(
                output,
                "{{\"kind\":{},\"location\":{},\"has_end_exec\":{}}}",
                q(&item.kind), loc_json(&item.location), item.has_end_exec
            );
        }
        output.push_str("],\n  \"diagnostics\": [");
        for (i, item) in self.diagnostics.iter().enumerate() {
            if i > 0 { output.push(','); }
            let _ = write!(
                output,
                "{{\"code\":{},\"message\":{},\"location\":{}}}",
                q(&item.code), q(&item.message),
                item.location.as_ref().map(loc_json).unwrap_or_else(|| "null".to_owned())
            );
        }
        output.push_str("]\n}\n");
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(source: &str) -> CobolIndex {
        index_fixed_format(source, IndexOptions::default())
    }

    #[test]
    fn locates_program_divisions_and_fixed_column_spans() {
        let source = concat!(
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. SETTLE01.\n",
            "       DATA DIVISION.\n",
            "       PROCEDURE DIVISION.\n",
            "           STOP RUN.\n"
        );
        let result = index(source);
        assert_eq!(result.program_id.as_ref().map(|p| p.name.as_str()), Some("SETTLE01"));
        assert_eq!(result.program_id.as_ref().unwrap().location.line, 2);
        assert_eq!(result.program_id.as_ref().unwrap().location.column_start, 20);
        assert_eq!(
            result.divisions.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            vec!["IDENTIFICATION", "DATA", "PROCEDURE"]
        );
    }

    #[test]
    fn ignores_comments_and_identification_area_after_column_72() {
        let source = concat!(
            "      * PROGRAM-ID. FALSEID.\n",
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. REALID.                                                FAKE-TEXT\n",
        );
        let result = index(source);
        assert_eq!(result.program_id.as_ref().map(|p| p.name.as_str()), Some("REALID"));
        assert!(!result.diagnostics.iter().any(|d| d.code == "duplicate_program_id"));
    }

    #[test]
    fn captures_copybook_and_replacing_clause() {
        let source = concat!(
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. COPYTEST.\n",
            "       DATA DIVISION.\n",
            "       COPY AMOUNT-REC REPLACING ==:TAG:== BY ==WS==.\n",
        );
        let result = index(source);
        assert_eq!(result.copy_references.len(), 1);
        assert_eq!(result.copy_references[0].name, "AMOUNT-REC");
        assert!(result.copy_references[0].has_replacing_clause);
    }

    #[test]
    fn extracts_packed_decimal_and_redefines_without_claiming_semantics() {
        let source = concat!(
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. DECIMAL1.\n",
            "       DATA DIVISION.\n",
            "       WORKING-STORAGE SECTION.\n",
            "       01 WS-AMOUNT PIC S9(7)V99 COMP-3.\n",
            "       01 WS-RAW REDEFINES WS-AMOUNT PIC X(5).\n",
        );
        let result = index(source);
        let packed = result.data_items.iter().find(|item| item.name == "WS-AMOUNT").unwrap();
        assert_eq!(packed.picture.as_deref(), Some("S9(7)V99"));
        assert_eq!(packed.usage.as_deref(), Some("COMP-3"));
        let redef = result.data_items.iter().find(|item| item.name == "WS-RAW").unwrap();
        assert_eq!(redef.redefines.as_deref(), Some("WS-AMOUNT"));
    }

    #[test]
    fn locates_fd_and_embedded_sql_cics_anchors() {
        let source = concat!(
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. FILESQL.\n",
            "       DATA DIVISION.\n",
            "       FD SETTLEMENT-FILE.\n",
            "       01 SETTLEMENT-REC PIC X(80).\n",
            "       PROCEDURE DIVISION.\n",
            "           EXEC SQL SELECT COUNT(*) INTO :WS-COUNT FROM ACCTS END-EXEC.\n",
            "           EXEC CICS RETURN END-EXEC.\n",
        );
        let result = index(source);
        assert_eq!(result.file_descriptions.len(), 1);
        assert_eq!(result.file_descriptions[0].name, "FD:SETTLEMENT-FILE");
        assert_eq!(result.embedded_blocks.iter().map(|block| block.kind.as_str()).collect::<Vec<_>>(), vec!["SQL", "CICS"]);
        assert!(result.embedded_blocks.iter().all(|block| block.has_end_exec));
    }

    #[test]
    fn fails_closed_on_non_ascii_tabs_unknown_indicator_and_opaque_continuation() {
        let source = concat!(
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. SAFE.\n",
            "       DISPLAY \"THIS STRING CONTINUES\n",
            "      -COPY NOT-A-COPY.\n",
            "      ?PROGRAM-ID. BAD.\n",
            "       \tCOPY TABBED.\n",
            "       DISPLAY CAFÉ.\n",
        );
        let result = index(source);
        let codes: Vec<&str> = result.diagnostics.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"unterminated_literal"));
        assert!(codes.contains(&"continued_literal_opaque"));
        assert!(codes.contains(&"unknown_fixed_format_indicator"));
        assert!(codes.contains(&"tab_in_fixed_format_record"));
        assert!(codes.contains(&"non_ascii_fixed_format_record"));
        assert!(!result.copy_references.iter().any(|copy| copy.name == "NOT-A-COPY"));
    }

    #[test]
    fn literal_text_does_not_create_copy_references_or_divisions() {
        let source = concat!(
            "       IDENTIFICATION DIVISION.\n",
            "       PROGRAM-ID. LITTEST.\n",
            "       PROCEDURE DIVISION.\n",
            "           DISPLAY \"COPY DANGEROUS AND DATA DIVISION.\".\n",
            "           STOP RUN.\n",
        );
        let result = index(source);
        assert!(result.copy_references.is_empty());
        assert_eq!(result.divisions.len(), 2);
    }

    #[test]
    fn deterministic_json_has_valid_escaping_and_claim_scope() {
        let result = index("       IDENTIFICATION DIVISION.\n       PROGRAM-ID. JSONTEST.\n");
        let json = result.to_json();
        assert!(json.starts_with("{\n  \"schema_version\": \"vleb-cobol-index-v1\""));
        assert!(json.contains("\"source_format\": \"fixed\""));
        assert!(json.contains("\"program_id\": {\"name\":\"JSONTEST\""));
        assert_eq!(json, index_fixed_format(
            "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. JSONTEST.\n",
            IndexOptions::default()
        ).to_json());
    }
}
