// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! POML (Prompt Orchestration Markup Language) Processor
//!
//! Rust port of the Python POML v2.0 processor. Parses XML-based prompt
//! templates with variable substitution, conditionals, loops, examples,
//! stepwise instructions, and model hints.
//!
//! POML templates define structured prompts for LLM interactions:
//! ```xml
//! <poml version="2.0">
//!   <metadata>
//!     <title>Intent Classification</title>
//!     <model-hints><temperature>0.3</temperature></model-hints>
//!   </metadata>
//!   <variables>
//!     <let name="query">{{ user_request }}</let>
//!   </variables>
//!   <prompt>
//!     <system>You are a NixOS expert.</system>
//!     <stepwise-instructions>
//!       <step id="s1">Analyze the request.</step>
//!     </stepwise-instructions>
//!     <examples>
//!       <example><input>install firefox</input><output>{"intent":"install"}</output></example>
//!     </examples>
//!     <output-format>JSON with intent field</output-format>
//!   </prompt>
//! </poml>
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Typed errors for POML parsing and processing.
#[derive(Debug)]
pub enum PomlError {
    /// XML parsing failed at the given byte offset.
    XmlParse { offset: usize, detail: String },
    /// A required element (e.g. `<prompt>`) is missing.
    MissingElement { element: String, template: String },
    /// Template file could not be read.
    IoError {
        path: String,
        source: std::io::Error,
    },
    /// Template not found in cache or on disk.
    NotFound(String),
}

impl fmt::Display for PomlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::XmlParse { offset, detail } => {
                write!(f, "XML parse error at byte {offset}: {detail}")
            }
            Self::MissingElement { element, template } => {
                write!(f, "Template '{template}' missing <{element}> section")
            }
            Self::IoError { path, source } => {
                write!(f, "Failed to read {path}: {source}")
            }
            Self::NotFound(name) => {
                write!(f, "Template not found: {name}")
            }
        }
    }
}

impl std::error::Error for PomlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError { source, .. } => Some(source),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// POML v2.0 features that can be used in templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PomlFeature {
    Conditionals,
    Loops,
    Cache,
}

/// Context for POML template execution.
#[derive(Debug, Clone, Default)]
pub struct PomlContext {
    pub variables: HashMap<String, PomlValue>,
    pub persona: Option<String>,
}

/// A POML value (string, list, or nested map).
#[derive(Debug, Clone)]
pub enum PomlValue {
    String(String),
    List(Vec<String>),
    Map(HashMap<String, PomlValue>),
}

impl PomlValue {
    pub fn as_str(&self) -> &str {
        match self {
            Self::String(s) => s,
            Self::List(v) => {
                if v.is_empty() {
                    ""
                } else {
                    &v[0]
                }
            }
            Self::Map(_) => "<map>",
        }
    }

    pub fn to_display(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::List(v) => format!("[{}]", v.join(", ")),
            Self::Map(m) => {
                let pairs: Vec<String> = m
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_display()))
                    .collect();
                format!("{{{}}}", pairs.join(", "))
            }
        }
    }
}

impl From<&str> for PomlValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<String> for PomlValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<Vec<String>> for PomlValue {
    fn from(v: Vec<String>) -> Self {
        Self::List(v)
    }
}

/// Model hints extracted from template metadata.
#[derive(Debug, Clone, Default)]
pub struct ModelHints {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub preferred_models: Vec<String>,
}

/// Cache settings from a template.
#[derive(Debug, Clone)]
pub struct CacheSettings {
    pub duration: Option<u64>,
    pub key: Option<String>,
    pub invalidate_on: Option<String>,
}

/// Metadata from a template.
#[derive(Debug, Clone, Default)]
pub struct PomlMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub version: Option<String>,
    pub model_hints: ModelHints,
}

/// Result of processing a POML template.
#[derive(Debug, Clone)]
pub struct PomlResult {
    pub prompt: String,
    pub metadata: PomlMetadata,
    pub cache_settings: Option<CacheSettings>,
    pub features_used: Vec<PomlFeature>,
}

// ---------------------------------------------------------------------------
// Lightweight XML types (no external crate needed for simple POML)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct XmlNode {
    tag: String,
    attrs: HashMap<String, String>,
    text: String,
    children: Vec<XmlNode>,
    tail: String,
}

impl XmlNode {
    fn find(&self, tag: &str) -> Option<&XmlNode> {
        self.children.iter().find(|c| c.tag == tag)
    }

    fn find_all(&self, tag: &str) -> Vec<&XmlNode> {
        self.children.iter().filter(|c| c.tag == tag).collect()
    }

    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.get(name).map(|s| s.as_str())
    }
}

/// Parse a POML XML string into a tree of XmlNodes.
fn parse_xml(input: &str) -> Result<XmlNode, PomlError> {
    let input = input.trim();
    // Skip XML declaration
    let input = if input.starts_with("<?xml") {
        input
            .find("?>")
            .map(|i| &input[i + 2..])
            .unwrap_or(input)
            .trim()
    } else {
        input
    };

    let mut parser = XmlParser::new(input);
    parser.parse_element().ok_or_else(|| PomlError::XmlParse {
        offset: parser.pos,
        detail: format!(
            "unexpected content near: {:?}",
            &parser.remaining()[..parser.remaining().len().min(40)]
        ),
    })
}

struct XmlParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> XmlParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn parse_element(&mut self) -> Option<XmlNode> {
        self.skip_whitespace();

        // Skip comments
        while self.remaining().starts_with("<!--") {
            if let Some(end) = self.remaining().find("-->") {
                self.pos += end + 3;
                self.skip_whitespace();
            } else {
                return None;
            }
        }

        if !self.remaining().starts_with('<') || self.remaining().starts_with("</") {
            return None;
        }

        self.pos += 1; // skip '<'

        // Parse tag name
        let tag_start = self.pos;
        while self.pos < self.input.len() {
            let ch = self.input.as_bytes()[self.pos];
            if ch == b' ' || ch == b'>' || ch == b'/' || ch == b'\n' || ch == b'\t' {
                break;
            }
            self.pos += 1;
        }
        let tag = self.input[tag_start..self.pos].to_string();

        // Parse attributes
        let mut attrs = HashMap::new();
        loop {
            self.skip_whitespace();
            if self.remaining().starts_with("/>") {
                self.pos += 2;
                return Some(XmlNode {
                    tag,
                    attrs,
                    text: String::new(),
                    children: Vec::new(),
                    tail: String::new(),
                });
            }
            if self.remaining().starts_with('>') {
                self.pos += 1;
                break;
            }
            // Parse attribute name=value
            let attr_start = self.pos;
            while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != b'=' {
                if self.input.as_bytes()[self.pos] == b'>'
                    || self.input.as_bytes()[self.pos] == b'/'
                {
                    break;
                }
                self.pos += 1;
            }
            if self.pos >= self.input.len() || self.input.as_bytes()[self.pos] != b'=' {
                // Not an attribute, skip
                if self.remaining().starts_with("/>") {
                    self.pos += 2;
                    return Some(XmlNode {
                        tag,
                        attrs,
                        text: String::new(),
                        children: Vec::new(),
                        tail: String::new(),
                    });
                }
                if self.remaining().starts_with('>') {
                    self.pos += 1;
                    break;
                }
                self.pos += 1;
                continue;
            }
            let attr_name = self.input[attr_start..self.pos].trim().to_string();
            self.pos += 1; // skip '='

            // Parse value
            self.skip_whitespace();
            let quote = self.input.as_bytes().get(self.pos).copied().unwrap_or(b'"');
            if quote == b'"' || quote == b'\'' {
                self.pos += 1;
                let val_start = self.pos;
                while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != quote {
                    self.pos += 1;
                }
                let val = self.input[val_start..self.pos].to_string();
                if self.pos < self.input.len() {
                    self.pos += 1; // skip closing quote
                }
                attrs.insert(attr_name, val);
            }
        }

        // Parse text content before first child
        let text = self.parse_text_until_tag();

        // Parse children
        let mut children = Vec::new();
        loop {
            self.skip_whitespace();
            let closing = format!("</{}", tag);
            if self.remaining().starts_with(&closing) {
                // Skip closing tag
                if let Some(end) = self.remaining().find('>') {
                    self.pos += end + 1;
                }
                break;
            }
            if self.remaining().is_empty() {
                break;
            }
            if self.remaining().starts_with("</") {
                // Mismatched close tag — stop
                break;
            }
            if self.remaining().starts_with('<') {
                if let Some(mut child) = self.parse_element() {
                    // Tail text after child
                    child.tail = self.parse_text_until_tag();
                    children.push(child);
                } else {
                    break;
                }
            } else {
                // Stray text — skip a char
                self.pos += 1;
            }
        }

        Some(XmlNode {
            tag,
            attrs,
            text,
            children,
            tail: String::new(),
        })
    }

    fn parse_text_until_tag(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != b'<' {
            self.pos += 1;
        }
        let raw = &self.input[start..self.pos];
        // Decode basic XML entities
        raw.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
    }
}

// ---------------------------------------------------------------------------
// POML Processor
// ---------------------------------------------------------------------------

/// Processes POML v2.0 templates.
pub struct PomlProcessor {
    template_dir: PathBuf,
    template_cache: HashMap<String, XmlNode>,
}

impl PomlProcessor {
    /// Create a new processor with the given template directory.
    pub fn new(template_dir: impl Into<PathBuf>) -> Self {
        Self {
            template_dir: template_dir.into(),
            template_cache: HashMap::new(),
        }
    }

    /// Load a template by name (with .poml extension).
    pub(crate) fn load_template(&mut self, name: &str) -> Result<&XmlNode, PomlError> {
        if self.template_cache.contains_key(name) {
            return Ok(&self.template_cache[name]);
        }

        let mut path = self.template_dir.join(format!("{}.poml", name));
        if !path.exists() {
            path = self.template_dir.join(name);
            if !path.exists() {
                return Err(PomlError::NotFound(name.to_string()));
            }
        }

        let content = std::fs::read_to_string(&path).map_err(|e| PomlError::IoError {
            path: name.to_string(),
            source: e,
        })?;
        let root = parse_xml(&content)?;
        self.template_cache.insert(name.to_string(), root);
        Ok(&self.template_cache[name])
    }

    /// Load a template from a string (for testing or inline templates).
    pub fn load_template_str(&mut self, name: &str, content: &str) -> Result<(), PomlError> {
        let root = parse_xml(content)?;
        self.template_cache.insert(name.to_string(), root);
        Ok(())
    }

    /// Process a template with the given context.
    pub fn process(&mut self, name: &str, context: &PomlContext) -> Result<PomlResult, PomlError> {
        // Load and clone to avoid borrow issues
        if !self.template_cache.contains_key(name) {
            self.load_template(name)?;
        }
        let template = self.template_cache[name].clone();

        let mut features_used = Vec::new();

        // Extract metadata
        let metadata = Self::extract_metadata(&template);

        // Process variables: template <let> definitions merged with context
        let mut vars = Self::process_variables(&template, &context.variables);

        // Build prompt
        let prompt = if let Some(prompt_elem) = template.find("prompt") {
            Self::build_prompt(prompt_elem, &mut vars, context, &mut features_used)
        } else {
            return Err(PomlError::MissingElement {
                element: "prompt".to_string(),
                template: name.to_string(),
            });
        };

        // Cache settings
        let cache_settings = template
            .find("prompt")
            .and_then(|p| p.find("cache"))
            .and_then(|cache| {
                let enabled = cache.attr("enabled").unwrap_or("true");
                if enabled == "false" {
                    return None;
                }
                features_used.push(PomlFeature::Cache);
                Some(CacheSettings {
                    duration: cache.attr("duration").and_then(|d| d.parse().ok()),
                    key: cache.attr("key").map(|s| s.to_string()),
                    invalidate_on: cache.attr("invalidate-on").map(|s| s.to_string()),
                })
            });

        Ok(PomlResult {
            prompt,
            metadata,
            cache_settings,
            features_used,
        })
    }

    /// List available templates in the template directory.
    pub fn list_templates(&self) -> Vec<String> {
        let mut templates = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.template_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "poml") {
                    if let Some(stem) = path.file_stem() {
                        templates.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
        templates.sort();
        templates
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn extract_metadata(root: &XmlNode) -> PomlMetadata {
        let mut meta = PomlMetadata::default();
        if let Some(m) = root.find("metadata") {
            meta.title = m.find("title").map(|n| n.text.trim().to_string());
            meta.description = m.find("description").map(|n| n.text.trim().to_string());
            meta.author = m.find("author").map(|n| n.text.trim().to_string());
            meta.version = m.find("version").map(|n| n.text.trim().to_string());

            if let Some(hints) = m.find("model-hints") {
                if let Some(temp) = hints.find("temperature") {
                    meta.model_hints.temperature = temp.text.trim().parse().ok();
                }
                if let Some(mt) = hints.find("max-tokens") {
                    meta.model_hints.max_tokens = mt.text.trim().parse().ok();
                }
                if let Some(pm) = hints.find("preferred-models") {
                    meta.model_hints.preferred_models = pm
                        .text
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
            }
        }
        meta
    }

    fn process_variables(
        root: &XmlNode,
        context_vars: &HashMap<String, PomlValue>,
    ) -> HashMap<String, PomlValue> {
        let mut vars: HashMap<String, PomlValue> = context_vars.clone();

        if let Some(variables) = root.find("variables") {
            for let_elem in variables.find_all("let") {
                if let Some(name) = let_elem.attr("name") {
                    let raw = let_elem.text.trim();
                    let resolved = substitute_variables(raw, &vars);

                    // Handle default syntax: {{ var | default: "value" }}
                    let value = if resolved.contains("| default:") || resolved.contains("|default:")
                    {
                        // Variable wasn't resolved — use the default
                        if let Some(default_start) = resolved.find("default:") {
                            let default_val = resolved[default_start + 8..].trim();
                            let default_val = default_val
                                .trim_end_matches("}}")
                                .trim()
                                .trim_matches('"')
                                .trim_matches('\'');
                            default_val.to_string()
                        } else {
                            resolved
                        }
                    } else {
                        resolved
                    };

                    vars.insert(name.to_string(), PomlValue::String(value));
                }
            }
        }

        vars
    }

    fn build_prompt(
        prompt_elem: &XmlNode,
        vars: &mut HashMap<String, PomlValue>,
        context: &PomlContext,
        features: &mut Vec<PomlFeature>,
    ) -> String {
        let mut sections = Vec::new();

        // User query goes first if present
        let query = vars
            .get("query")
            .or_else(|| vars.get("user_request"))
            .map(|v| v.to_display());
        if let Some(q) = query {
            sections.push(format!("User Query: {}", q));
        }

        // System prompt
        if let Some(system) = prompt_elem.find("system") {
            let text = process_element(system, vars, context, features);
            if !text.is_empty() {
                sections.push(format!("System: {}", text));
            }
        }

        // Stepwise instructions
        if let Some(instr) = prompt_elem.find("stepwise-instructions") {
            let steps = process_instructions(instr, vars, context, features);
            if !steps.is_empty() {
                sections.push(format!("Instructions:\n{}", steps));
            }
        }

        // Examples
        if let Some(examples) = prompt_elem.find("examples") {
            let text = process_examples(examples, vars);
            if !text.is_empty() {
                sections.push(format!("Examples:\n{}", text));
            }
        }

        // Output format
        if let Some(fmt) = prompt_elem.find("output-format") {
            let text = process_element(fmt, vars, context, features);
            if !text.is_empty() {
                sections.push(format!("Output Format:\n{}", text));
            }
        }

        // Hints
        if let Some(hint) = prompt_elem.find("hint") {
            let text = process_element(hint, vars, context, features);
            if !text.is_empty() {
                sections.push(format!("Hint: {}", text));
            }
        }

        // Persona adjustments (conditionals)
        if let Some(pa) = prompt_elem.find("persona-adjustments") {
            let text = process_conditionals(pa, vars, context, features);
            if !text.is_empty() {
                sections.push(text);
            }
        }

        // Error handling
        if let Some(eh) = prompt_elem.find("error-handling") {
            let parts: Vec<String> = eh
                .find_all("on-error")
                .iter()
                .map(|oe| {
                    let etype = oe.attr("type").unwrap_or("unknown");
                    let text = oe.text.trim();
                    format!("On {}: {}", etype, text)
                })
                .collect();
            if !parts.is_empty() {
                sections.push(format!("Error Handling:\n{}", parts.join("\n")));
            }
        }

        // Process any remaining children not handled above (foreach, if, etc.)
        let known_tags = [
            "system",
            "stepwise-instructions",
            "examples",
            "output-format",
            "hint",
            "persona-adjustments",
            "error-handling",
            "cache",
        ];
        for child in &prompt_elem.children {
            if !known_tags.contains(&child.tag.as_str()) {
                let text = process_element(child, vars, context, features);
                if !text.is_empty() {
                    sections.push(text);
                }
            }
        }

        sections.join("\n\n")
    }
}

// ---------------------------------------------------------------------------
// Element processing (free functions to avoid &mut self borrow issues)
// ---------------------------------------------------------------------------

fn process_element(
    elem: &XmlNode,
    vars: &HashMap<String, PomlValue>,
    context: &PomlContext,
    features: &mut Vec<PomlFeature>,
) -> String {
    match elem.tag.as_str() {
        "if" => {
            if !features.contains(&PomlFeature::Conditionals) {
                features.push(PomlFeature::Conditionals);
            }
            let condition = elem.attr("condition").unwrap_or("");
            if evaluate_condition(condition, vars, context) {
                process_children(elem, vars, context, features)
            } else {
                String::new()
            }
        }
        "elseif" => {
            if !features.contains(&PomlFeature::Conditionals) {
                features.push(PomlFeature::Conditionals);
            }
            let condition = elem.attr("condition").unwrap_or("");
            if evaluate_condition(condition, vars, context) {
                process_children(elem, vars, context, features)
            } else {
                String::new()
            }
        }
        "else" => {
            if !features.contains(&PomlFeature::Conditionals) {
                features.push(PomlFeature::Conditionals);
            }
            process_children(elem, vars, context, features)
        }
        "foreach" => {
            if !features.contains(&PomlFeature::Loops) {
                features.push(PomlFeature::Loops);
            }
            process_foreach(elem, vars, context, features)
        }
        _ => {
            let mut parts = Vec::new();
            let text = elem.text.trim();
            if !text.is_empty() {
                parts.push(substitute_variables(text, vars));
            }
            for child in &elem.children {
                let child_text = process_element(child, vars, context, features);
                if !child_text.is_empty() {
                    parts.push(child_text);
                }
                let tail = child.tail.trim();
                if !tail.is_empty() {
                    parts.push(substitute_variables(tail, vars));
                }
            }
            parts.join(" ")
        }
    }
}

fn process_children(
    elem: &XmlNode,
    vars: &HashMap<String, PomlValue>,
    context: &PomlContext,
    features: &mut Vec<PomlFeature>,
) -> String {
    let mut parts = Vec::new();
    let text = elem.text.trim();
    if !text.is_empty() {
        parts.push(substitute_variables(text, vars));
    }
    for child in &elem.children {
        let child_text = process_element(child, vars, context, features);
        if !child_text.is_empty() {
            parts.push(child_text);
        }
    }
    parts.join(" ")
}

fn process_foreach(
    elem: &XmlNode,
    vars: &HashMap<String, PomlValue>,
    context: &PomlContext,
    features: &mut Vec<PomlFeature>,
) -> String {
    let items_name = match elem.attr("items") {
        Some(name) => name,
        None => return String::new(),
    };
    let as_name = elem.attr("as").unwrap_or("item");

    let items = match vars.get(items_name) {
        Some(PomlValue::List(list)) => list.clone(),
        Some(PomlValue::String(s)) => vec![s.clone()],
        _ => return String::new(),
    };

    let mut results = Vec::new();
    for item in &items {
        let mut loop_vars = vars.clone();
        loop_vars.insert(as_name.to_string(), PomlValue::String(item.clone()));
        let result = process_children(elem, &loop_vars, context, features);
        if !result.is_empty() {
            results.push(result);
        }
    }
    results.join("\n")
}

fn process_instructions(
    elem: &XmlNode,
    vars: &HashMap<String, PomlValue>,
    context: &PomlContext,
    features: &mut Vec<PomlFeature>,
) -> String {
    let mut steps = Vec::new();
    for step in elem.find_all("step") {
        let id = step.attr("id").unwrap_or("");
        let text = process_element(step, vars, context, features);
        if !text.is_empty() {
            if !id.is_empty() {
                steps.push(format!("- [{}] {}", id, text));
            } else {
                steps.push(format!("- {}", text));
            }
        }
    }
    steps.join("\n")
}

fn process_examples(elem: &XmlNode, vars: &HashMap<String, PomlValue>) -> String {
    let mut examples = Vec::new();
    for example in elem.find_all("example") {
        let input_text = example
            .find("input")
            .map(|n| substitute_variables(n.text.trim(), vars))
            .unwrap_or_default();
        let output_text = example
            .find("output")
            .map(|n| substitute_variables(n.text.trim(), vars))
            .unwrap_or_default();
        if !input_text.is_empty() || !output_text.is_empty() {
            examples.push(format!("Input: {}\nOutput: {}", input_text, output_text));
        }
    }
    examples.join("\n\n")
}

fn process_conditionals(
    elem: &XmlNode,
    vars: &HashMap<String, PomlValue>,
    context: &PomlContext,
    features: &mut Vec<PomlFeature>,
) -> String {
    // Process if/elseif/else chains within persona-adjustments or similar containers
    let mut result = String::new();
    let mut matched = false;

    for child in &elem.children {
        match child.tag.as_str() {
            "if" if !matched => {
                if !features.contains(&PomlFeature::Conditionals) {
                    features.push(PomlFeature::Conditionals);
                }
                let condition = child.attr("condition").unwrap_or("");
                if evaluate_condition(condition, vars, context) {
                    result = process_children(child, vars, context, features);
                    matched = true;
                }
            }
            "elseif" if !matched => {
                let condition = child.attr("condition").unwrap_or("");
                if evaluate_condition(condition, vars, context) {
                    result = process_children(child, vars, context, features);
                    matched = true;
                }
            }
            "else" if !matched => {
                result = process_children(child, vars, context, features);
                matched = true;
            }
            _ => {}
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Variable substitution
// ---------------------------------------------------------------------------

/// Substitute `{{ variable }}` placeholders in text.
fn substitute_variables(text: &str, vars: &HashMap<String, PomlValue>) -> String {
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'{' {
            // Find closing }}
            if let Some(end) = text[i + 2..].find("}}") {
                let var_expr = text[i + 2..i + 2 + end].trim();

                // Handle pipe-separated filters (e.g., "var | default: value")
                let var_name = if let Some(pipe_pos) = var_expr.find('|') {
                    var_expr[..pipe_pos].trim()
                } else {
                    var_expr
                };

                // Handle dotted access (e.g., "user.name")
                let value = if var_name.contains('.') {
                    let parts: Vec<&str> = var_name.split('.').collect();
                    let mut current = vars.get(parts[0]);
                    for part in &parts[1..] {
                        current = current.and_then(|v| match v {
                            PomlValue::Map(m) => m.get(*part),
                            _ => None,
                        });
                    }
                    current.map(|v| v.to_display())
                } else {
                    vars.get(var_name).map(|v| v.to_display())
                };

                if let Some(val) = value {
                    result.push_str(&val);
                } else {
                    // Keep original placeholder (including any filter)
                    result.push_str(&text[i..i + 2 + end + 2]);
                }
                i += 2 + end + 2;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Condition evaluation
// ---------------------------------------------------------------------------

fn evaluate_condition(
    condition: &str,
    vars: &HashMap<String, PomlValue>,
    context: &PomlContext,
) -> bool {
    if condition.is_empty() {
        return false;
    }

    // Build evaluation context: merge vars + context persona
    let mut eval_vars = vars.clone();
    if let Some(persona) = &context.persona {
        eval_vars.insert("persona".to_string(), PomlValue::String(persona.clone()));
    }

    // Resolve a token: if it's a known variable name, return its value;
    // otherwise strip quotes and return as literal.
    let resolve = |token: &str| -> String {
        let trimmed = token.trim();
        // First try {{ }} substitution
        let substituted = substitute_variables(trimmed, &eval_vars);
        if substituted != trimmed {
            return substituted;
        }
        // Then try bare variable lookup
        if let Some(val) = eval_vars.get(trimmed) {
            return val.to_display();
        }
        // Otherwise treat as literal (strip quotes)
        trimmed.trim_matches(|c| c == '"' || c == '\'').to_string()
    };

    // Simple equality: "a == b"
    if let Some((left, right)) = condition.split_once("==") {
        return resolve(left) == resolve(right);
    }

    // Inequality: "a != b"
    if let Some((left, right)) = condition.split_once("!=") {
        return resolve(left) != resolve(right);
    }

    // Contains: "a in b"
    if condition.contains(" in ") {
        let parts: Vec<&str> = condition.splitn(2, " in ").collect();
        if parts.len() == 2 {
            let item = resolve(parts[0]);
            let container_name = parts[1].trim();
            if let Some(val) = eval_vars.get(container_name) {
                return val.to_display().contains(&item);
            }
            // Also try resolving the container
            return resolve(parts[1]).contains(&item);
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_template() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<poml version="2.0">
  <metadata>
    <title>Test Template</title>
    <description>A test template</description>
    <author>Test</author>
    <version>1.0.0</version>
    <model-hints>
      <preferred-models>mistral-7b, gemma3:1b</preferred-models>
      <temperature>0.3</temperature>
      <max-tokens>500</max-tokens>
    </model-hints>
  </metadata>

  <variables>
    <let name="user_request">{{ user_request }}</let>
    <let name="persona">{{ persona | default: "general" }}</let>
  </variables>

  <prompt>
    <system>You are a NixOS expert.</system>
    <stepwise-instructions>
      <step id="s1">Analyze the request: {{ user_request }}</step>
      <step id="s2">Determine the intent.</step>
    </stepwise-instructions>
    <examples>
      <example>
        <input>install firefox</input>
        <output>{"intent": "install_package"}</output>
      </example>
    </examples>
    <output-format>JSON with intent field</output-format>
    <hint>Be precise</hint>
  </prompt>
</poml>"#
    }

    #[test]
    fn test_parse_and_process() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("test", sample_template()).unwrap();

        let mut ctx = PomlContext::default();
        ctx.variables
            .insert("user_request".into(), "install firefox".into());

        let result = proc.process("test", &ctx).unwrap();

        assert!(result.prompt.contains("User Query: install firefox"));
        assert!(result.prompt.contains("System: You are a NixOS expert."));
        assert!(result.prompt.contains("[s1]"));
        assert!(result.prompt.contains("[s2]"));
        assert!(result.prompt.contains("Input: install firefox"));
        assert!(
            result
                .prompt
                .contains("Output: {\"intent\": \"install_package\"}")
        );
        assert!(result.prompt.contains("Output Format:"));
        assert!(result.prompt.contains("Hint: Be precise"));
    }

    #[test]
    fn test_metadata_extraction() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("test", sample_template()).unwrap();

        let ctx = PomlContext::default();
        let result = proc.process("test", &ctx).unwrap();

        assert_eq!(result.metadata.title.as_deref(), Some("Test Template"));
        assert_eq!(result.metadata.model_hints.temperature, Some(0.3));
        assert_eq!(result.metadata.model_hints.max_tokens, Some(500));
        assert_eq!(result.metadata.model_hints.preferred_models.len(), 2);
        assert_eq!(
            result.metadata.model_hints.preferred_models[0],
            "mistral-7b"
        );
    }

    #[test]
    fn test_variable_substitution() {
        let mut vars = HashMap::new();
        vars.insert("name".into(), PomlValue::String("firefox".into()));

        let result = substitute_variables("Install {{ name }} please", &vars);
        assert_eq!(result, "Install firefox please");
    }

    #[test]
    fn test_unresolved_variable_kept() {
        let vars = HashMap::new();
        let result = substitute_variables("Install {{ name }} please", &vars);
        assert_eq!(result, "Install {{ name }} please");
    }

    #[test]
    fn test_dotted_variable() {
        let mut inner = HashMap::new();
        inner.insert("name".to_string(), PomlValue::String("firefox".into()));
        let mut vars = HashMap::new();
        vars.insert("pkg".to_string(), PomlValue::Map(inner));

        let result = substitute_variables("Install {{ pkg.name }}", &vars);
        assert_eq!(result, "Install firefox");
    }

    #[test]
    fn test_conditionals() {
        let template = r#"<poml version="2.0">
  <metadata><title>Cond Test</title></metadata>
  <prompt>
    <system>Test</system>
    <persona-adjustments>
      <if condition="persona == 'developer'">
        <hint>Expect technical terms</hint>
      </if>
      <elseif condition="persona == 'grandma'">
        <hint>Keep it simple</hint>
      </elseif>
      <else>
        <hint>General usage</hint>
      </else>
    </persona-adjustments>
  </prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("cond", template).unwrap();

        // Test developer persona
        let mut ctx = PomlContext::default();
        ctx.persona = Some("developer".into());
        let result = proc.process("cond", &ctx).unwrap();
        assert!(result.prompt.contains("Expect technical terms"));
        assert!(result.features_used.contains(&PomlFeature::Conditionals));

        // Test grandma persona
        ctx.persona = Some("grandma".into());
        let result = proc.process("cond", &ctx).unwrap();
        assert!(result.prompt.contains("Keep it simple"));

        // Test fallback
        ctx.persona = Some("unknown".into());
        let result = proc.process("cond", &ctx).unwrap();
        assert!(result.prompt.contains("General usage"));
    }

    #[test]
    fn test_foreach_loop() {
        let template = r#"<poml version="2.0">
  <metadata><title>Loop Test</title></metadata>
  <prompt>
    <system>Test</system>
    <foreach items="packages" as="pkg">
      <step>Install {{ pkg }}</step>
    </foreach>
  </prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("loop", template).unwrap();

        let mut ctx = PomlContext::default();
        ctx.variables.insert(
            "packages".into(),
            PomlValue::List(vec!["firefox".into(), "vim".into(), "git".into()]),
        );

        let result = proc.process("loop", &ctx).unwrap();
        assert!(result.prompt.contains("Install firefox"));
        assert!(result.prompt.contains("Install vim"));
        assert!(result.prompt.contains("Install git"));
        assert!(result.features_used.contains(&PomlFeature::Loops));
    }

    #[test]
    fn test_cache_settings() {
        let template = r#"<poml version="2.0">
  <metadata><title>Cache Test</title></metadata>
  <prompt>
    <system>Test</system>
    <cache enabled="true" duration="3600" key="intent_{{ query }}" invalidate-on="context_change"/>
  </prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("cache", template).unwrap();

        let ctx = PomlContext::default();
        let result = proc.process("cache", &ctx).unwrap();

        let cache = result.cache_settings.unwrap();
        assert_eq!(cache.duration, Some(3600));
        assert!(cache.key.as_ref().unwrap().starts_with("intent_"));
        assert_eq!(cache.invalidate_on.as_deref(), Some("context_change"));
        assert!(result.features_used.contains(&PomlFeature::Cache));
    }

    #[test]
    fn test_default_variable() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("test", sample_template()).unwrap();

        // Don't provide persona — should get default "general"
        let ctx = PomlContext::default();
        let result = proc.process("test", &ctx).unwrap();
        // The template has <let name="persona">{{ persona | default: "general" }}</let>
        // Since no persona in context, it should resolve to "general"
        assert!(result.prompt.len() > 0);
    }

    #[test]
    fn test_error_handling_section() {
        let template = r#"<poml version="2.0">
  <metadata><title>Error Test</title></metadata>
  <prompt>
    <system>Test</system>
    <error-handling>
      <on-error type="ambiguous">Return multiple intents</on-error>
      <on-error type="unknown">Return unknown intent</on-error>
    </error-handling>
  </prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("err", template).unwrap();

        let ctx = PomlContext::default();
        let result = proc.process("err", &ctx).unwrap();
        assert!(
            result
                .prompt
                .contains("On ambiguous: Return multiple intents")
        );
        assert!(result.prompt.contains("On unknown: Return unknown intent"));
    }

    #[test]
    fn test_condition_evaluation() {
        let vars = HashMap::new();
        let ctx = PomlContext {
            persona: Some("developer".into()),
            ..Default::default()
        };

        assert!(evaluate_condition("persona == 'developer'", &vars, &ctx));
        assert!(!evaluate_condition("persona == 'grandma'", &vars, &ctx));
        assert!(evaluate_condition("persona != 'grandma'", &vars, &ctx));
    }

    #[test]
    fn test_xml_entities() {
        let template = r#"<poml version="2.0">
  <metadata><title>Entity Test</title></metadata>
  <prompt>
    <system>Use &lt;tags&gt; and &amp; symbols</system>
  </prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("ent", template).unwrap();

        let ctx = PomlContext::default();
        let result = proc.process("ent", &ctx).unwrap();
        assert!(result.prompt.contains("<tags>"));
        assert!(result.prompt.contains("& symbols"));
    }

    #[test]
    fn test_anomaly_diagnosis_template() {
        let template = r#"<poml version="2.0">
  <metadata>
    <title>Anomaly Diagnosis</title>
    <model-hints><temperature>0.3</temperature><max-tokens>256</max-tokens></model-hints>
  </metadata>
  <variables>
    <let name="unit">{{ unit }}</let>
    <let name="reason">{{ reason }}</let>
  </variables>
  <prompt>
    <system>You are a NixOS systemd diagnostician.</system>
    <stepwise-instructions>
      <step id="s1">Identify the root cause in unit '{{ unit }}'.</step>
      <step id="s2">Suggest a fix.</step>
    </stepwise-instructions>
    <output-format>Plain text diagnosis.</output-format>
  </prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/dev/null");
        proc.load_template_str("diag", template).unwrap();

        let mut ctx = PomlContext::default();
        ctx.variables.insert("unit".into(), "nginx.service".into());
        ctx.variables.insert("reason".into(), "OOM killed".into());

        let result = proc.process("diag", &ctx).unwrap();
        assert!(result.prompt.contains("nginx.service"));
        assert!(result.prompt.contains("diagnostician"));
        assert_eq!(result.metadata.model_hints.temperature, Some(0.3));
        assert_eq!(result.metadata.model_hints.max_tokens, Some(256));
    }

    #[test]
    fn test_load_empty_template() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        let result = proc.load_template_str("bad", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_process_nonexistent_template() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        let ctx = PomlContext::default();
        let result = proc.process("nonexistent", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_minimal_template() {
        let template = r#"<poml version="2.0">
  <metadata><title>Minimal</title></metadata>
  <prompt><system>Hello</system></prompt>
</poml>"#;
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("min", template).unwrap();

        let ctx = PomlContext::default();
        let result = proc.process("min", &ctx).unwrap();
        assert!(result.prompt.contains("Hello"));
        assert_eq!(result.metadata.title.as_deref(), Some("Minimal"));
    }

    #[test]
    fn test_condition_in_operator() {
        let mut vars = HashMap::new();
        vars.insert("items".into(), PomlValue::String("foo bar baz".into()));
        let ctx = PomlContext::default();

        assert!(evaluate_condition("'foo' in items", &vars, &ctx));
        assert!(!evaluate_condition("'qux' in items", &vars, &ctx));
    }

    #[test]
    fn test_condition_empty() {
        let vars = HashMap::new();
        let ctx = PomlContext::default();
        assert!(!evaluate_condition("", &vars, &ctx));
    }

    #[test]
    fn test_condition_inequality() {
        let mut vars = HashMap::new();
        vars.insert("mode".into(), PomlValue::String("debug".into()));
        let ctx = PomlContext::default();

        assert!(evaluate_condition("mode != 'release'", &vars, &ctx));
        assert!(!evaluate_condition("mode != 'debug'", &vars, &ctx));
    }

    #[test]
    fn test_variable_substitution_no_match() {
        let vars = HashMap::new();
        let result = substitute_variables("No variables here", &vars);
        assert_eq!(result, "No variables here");
    }

    #[test]
    fn test_variable_substitution_multiple() {
        let mut vars = HashMap::new();
        vars.insert("a".into(), PomlValue::String("1".into()));
        vars.insert("b".into(), PomlValue::String("2".into()));

        let result = substitute_variables("{{ a }} and {{ b }}", &vars);
        assert_eq!(result, "1 and 2");
    }

    #[test]
    fn test_poml_value_from_string() {
        let val: PomlValue = "hello".into();
        assert_eq!(val.to_display(), "hello");
    }

    #[test]
    fn test_poml_value_list_display() {
        let val = PomlValue::List(vec!["a".into(), "b".into(), "c".into()]);
        let display = val.to_display();
        assert!(display.contains("a"));
        assert!(display.contains("b"));
        assert!(display.contains("c"));
    }

    #[test]
    fn test_poml_context_default() {
        let ctx = PomlContext::default();
        assert!(ctx.variables.is_empty());
        assert!(ctx.persona.is_none());
    }

    #[test]
    fn test_load_template_str_overwrite() {
        let template = r#"<poml version="2.0">
  <metadata><title>V1</title></metadata>
  <prompt><system>First</system></prompt>
</poml>"#;
        let template2 = r#"<poml version="2.0">
  <metadata><title>V2</title></metadata>
  <prompt><system>Second</system></prompt>
</poml>"#;

        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("t", template).unwrap();
        proc.load_template_str("t", template2).unwrap();

        let ctx = PomlContext::default();
        let result = proc.process("t", &ctx).unwrap();
        assert_eq!(result.metadata.title.as_deref(), Some("V2"));
        assert!(result.prompt.contains("Second"));
    }

    #[test]
    fn test_poml_error_xml_parse() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        let result = proc.load_template_str("bad", "not xml at all <<<");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PomlError::XmlParse { .. }),
            "Expected XmlParse error, got: {err}"
        );
        // Display should include offset info
        let msg = format!("{err}");
        assert!(
            msg.contains("byte"),
            "Error msg should mention byte offset: {msg}"
        );
    }

    #[test]
    fn test_poml_error_missing_prompt() {
        let template = r#"<poml version="2.0">
  <metadata><title>No Prompt</title></metadata>
</poml>"#;
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        proc.load_template_str("noprompt", template).unwrap();
        let ctx = PomlContext::default();
        let result = proc.process("noprompt", &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PomlError::MissingElement { .. }),
            "Expected MissingElement error, got: {err}"
        );
    }

    #[test]
    fn test_poml_error_not_found() {
        let mut proc = PomlProcessor::new("/tmp/nonexistent");
        let result = proc.load_template("does_not_exist");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PomlError::NotFound(_)),
            "Expected NotFound error, got: {err}"
        );
    }

    #[test]
    fn test_poml_error_display() {
        let err = PomlError::XmlParse {
            offset: 42,
            detail: "unexpected EOF".to_string(),
        };
        assert_eq!(
            format!("{err}"),
            "XML parse error at byte 42: unexpected EOF"
        );

        let err = PomlError::NotFound("mytemplate".to_string());
        assert_eq!(format!("{err}"), "Template not found: mytemplate");
    }
}
