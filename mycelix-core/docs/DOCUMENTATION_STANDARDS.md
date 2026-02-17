# 📝 Mycelix Documentation Standards

**Guidelines for creating and maintaining excellent documentation**

---

## 🎯 Core Principles

### 1. **Progressive Disclosure**
Start simple, reveal complexity as needed. Documents should be scannable at multiple depths:
- **Level 1**: Title + summary (30 seconds)
- **Level 2**: Quick start / key points (5 minutes)
- **Level 3**: Complete documentation (30+ minutes)

### 2. **Single Source of Truth**
Each concept should have ONE canonical document. All other mentions should link to it.

### 3. **Explicit Versioning**
All significant documents must include version numbers and supersession tracking.

### 4. **Navigation Aids**
Every document should help users understand where they are and where they can go next.

### 5. **Honesty Over Hype**
Documentation reflects reality, not aspiration. Mark estimates, projections, and TODO items clearly.

---

## 📋 Document Structure

### Required Elements

Every significant document (>100 lines) must include:

#### 1. Header Block
```markdown
# Document Title

**Brief one-line description**

**Version**: v1.0
**Status**: Current | Draft | Superseded | Archived
**Last Updated**: YYYY-MM-DD
**Author(s)**: Name(s)
**Supersedes**: [Link to previous version] (if applicable)
**Superseded By**: [Link to newer version] (if applicable)
```

#### 2. Table of Contents (for long documents)
```markdown
## Table of Contents

1. [Section 1](#section-1)
2. [Section 2](#section-2)
...
```

Use `[Link Text](#lowercase-with-hyphens)` format for internal links.

#### 3. Main Content
Organized by sections with clear headings (H2, H3, H4).

#### 4. Navigation Footer
```markdown
---

📍 **Navigation**: [← Previous Doc](./link.md) | [↑ Parent Index](../README.md) | [Next Doc →](./link.md)

**Related Documents**:
- [Related Doc 1](./link1.md)
- [Related Doc 2](./link2.md)

**Last Updated**: YYYY-MM-DD
```

---

## 🏷️ Version Numbering

### Semantic Versioning for Documents

#### Major Version (v1.0 → v2.0)
Breaking changes that fundamentally alter the document's approach.

**Examples**:
- Epistemic Charter v1.0 → v2.0 (1D → 3D model)
- Constitution v0.24 → v1.0 (first stable release)

**When to increment**: Incompatible changes, complete rewrites

#### Minor Version (v1.0 → v1.1)
Additive changes that don't break existing understanding.

**Examples**:
- Adding new sections
- Clarifying ambiguous language
- Adding examples

**When to increment**: New content, clarifications

#### Patch Version (v1.1.0 → v1.1.1) - Optional
Typo fixes, formatting improvements.

**When to increment**: Minor corrections

### Date-Based Versioning
For ephemeral documents (session notes, status reports), use ISO dates:
- `SESSION_2025-11-10.md`
- `STATUS_2025-Q4.md`

---

## 📊 Status Labels

### Document Lifecycle

| Status | Meaning | Action |
|--------|---------|--------|
| **Draft** | Work in progress, not yet approved | Review and iterate |
| **Current** | Official, active version | Use as reference |
| **Deprecated** | No longer recommended, but not replaced | Migrate away when possible |
| **Superseded** | Replaced by newer version | Use the replacement |
| **Archived** | Historical reference only | Read-only, don't use |

### Status Markers in Documents

```markdown
**Status**: ✅ Current
**Status**: 🚧 Draft
**Status**: ⚠️ Deprecated
**Status**: 🔄 Superseded by [v2.0](./v2.md)
**Status**: 🗄️ Archived
```

---

## 🎨 Formatting Guidelines

### Headings

```markdown
# H1: Document Title (once per document)
## H2: Major Sections
### H3: Subsections
#### H4: Minor subsections
```

**Never skip levels** (don't go from H2 to H4).

### Emphasis

```markdown
**Bold**: Important terms, key concepts
*Italic*: Emphasis, first use of technical terms
`Code`: Commands, file paths, variable names
```

### Lists

**Unordered lists** for non-sequential items:
```markdown
- First item
- Second item
  - Nested item
```

**Ordered lists** for sequential steps:
```markdown
1. First step
2. Second step
3. Third step
```

### Code Blocks

Always specify the language:

````markdown
```bash
# Shell commands
nix develop
```

```python
# Python code
def example():
    return True
```

```json
{
  "example": "JSON data"
}
```
````

### Admonitions (Important Callouts)

```markdown
**⚠️ WARNING**: Critical information that prevents errors
**💡 TIP**: Helpful suggestions
**📋 NOTE**: Additional context
**🚨 CRITICAL**: Urgent, must-read information
```

### Links

**Internal links** (to other docs):
```markdown
[Link Text](./relative/path/to/doc.md)
[Link to Section](#section-heading)
```

**External links**:
```markdown
[Link Text](https://example.com)
```

**Always use descriptive link text** (not "click here").

### Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data     | Data     | Data     |
```

Keep tables simple. For complex data, consider separate documents.

---

## 📁 File Organization

### Naming Conventions

**File names**:
- Lowercase with hyphens: `my-document.md`
- Version in name if multiple versions exist: `epistemic-charter-v2.0.md`
- NO spaces, underscores, or special characters

**Directory names**:
- Numbered for ordered sections: `01-getting-started/`
- Descriptive: `architecture/`, `examples/`

### Folder Structure

```
docs/
├── 00-overview/           # High-level introductions
├── 01-constitution/       # Foundational principles
├── 02-charters/          # Governance framework
├── 03-architecture/      # Technical design
├── 04-sdk/               # Developer resources
├── 05-roadmap/           # Future planning
├── adr/                  # Architecture Decision Records
├── whitepaper/           # Academic papers
└── grants/               # Grant applications
```

---

## ✅ Quality Checklist

Before submitting any documentation, verify:

### Content
- [ ] Clear purpose stated in first paragraph
- [ ] All claims backed by evidence or marked as estimates/projections
- [ ] No broken internal links
- [ ] No orphaned documents (all linked from an index)
- [ ] Examples provided for complex concepts
- [ ] Consistent terminology (use glossary)

### Structure
- [ ] Header block with version/status/date
- [ ] Table of contents (if >100 lines)
- [ ] Logical section hierarchy
- [ ] Navigation footer
- [ ] Related documents linked

### Formatting
- [ ] Proper markdown syntax
- [ ] Code blocks specify language
- [ ] Lists formatted correctly
- [ ] No heading level skips
- [ ] Consistent use of emphasis

### Metadata
- [ ] Version number present
- [ ] Status clearly marked
- [ ] Last Updated date is current
- [ ] Author(s) credited
- [ ] Supersession tracked (if applicable)

---

## 🔧 Tools & Automation

### Documentation Health Checks

Run the automated checker:
```bash
./scripts/check-docs.sh
```

This checks for:
- Missing required files
- Broken internal links
- Orphaned documents
- Outdated "Last Updated" dates
- Missing version markers
- Inconsistent navigation

### MkDocs Preview

Build and preview the documentation site locally:
```bash
# Install dependencies
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-minify-plugin

# Serve locally
mkdocs serve

# Open browser to http://localhost:8000
```

### Linting

Use markdownlint for consistent formatting:
```bash
# Install
npm install -g markdownlint-cli

# Check
markdownlint docs/

# Fix auto-fixable issues
markdownlint --fix docs/
```

---

## 🎯 Special Document Types

### README Files

Every directory should have a `README.md` that:
- Explains the directory's purpose
- Links to key documents within
- Provides navigation to parent/sibling directories

**Template**:
```markdown
# Directory Name

**Purpose of this directory**

## Contents

- [Document 1](./doc1.md) - Description
- [Document 2](./doc2.md) - Description

---

📍 **Navigation**: [← Parent](../README.md) | [Sibling →](../other-dir/README.md)
```

### Architecture Decision Records (ADRs)

See [ADR Template](./adr/template.md) for structure.

Key elements:
- Context (why we need a decision)
- Decision (what we decided)
- Consequences (positive and negative outcomes)
- Alternatives considered

### API Documentation

For code APIs, follow this structure:
1. Overview (what this API does)
2. Quick example (minimal working code)
3. Reference (all methods/parameters)
4. Advanced usage
5. Error handling

### Tutorials

For step-by-step guides:
1. Prerequisites (what you need first)
2. Learning objectives (what you'll achieve)
3. Step-by-step instructions
4. Verification (how to check it worked)
5. Next steps (where to go from here)

---

## 🌟 Examples of Excellence

### Internal Examples
- [0TML Documentation](../0TML/docs/README.md) - Excellent numbered structure
- [Luminous Nix Documentation](../../11-meta-consciousness/luminous-nix/docs/README.md) - Progressive disclosure
- [Epistemic Charter v2.0](./architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md) - Clear versioning

### External Examples
- [Rust Book](https://doc.rust-lang.org/book/) - Narrative + reference
- [Nix Pills](https://nixos.org/guides/nix-pills/) - Progressive complexity
- [Stripe API Docs](https://stripe.com/docs/api) - Clear structure + examples

---

## 🚫 Common Mistakes to Avoid

### Content Mistakes
- ❌ Outdated information without "deprecated" markers
- ❌ Broken links
- ❌ Aspirational features documented as current
- ❌ Missing context (assumes too much prior knowledge)
- ❌ No examples for complex concepts

### Structural Mistakes
- ❌ No table of contents in long documents
- ❌ Heading level skips (H2 → H4)
- ❌ Missing navigation footer
- ❌ No version/status information

### Formatting Mistakes
- ❌ Code blocks without language specified
- ❌ Inconsistent list formatting
- ❌ Using "click here" as link text
- ❌ Excessive use of ALL CAPS

---

## 🔄 Review Process

### Self-Review
Before submitting:
1. Run `./scripts/check-docs.sh`
2. Read through once as a new user would
3. Verify all links work
4. Check spelling and grammar

### Peer Review
Documentation PRs should be reviewed for:
- Accuracy (is the content correct?)
- Clarity (will readers understand?)
- Completeness (are all questions answered?)
- Consistency (matches existing style?)

### Acceptance Criteria
Documentation is ready to merge when:
- ✅ Health check passes
- ✅ At least one reviewer approves
- ✅ All feedback addressed
- ✅ Navigation updated in parent README

---

## 📚 Related Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Write the Docs](https://www.writethedocs.org/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)

---

📍 **Navigation**: [← Docs Home](./README.md) | [ADR Guidelines →](./adr/README.md)

**Last Updated**: November 10, 2025
**Version**: 1.0
