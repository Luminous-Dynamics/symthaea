//! Code Perception - Giving Sophia the ability to understand source code
//!
//! Uses `tree-sitter`, `syn`, and `ignore` crates for code analysis.

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use syn::{visit::Visit, Item};

/// Project structure information
#[derive(Debug, Clone)]
pub struct ProjectStructure {
    /// Root path of the project
    pub root: PathBuf,

    /// Total number of files
    pub file_count: usize,

    /// Files by language/extension
    pub files_by_type: HashMap<String, Vec<PathBuf>>,

    /// Total lines of code
    pub total_lines: usize,

    /// Main programming languages detected
    pub languages: Vec<String>,
}

/// Rust code semantics information
#[derive(Debug, Clone, Default)]
pub struct RustCodeSemantics {
    /// Number of functions
    pub function_count: usize,

    /// Number of structs
    pub struct_count: usize,

    /// Number of enums
    pub enum_count: usize,

    /// Number of traits
    pub trait_count: usize,

    /// Number of impl blocks
    pub impl_count: usize,

    /// Number of modules
    pub module_count: usize,

    /// Public vs private items ratio (0.0 = all private, 1.0 = all public)
    pub public_ratio: f32,

    /// Function names found
    pub function_names: Vec<String>,

    /// Struct names found
    pub struct_names: Vec<String>,
}

/// Code quality analysis
#[derive(Debug, Clone)]
pub struct CodeQualityAnalysis {
    /// Average function length (lines)
    pub avg_function_length: f32,

    /// Maximum function length (lines)
    pub max_function_length: usize,

    /// Documentation coverage (0.0-1.0)
    pub doc_coverage: f32,

    /// Test coverage estimate (0.0-1.0)
    pub test_coverage_estimate: f32,

    /// Complexity indicators
    pub complexity_indicators: Vec<String>,

    /// Potential improvements
    pub suggestions: Vec<String>,
}

/// Code Perception Cortex - Understanding source code
pub struct CodePerceptionCortex {
    /// Maximum files to analyze in one pass
    max_files: usize,

    /// Maximum lines per file to analyze
    max_lines_per_file: usize,
}

impl Default for CodePerceptionCortex {
    fn default() -> Self {
        Self {
            max_files: 10000,
            max_lines_per_file: 50000,
        }
    }
}

impl CodePerceptionCortex {
    /// Create a new code perception cortex
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze project structure by walking directory tree
    pub fn analyze_project(&self, path: &Path) -> Result<ProjectStructure> {
        if !path.exists() {
            anyhow::bail!("Path does not exist: {:?}", path);
        }

        let mut files_by_type: HashMap<String, Vec<PathBuf>> = HashMap::new();
        let mut file_count = 0;
        let mut total_lines = 0;

        // Use ignore crate to respect .gitignore
        let walker = WalkBuilder::new(path)
            .hidden(false) // Include hidden files
            .git_ignore(true) // Respect .gitignore
            .build();

        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                let path = entry.path().to_path_buf();

                // Count lines
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let lines = content.lines().count();
                    if lines <= self.max_lines_per_file {
                        total_lines += lines;

                        // Categorize by extension
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            files_by_type
                                .entry(ext.to_string())
                                .or_insert_with(Vec::new)
                                .push(path);
                        }

                        file_count += 1;

                        if file_count >= self.max_files {
                            break;
                        }
                    }
                }
            }
        }

        // Detect main languages
        let mut languages: Vec<(String, usize)> = files_by_type
            .iter()
            .map(|(ext, files)| (Self::extension_to_language(ext), files.len()))
            .collect();
        languages.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        let languages = languages
            .into_iter()
            .take(5)
            .map(|(lang, _)| lang)
            .collect();

        Ok(ProjectStructure {
            root: path.to_path_buf(),
            file_count,
            files_by_type,
            total_lines,
            languages,
        })
    }

    /// Map file extension to programming language name
    fn extension_to_language(ext: &str) -> String {
        match ext {
            "rs" => "Rust",
            "py" => "Python",
            "js" | "jsx" => "JavaScript",
            "ts" | "tsx" => "TypeScript",
            "c" | "h" => "C",
            "cpp" | "cc" | "cxx" | "hpp" => "C++",
            "go" => "Go",
            "java" => "Java",
            "rb" => "Ruby",
            "php" => "PHP",
            "swift" => "Swift",
            "kt" | "kts" => "Kotlin",
            "md" => "Markdown",
            "json" => "JSON",
            "yaml" | "yml" => "YAML",
            "toml" => "TOML",
            "nix" => "Nix",
            _ => ext,
        }
        .to_string()
    }

    /// Understand Rust code using syn parser
    pub fn understand_rust_code(&self, path: &Path) -> Result<RustCodeSemantics> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read Rust file: {:?}", path))?;

        let syntax_tree = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse Rust file: {:?}", path))?;

        let mut visitor = RustVisitor::default();
        visitor.visit_file(&syntax_tree);
        visitor.finalize();

        Ok(visitor.semantics.clone())
    }

    /// Analyze code quality (basic heuristics)
    pub fn analyze_code_quality(&self, path: &Path) -> Result<CodeQualityAnalysis> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {:?}", path))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Count functions (simple heuristic)
        let function_count = lines.iter()
            .filter(|line| line.trim_start().starts_with("fn ") || line.trim_start().starts_with("pub fn "))
            .count();

        // Count doc comments
        let doc_lines = lines.iter()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("///") || trimmed.starts_with("//!")
            })
            .count();

        // Count test functions
        let test_count = lines.iter()
            .filter(|line| line.trim_start().starts_with("#[test]"))
            .count();

        // Estimate function lengths
        let avg_function_length = if function_count > 0 {
            (total_lines as f32 / function_count as f32).min(1000.0)
        } else {
            0.0
        };

        let max_function_length = avg_function_length as usize * 3; // Rough estimate

        // Documentation coverage
        let doc_coverage = if total_lines > 0 {
            (doc_lines as f32 / total_lines as f32).min(1.0)
        } else {
            0.0
        };

        // Test coverage estimate
        let test_coverage_estimate = if function_count > 0 {
            (test_count as f32 / function_count as f32).min(1.0)
        } else {
            0.0
        };

        // Complexity indicators
        let mut complexity_indicators = Vec::new();
        if avg_function_length > 50.0 {
            complexity_indicators.push("Long functions detected".to_string());
        }
        if doc_coverage < 0.1 {
            complexity_indicators.push("Low documentation coverage".to_string());
        }
        if test_coverage_estimate < 0.3 {
            complexity_indicators.push("Low test coverage".to_string());
        }

        // Suggestions
        let mut suggestions = Vec::new();
        if avg_function_length > 50.0 {
            suggestions.push("Consider breaking down long functions".to_string());
        }
        if doc_coverage < 0.2 {
            suggestions.push("Add more documentation comments".to_string());
        }
        if test_coverage_estimate < 0.5 {
            suggestions.push("Increase test coverage".to_string());
        }

        Ok(CodeQualityAnalysis {
            avg_function_length,
            max_function_length,
            doc_coverage,
            test_coverage_estimate,
            complexity_indicators,
            suggestions,
        })
    }
}

/// Visitor pattern for traversing Rust syntax tree
#[derive(Default)]
struct RustVisitor {
    semantics: RustCodeSemantics,
    public_count: usize,
    private_count: usize,
}

impl<'ast> Visit<'ast> for RustVisitor {
    fn visit_item(&mut self, item: &'ast Item) {
        match item {
            Item::Fn(func) => {
                self.semantics.function_count += 1;
                self.semantics.function_names.push(func.sig.ident.to_string());

                if matches!(func.vis, syn::Visibility::Public(_)) {
                    self.public_count += 1;
                } else {
                    self.private_count += 1;
                }
            }
            Item::Struct(s) => {
                self.semantics.struct_count += 1;
                self.semantics.struct_names.push(s.ident.to_string());

                if matches!(s.vis, syn::Visibility::Public(_)) {
                    self.public_count += 1;
                } else {
                    self.private_count += 1;
                }
            }
            Item::Enum(_) => {
                self.semantics.enum_count += 1;
            }
            Item::Trait(_) => {
                self.semantics.trait_count += 1;
            }
            Item::Impl(_) => {
                self.semantics.impl_count += 1;
            }
            Item::Mod(_) => {
                self.semantics.module_count += 1;
            }
            _ => {}
        }

        // Continue visiting children
        syn::visit::visit_item(self, item);
    }
}

impl RustVisitor {
    /// Finalize the semantics by calculating public_ratio
    fn finalize(&mut self) {
        let total = self.public_count + self.private_count;
        self.semantics.public_ratio = if total > 0 {
            self.public_count as f32 / total as f32
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    /// Create a temporary Rust file for testing
    fn create_temp_rust_file(content: &str) -> Result<(TempDir, PathBuf)> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.rs");
        let mut file = fs::File::create(&file_path)?;
        file.write_all(content.as_bytes())?;
        Ok((temp_dir, file_path))
    }

    #[test]
    fn test_cortex_creation() {
        let cortex = CodePerceptionCortex::new();
        assert_eq!(cortex.max_files, 10000);
        assert_eq!(cortex.max_lines_per_file, 50000);
    }

    #[test]
    fn test_understand_rust_code() {
        let code = r#"
            pub struct MyStruct {
                field: i32,
            }

            pub fn my_function() -> i32 {
                42
            }

            fn private_function() {
                println!("Hello");
            }

            pub enum MyEnum {
                Variant1,
                Variant2,
            }
        "#;

        let (_temp, path) = create_temp_rust_file(code).unwrap();
        let cortex = CodePerceptionCortex::new();

        let semantics = cortex.understand_rust_code(&path).unwrap();

        assert_eq!(semantics.struct_count, 1);
        assert_eq!(semantics.function_count, 2);
        assert_eq!(semantics.enum_count, 1);
        assert!(semantics.public_ratio > 0.5); // More public than private
    }

    #[test]
    fn test_analyze_code_quality() {
        let code = r#"
            /// This is a documented function
            pub fn well_documented() -> i32 {
                42
            }

            fn undocumented() {
                println!("No docs");
            }

            #[test]
            fn test_something() {
                assert_eq!(well_documented(), 42);
            }
        "#;

        let (_temp, path) = create_temp_rust_file(code).unwrap();
        let cortex = CodePerceptionCortex::new();

        let quality = cortex.analyze_code_quality(&path).unwrap();

        assert!(quality.doc_coverage > 0.0);
        assert!(quality.test_coverage_estimate > 0.0);
    }

    #[test]
    fn test_extension_to_language() {
        assert_eq!(CodePerceptionCortex::extension_to_language("rs"), "Rust");
        assert_eq!(CodePerceptionCortex::extension_to_language("py"), "Python");
        assert_eq!(CodePerceptionCortex::extension_to_language("js"), "JavaScript");
        assert_eq!(CodePerceptionCortex::extension_to_language("nix"), "Nix");
    }
}
