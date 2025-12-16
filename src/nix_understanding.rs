/*!
NixOS Understanding Module
Maps natural language → NixOS operations using HDC + LTC
*/

use anyhow::Result;

/// NixOS command understanding
pub struct NixUnderstanding {
    /// Command templates
    templates: Vec<CommandTemplate>,
}

pub struct CommandTemplate {
    pub intent: String,
    pub pattern: String,
    pub nix_command: String,
    pub confidence: f32,
}

impl NixUnderstanding {
    pub fn new() -> Self {
        Self {
            templates: vec![
                CommandTemplate {
                    intent: "install".to_string(),
                    pattern: "install *".to_string(),
                    nix_command: "nix-env -i".to_string(),
                    confidence: 0.9,
                },
                CommandTemplate {
                    intent: "search".to_string(),
                    pattern: "search *".to_string(),
                    nix_command: "nix search nixpkgs".to_string(),
                    confidence: 0.9,
                },
                CommandTemplate {
                    intent: "configure".to_string(),
                    pattern: "configure *".to_string(),
                    nix_command: "# Edit /etc/nixos/configuration.nix".to_string(),
                    confidence: 0.8,
                },
            ],
        }
    }

    /// Map query to NixOS command
    pub fn understand(&self, query: &str) -> Result<String> {
        // Find best matching template
        let query_lower = query.to_lowercase();

        for template in &self.templates {
            if query_lower.contains(&template.intent) {
                return Ok(format!(
                    "{} (confidence: {:.0}%)",
                    template.nix_command,
                    template.confidence * 100.0
                ));
            }
        }

        Ok("I'm not sure how to help with that yet.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nix_understanding() {
        let nix = NixUnderstanding::new();

        let result = nix.understand("install firefox").unwrap();
        assert!(result.contains("nix-env -i"));

        let result = nix.understand("search vim").unwrap();
        assert!(result.contains("nix search"));
    }
}
