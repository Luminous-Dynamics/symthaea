//! NixOS Package Encoder
//!
//! Encodes package metadata (name, description, categories) into ContinuousHV.
//! Packages in the same category or with similar names will have higher
//! similarity in HDC space.

use super::codebook::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// Metadata about a Nix package for encoding.
#[derive(Debug, Clone)]
pub struct PackageMetadata {
    /// Package attribute path (e.g., "firefox", "python311Packages.numpy")
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Category tags (e.g., ["web", "browser"])
    pub categories: Vec<String>,
}

/// Encodes Nix packages into HDC vectors.
pub struct PackageEncoder<'a> {
    codebook: &'a mut NixCodebook,
}

impl<'a> PackageEncoder<'a> {
    /// Create a new package encoder using a shared codebook.
    pub fn new(codebook: &'a mut NixCodebook) -> Self {
        Self { codebook }
    }

    /// Encode a package name into a hypervector.
    ///
    /// The name is split on `-` and `.` to capture sub-tokens:
    /// `python311Packages.numpy` → bundle(python311Packages, numpy)
    pub fn encode_name(&mut self, name: &str) -> ContinuousHV {
        let tokens: Vec<&str> = name
            .split(['.', '-', '_'])
            .filter(|t| !t.is_empty())
            .collect();

        if tokens.is_empty() {
            return ContinuousHV::zero(self.codebook.dim());
        }

        let role = self.codebook.package_role.clone();
        let encoded: Vec<ContinuousHV> = tokens
            .iter()
            .map(|tok| {
                let basis = self.codebook.get_or_create(tok).clone();
                basis.bind(&role)
            })
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Encode package metadata (name + description + categories).
    ///
    /// Returns a weighted bundle where name has highest weight.
    pub fn encode_metadata(&mut self, metadata: &PackageMetadata) -> ContinuousHV {
        let mut components = Vec::new();
        let mut weights = Vec::new();

        // Name (highest weight)
        components.push(self.encode_name(&metadata.name));
        weights.push(1.0);

        // Description tokens (if available)
        if let Some(ref desc) = metadata.description {
            let desc_hv = self.encode_text_tokens(desc);
            components.push(desc_hv);
            weights.push(0.3);
        }

        // Category tags
        if !metadata.categories.is_empty() {
            let cat_hvs: Vec<ContinuousHV> = metadata
                .categories
                .iter()
                .map(|cat| self.codebook.get_or_create(cat).clone())
                .collect();
            let refs: Vec<&ContinuousHV> = cat_hvs.iter().collect();
            components.push(ContinuousHV::bundle(&refs));
            weights.push(0.5);
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode description text by tokenizing and bundling.
    fn encode_text_tokens(&mut self, text: &str) -> ContinuousHV {
        let stopwords = [
            "a", "an", "the", "is", "are", "was", "were", "for", "and", "or", "but", "in", "on",
            "at", "to", "of", "with", "by",
        ];

        let tokens: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 2 && !stopwords.contains(w))
            .take(20) // Limit to avoid noise from long descriptions
            .collect();

        if tokens.is_empty() {
            return ContinuousHV::zero(self.codebook.dim());
        }

        let encoded: Vec<ContinuousHV> = tokens
            .iter()
            .map(|tok| self.codebook.get_or_create(&tok.to_lowercase()).clone())
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similar_packages_similar_vectors() {
        let mut cb = NixCodebook::new();

        let firefox = {
            let mut enc = PackageEncoder::new(&mut cb);
            enc.encode_name("firefox")
        };
        let firefox_esr = {
            let mut enc = PackageEncoder::new(&mut cb);
            enc.encode_name("firefox-esr")
        };
        let postgresql = {
            let mut enc = PackageEncoder::new(&mut cb);
            enc.encode_name("postgresql")
        };

        let sim_ff = firefox.similarity(&firefox_esr);
        let sim_cross = firefox.similarity(&postgresql);

        assert!(
            sim_ff > sim_cross,
            "firefox and firefox-esr ({:.3}) should be more similar than firefox and postgresql ({:.3})",
            sim_ff, sim_cross,
        );
    }

    #[test]
    fn test_metadata_encoding() {
        let mut cb = NixCodebook::new();
        let meta = PackageMetadata {
            name: "nginx".to_string(),
            description: Some("High performance web server".to_string()),
            categories: vec!["web".to_string(), "server".to_string()],
        };

        let hv = {
            let mut enc = PackageEncoder::new(&mut cb);
            enc.encode_metadata(&meta)
        };

        assert!(hv.dim() > 0);
        assert!(hv.norm() > 0.0);
    }
}
