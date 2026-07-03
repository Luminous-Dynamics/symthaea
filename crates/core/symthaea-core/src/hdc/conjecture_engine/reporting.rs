use super::*;

impl ConjectureEngine {
    /// Get the best verified conjecture for a given source.
    pub fn best_for(&self, source: &str) -> Option<&Conjecture> {
        self.conjectures
            .iter()
            .filter(|c| c.source == source)
            .min_by(|a, b| compare_conjectures_for_selection(a, b))
    }

    /// Generate a human-readable report of all conjectures.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("═══ Conjecture Engine Report ═══".to_string());
        lines.push(format!("Sequences observed: {}", self.observations.len()));
        lines.push(format!("Conjectures generated: {}", self.conjectures.len()));
        lines.push(String::new());

        for (i, c) in self.conjectures.iter().enumerate().take(10) {
            lines.push(format!("#{}: {} ≈ {}", i + 1, c.source, c.formula_str,));
            lines.push(format!(
                "   MSE={:.2e}, complexity={}, confidence={:.2}, status={:?}",
                c.training_mse, c.complexity, c.confidence, c.status,
            ));
            if let Some(label) = eml_backend_label(c) {
                lines.push(format!("   backend={label}"));
            }
        }
        lines.join("\n")
    }

    /// Emit a paper-ready LaTeX table of the best conjecture per source.
    pub fn discovery_report_latex(
        &self,
        annotations: Option<&std::collections::HashMap<String, String>>,
    ) -> String {
        let mut out = String::new();

        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut rows: Vec<&Conjecture> = Vec::new();
        for c in &self.conjectures {
            if seen.insert(c.source.as_str())
                && let Some(best) = self.best_for(&c.source)
            {
                rows.push(best);
            }
        }

        let has_annotations = annotations.map(|m| !m.is_empty()).unwrap_or(false);
        let col_spec = if has_annotations { "llrll" } else { "llrl" };

        out.push_str("\\begin{table}[htbp]\n");
        out.push_str("\\centering\n");
        out.push_str(
            "\\caption{Autonomous discoveries from the Ramanujan Protocol conjecture engine.}\n",
        );
        out.push_str("\\label{tab:ramanujan_discoveries}\n");
        out.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
        out.push_str("\\toprule\n");
        if has_annotations {
            out.push_str("Sequence & Discovered Formula & MSE & Status & Recognition \\\\\n");
        } else {
            out.push_str("Sequence & Discovered Formula & MSE & Status \\\\\n");
        }
        out.push_str("\\midrule\n");

        for c in &rows {
            let formula_latex = expr_to_latex(&c.formula);
            let mut status_tag = match &c.status {
                ConjectureStatus::FormallyVerified { .. } => "\\textbf{Formal}",
                ConjectureStatus::NumericallyTested { .. } => "Numeric",
                ConjectureStatus::SymbolicallyChecked => "Symbolic",
                ConjectureStatus::Refuted { .. } => "Refuted",
                ConjectureStatus::Proposed => "Proposed",
            }
            .to_string();
            if let Some(label) = eml_backend_label(c) {
                status_tag.push_str(" / ");
                status_tag.push_str(&label);
            }

            let sanitized_source = latex_escape(&c.source);

            let mse_display = if c.training_mse < 1e-10 {
                "$< 10^{-10}$".to_string()
            } else if c.training_mse < 1.0 {
                format!("${:.2e}$", c.training_mse)
            } else {
                format!("${:.3}$", c.training_mse)
            };

            if has_annotations {
                let ann = annotations
                    .and_then(|m| m.get(&c.source))
                    .cloned()
                    .unwrap_or_else(|| "--".to_string());
                let sanitized_ann = latex_escape(&ann);
                out.push_str(&format!(
                    "{} & ${}$ & {} & {} & {} \\\\\n",
                    sanitized_source,
                    formula_latex,
                    mse_display,
                    latex_escape(&status_tag),
                    sanitized_ann
                ));
            } else {
                out.push_str(&format!(
                    "{} & ${}$ & {} & {} \\\\\n",
                    sanitized_source,
                    formula_latex,
                    mse_display,
                    latex_escape(&status_tag)
                ));
            }
        }

        out.push_str("\\bottomrule\n");
        out.push_str("\\end{tabular}\n");
        out.push_str("\\end{table}\n");

        out
    }

    /// Emit a plain-text summary of the best conjecture per source with optional
    /// recognition annotations, ready for console display.
    pub fn discovery_report_text(
        &self,
        annotations: Option<&std::collections::HashMap<String, String>>,
    ) -> String {
        let mut out = String::new();

        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut rows: Vec<&Conjecture> = Vec::new();
        for c in &self.conjectures {
            if seen.insert(c.source.as_str())
                && let Some(best) = self.best_for(&c.source)
            {
                rows.push(best);
            }
        }

        out.push_str("╔══════════════════════════════════════════════════════════════════════╗\n");
        out.push_str("║              RAMANUJAN PROTOCOL — DISCOVERY REPORT                   ║\n");
        out.push_str("╠══════════════════════════════════════════════════════════════════════╣\n");

        for c in &rows {
            let mut status_tag = match &c.status {
                ConjectureStatus::FormallyVerified { proof_steps } => {
                    format!("FORMAL ✓ ({} steps)", proof_steps)
                }
                ConjectureStatus::NumericallyTested { .. } => "Numeric".to_string(),
                ConjectureStatus::SymbolicallyChecked => "Symbolic".to_string(),
                ConjectureStatus::Refuted { .. } => "REFUTED".to_string(),
                ConjectureStatus::Proposed => "Proposed".to_string(),
            };
            if let Some(label) = eml_backend_label(c) {
                status_tag.push_str(" / ");
                status_tag.push_str(&label);
            }

            out.push_str(&format!(
                "║ {:35} │ MSE {:.2e} │ {}\n",
                truncate(&c.source, 35),
                c.training_mse,
                status_tag
            ));
            out.push_str(&format!("║   {}\n", c.formula_str));
            if let Some(anns) = annotations
                && let Some(headline) = anns.get(&c.source)
            {
                out.push_str(&format!("║   {}\n", headline));
            }
        }

        out.push_str("╚══════════════════════════════════════════════════════════════════════╝\n");
        out
    }

    /// Test whether a conjecture's formula fits a sequence from a different domain.
    pub fn cross_fit(conjecture: &Conjecture, target_seq: &ObservedSequence) -> Option<f64> {
        if conjecture.domain == target_seq.domain {
            return None;
        }
        let cross_mse = compute_mse(&conjecture.formula, &target_seq.data);
        if !cross_mse.is_finite() || conjecture.training_mse <= 0.0 {
            return None;
        }
        Some(cross_mse / conjecture.training_mse)
    }

    /// Discover all cross-domain formula matches.
    pub fn discover_cross_domain_formulas(
        &self,
        max_mse_ratio: f64,
    ) -> Vec<CrossDomainFormulaMatch> {
        let mut matches = Vec::new();

        for conjecture in &self.conjectures {
            if conjecture.confidence < 0.3 {
                continue;
            }

            for target_seq in &self.observations {
                if let Some(mse_ratio) = Self::cross_fit(conjecture, target_seq)
                    && mse_ratio < max_mse_ratio
                {
                    matches.push(CrossDomainFormulaMatch {
                        formula_str: conjecture.formula_str.clone(),
                        source_seq: conjecture.source.clone(),
                        source_domain: conjecture.domain,
                        target_seq: target_seq.name.clone(),
                        target_domain: target_seq.domain,
                        source_mse: conjecture.training_mse,
                        target_mse: mse_ratio * conjecture.training_mse,
                        mse_ratio,
                        confidence: conjecture.confidence,
                    });
                }
            }
        }

        matches.sort_by(|a, b| {
            a.mse_ratio
                .partial_cmp(&b.mse_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Autonomous Langlands discovery: feed all known elliptic curve L-functions
    /// and modular form q-expansions, then search for correspondences.
    pub fn discover_langlands(&mut self, max_p: u64) -> Vec<LanglandsDiscovery> {
        let pairs = super::super::langlands::langlands_observation_set(max_p);
        let mut discoveries = Vec::new();

        let mut curve_seqs = Vec::new();
        let mut form_seqs = Vec::new();

        for (l_seq, q_seq) in &pairs {
            curve_seqs.push(l_seq.clone());
            form_seqs.push(q_seq.clone());
        }

        for seq in curve_seqs.iter().chain(form_seqs.iter()) {
            self.observe(seq.clone());
        }

        for curve_seq in &curve_seqs {
            for form_seq in &form_seqs {
                let curve_map: std::collections::HashMap<i64, f64> = curve_seq
                    .data
                    .iter()
                    .map(|(x, y)| (*x as i64, *y))
                    .collect();
                let form_map: std::collections::HashMap<i64, f64> =
                    form_seq.data.iter().map(|(x, y)| (*x as i64, *y)).collect();

                let common: Vec<i64> = curve_map
                    .keys()
                    .filter(|k| form_map.contains_key(k))
                    .cloned()
                    .collect();

                if common.len() < 3 {
                    continue;
                }

                let matches = common
                    .iter()
                    .filter(|k| (curve_map[k] - form_map[k]).abs() < 0.5)
                    .count();

                let total = common.len();
                let match_rate = matches as f64 / total as f64;

                if match_rate > 0.9 {
                    discoveries.push(LanglandsDiscovery {
                        curve: curve_seq.name.clone(),
                        form: form_seq.name.clone(),
                        relation: if match_rate > 0.99 {
                            format!("IDENTITY: a_p = c_p ({}/{} exact)", matches, total)
                        } else {
                            format!(
                                "APPROXIMATE: {}/{} match ({:.1}%)",
                                matches,
                                total,
                                match_rate * 100.0
                            )
                        },
                        matching_primes: matches,
                        total_primes: total,
                        is_identity: match_rate > 0.99,
                    });
                }
            }
        }

        for curve_seq in &curve_seqs {
            for form_seq in &form_seqs {
                let relations = discover_cross_sequence_relations(curve_seq, form_seq);
                for rel in relations {
                    if rel.r_squared > 0.9 {
                        discoveries.push(LanglandsDiscovery {
                            curve: curve_seq.name.clone(),
                            form: form_seq.name.clone(),
                            relation: format!("{}", rel),
                            matching_primes: 0,
                            total_primes: 0,
                            is_identity: rel.r_squared > 0.999,
                        });
                    }
                }
            }
        }

        discoveries.sort_by_key(|d| std::cmp::Reverse(d.matching_primes));
        discoveries
    }
}

/// A cross-domain formula match: one formula fits data from two different domains.
#[derive(Debug, Clone)]
pub struct CrossDomainFormulaMatch {
    pub formula_str: String,
    pub source_seq: String,
    pub source_domain: MathDomain,
    pub target_seq: String,
    pub target_domain: MathDomain,
    pub source_mse: f64,
    pub target_mse: f64,
    pub mse_ratio: f64,
    pub confidence: f64,
}

impl fmt::Display for CrossDomainFormulaMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}) → {} ({:?}): f(n) ≈ {} [ratio={:.2}]",
            self.source_seq,
            self.source_domain,
            self.target_seq,
            self.target_domain,
            self.formula_str,
            self.mse_ratio
        )
    }
}

/// Result of autonomous Langlands discovery.
#[derive(Debug, Clone)]
pub struct LanglandsDiscovery {
    pub curve: String,
    pub form: String,
    pub relation: String,
    pub matching_primes: usize,
    pub total_primes: usize,
    pub is_identity: bool,
}

impl std::fmt::Display for LanglandsDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_identity {
            write!(
                f,
                "MODULARITY: {} ↔ {} ({}/{} primes, {})",
                self.curve, self.form, self.matching_primes, self.total_primes, self.relation
            )
        } else {
            write!(
                f,
                "RELATION: {} ~ {} ({})",
                self.curve, self.form, self.relation
            )
        }
    }
}
