// iac_repair.rs
// Example IaC repair paths for Symthaea Broca
// Wires LanguageGateRegistry + Emotional Gating into multi-tool IaC validation & self-repair
// Supports: Terraform, Kubernetes, Ansible, CloudFormation, Pulumi, Docker Compose, HCL
// In full integration: called from CodeGate or code_orchestrator after generation

use crate::emotional_gating_integration::{apply_frustration_trigger, modulate_by_emotion};
use crate::encoder::ThoughtChannels; // Assume exists
use crate::language_gates::LanguageGateRegistry;
use std::process::Command;

/// Real IaC verifier using actual CLI tool calls via std::process::Command + output parsing.
/// Falls back to heuristic if tool not available or for unsupported intents.
/// Requires IaC CLIs installed in PATH (terraform, kubectl, ansible, etc.) for full effect.
#[derive(Debug)]
pub enum IaCVerifierVerdict {
    Pass,
    Fail {
        tool: String,
        error_count: usize,
        last_error: String,
        suggestion: Option<String>,
    },
}

/// Helper: write code to temp file and run validator command, parse stderr/stdout for errors.
fn run_validator(
    tool: &str,
    args: &[&str],
    code: &str,
    tmp_file: &str,
    success_marker: &str,
) -> IaCVerifierVerdict {
    // Write code to temp file
    if let Err(e) = std::fs::write(tmp_file, code) {
        return IaCVerifierVerdict::Fail {
            tool: tool.to_string(),
            error_count: 1,
            last_error: format!("Failed to write temp file: {}", e),
            suggestion: None,
        };
    }

    let output = std::process::Command::new(tool).args(args).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let combined = format!("{}{}", stdout, stderr);
            if out.status.success() || combined.contains(success_marker) {
                IaCVerifierVerdict::Pass
            } else {
                let error_lines: Vec<&str> = combined
                    .lines()
                    .filter(|l| l.contains("Error") || l.contains("error") || l.contains("fail"))
                    .take(3)
                    .collect();
                let last_error = if error_lines.is_empty() {
                    combined.chars().take(200).collect()
                } else {
                    error_lines.join("; ")
                };
                IaCVerifierVerdict::Fail {
                    tool: tool.to_string(),
                    error_count: error_lines.len().max(1),
                    last_error,
                    suggestion: Some(format!(
                        "Run `{} {}` manually to debug",
                        tool,
                        args.join(" ")
                    )),
                }
            }
        }
        Err(e) => IaCVerifierVerdict::Fail {
            tool: tool.to_string(),
            error_count: 1,
            last_error: format!(
                "Tool '{}' not found or failed to spawn: {}. Install it for real validation.",
                tool, e
            ),
            suggestion: Some("Install the IaC CLI tool (e.g. brew install terraform)".to_string()),
        },
    }
}

pub fn verify_iac(code: &str, intent: &str) -> IaCVerifierVerdict {
    let code_lower = code.to_lowercase();
    let intent_lower = intent.to_lowercase();

    if intent_lower.contains("kubernetes") || intent_lower.contains("k8s") {
        // Real: kubectl apply --dry-run=server -f -
        // For simplicity, use file based
        run_validator(
            "kubectl",
            &["apply", "--dry-run=server", "-f", "/tmp/k8s-manifest.yaml"],
            code,
            "/tmp/k8s-manifest.yaml",
            "validated",
        )
    } else if intent_lower.contains("ansible") {
        run_validator(
            "ansible-playbook",
            &["--syntax-check", "-"],
            code,
            "/tmp/playbook.yaml",
            "Syntax OK",
        )
    } else if intent_lower.contains("cloudformation") || intent_lower.contains("cfn") {
        // aws cloudformation validate-template --template-body file://...
        run_validator(
            "aws",
            &[
                "cloudformation",
                "validate-template",
                "--template-body",
                "file:///tmp/cfn-template.yaml",
            ],
            code,
            "/tmp/cfn-template.yaml",
            "Validation succeeded",
        )
    } else if intent_lower.contains("pulumi") {
        // pulumi preview --json or just check syntax via build
        run_validator(
            "pulumi",
            &["preview", "--non-interactive", "--json"],
            code,
            "/tmp/Pulumi.yaml",
            "Preview succeeded",
        )
    } else if intent_lower.contains("terraform")
        || intent_lower.contains("hcl")
        || intent_lower.contains("opentofu")
    {
        run_validator(
            "terraform",
            &["validate", "-json"],
            code,
            "/tmp/main.tf",
            "Success",
        )
    } else if intent_lower.contains("bicep") {
        run_validator(
            "bicep",
            &["build", "/tmp/main.bicep", "--stdout"],
            code,
            "/tmp/main.bicep",
            "Build succeeded",
        )
    } else if intent_lower.contains("helm") || intent_lower.contains("helm_values") {
        // helm template . --values /tmp/values.yaml or lint
        run_validator(
            "helm",
            &["lint", "/tmp/chart", "--values", "/tmp/values.yaml"],
            code,
            "/tmp/values.yaml",
            "linted",
        )
    } else if intent_lower.contains("argocd") {
        // argocd app create --dry-run or just heuristic for YAML
        if code.contains("apiVersion: argoproj.io/v1alpha1") && code.contains("kind: Application") {
            IaCVerifierVerdict::Pass
        } else {
            IaCVerifierVerdict::Fail {
                tool: "argocd".to_string(),
                error_count: 1,
                last_error: "Missing Argo CD Application CRD fields".to_string(),
                suggestion: Some(
                    "Add 'apiVersion: argoproj.io/v1alpha1' and 'kind: Application'".to_string(),
                ),
            }
        }
    } else if intent_lower.contains("crossplane") {
        if code.contains("apiVersion: apiextensions.crossplane.io")
            || code.contains("kind: Composition")
        {
            IaCVerifierVerdict::Pass
        } else {
            IaCVerifierVerdict::Fail {
                tool: "crossplane".to_string(),
                error_count: 1,
                last_error: "Invalid Crossplane XRD/Composition".to_string(),
                suggestion: Some("Use correct Crossplane CRDs".to_string()),
            }
        }
    } else if intent_lower.contains("cdk") {
        // Would require node/ts, skip real for now
        if code.contains("cdk") || code.contains("Stack") {
            IaCVerifierVerdict::Pass
        } else {
            IaCVerifierVerdict::Fail {
                tool: "cdk".to_string(),
                error_count: 1,
                last_error: "Missing CDK constructs".to_string(),
                suggestion: None,
            }
        }
    } else {
        // Fallback mock for others
        if code.contains("resource")
            || code.contains("apiVersion")
            || code.contains("pulumi")
            || code.contains("bicep")
        {
            IaCVerifierVerdict::Pass
        } else {
            IaCVerifierVerdict::Fail {
                tool: "unknown".to_string(),
                error_count: 1,
                last_error: "No valid IaC structure detected".to_string(),
                suggestion: Some(
                    "Add standard IaC keywords like 'resource' or 'apiVersion'".to_string(),
                ),
            }
        }
    }
}

/// Main IaC generation + repair loop with emotional gating
/// This wires LanguageGateRegistry (from language_gates.rs) + frustration triggers
pub fn generate_iac_with_self_repair(
    registry: &LanguageGateRegistry,
    channels: &mut ThoughtChannels,
    prompt: &str,
    max_iterations: usize,
) -> String {
    let mut generated = String::new();
    let mut consecutive_failures = 0;

    // Detect intent and set initial gate
    let intent = if let Some(gate) = registry.detect_intent(channels) {
        gate.name.clone()
    } else {
        "general".to_string()
    };

    println!(
        "🎯 IaC Intent detected: {} (from prompt: \"{}\")",
        intent, prompt
    );

    for iter in 0..max_iterations {
        // Simulate generation (in real: call Broca model with boosted logits via CodeGate)
        generated = match intent.as_str() {
            "Kubernetes" => format!(
                "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: my-app\nspec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: my-app\n  template:\n    metadata:\n      labels:\n        app: my-app\n    spec:\n      containers:\n      - name: app\n        image: nginx:latest"
            ),
            "Ansible" => format!(
                "- hosts: all\n  become: yes\n  tasks:\n    - name: Install nginx\n      apt:\n        name: nginx\n        state: present\n    - name: Start service\n      service:\n        name: nginx\n        state: started"
            ),
            "CloudFormation" => format!(
                "AWSTemplateFormatVersion: '2010-09-09'\nDescription: S3 Bucket\nResources:\n  MyBucket:\n    Type: AWS::S3::Bucket\n    Properties:\n      BucketName: my-unique-bucket\n      VersioningConfiguration:\n        Status: Enabled"
            ),
            "Pulumi" => format!(
                "import * as pulumi from '@pulumi/pulumi';\nimport * as aws from '@pulumi/aws';\n\nconst bucket = new aws.s3.Bucket('my-bucket', {{\n  versioning: {{\n    enabled: true\n  }}\n}});\n\nexport const bucketName = bucket.id;"
            ),
            _ => format!("// {} IaC generated for: {}", intent, prompt),
        };

        // Verify with appropriate IaC tool
        let verdict = verify_iac(&generated, &intent);

        match &verdict {
            IaCVerifierVerdict::Pass => {
                println!("✅ Iteration {}: {} validation PASSED", iter + 1, intent);
                // Success → calm emotional state
                apply_frustration_trigger(
                    channels,
                    &crate::compiler_trainer::CompilerVerdict::Pass { eval_time_ms: 0 },
                    consecutive_failures,
                );
                break;
            }
            IaCVerifierVerdict::Fail {
                tool: _,
                error_count: _,
                last_error,
                suggestion,
            } => {
                println!(
                    "❌ Iteration {}: {} validation FAILED (errors: {})",
                    iter + 1,
                    intent,
                    last_error
                );
                consecutive_failures += 1;

                // Trigger frustration (updates valence/arousal)
                apply_frustration_trigger(
                    channels,
                    &crate::compiler_trainer::CompilerVerdict::Fail {
                        error: last_error.clone(),
                        stage: "iac_validation",
                    },
                    consecutive_failures,
                );

                // Modulate generation params for next try (creative mode if frustrated)
                let (temp, top_p, gate_strength) = modulate_by_emotion(channels, 0.85, 0.88, 2.0);
                println!(
                    "   -> Emotional state: arousal={:.2}, valence={:.2} | temp={:.2}, gate_boost={:.2}",
                    channels.arousal(),
                    channels.valence(),
                    temp,
                    gate_strength
                );

                if let Some(sug) = suggestion {
                    println!("   Suggestion: {}", sug);
                    // In real: inject suggestion into next prompt or repair the code
                    generated = format!("{}\n# Repair suggestion: {}", generated, sug);
                }
            }
        }
    }

    generated
}

// Example usage (for demo; in real called from CodeGate or orchestrator)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iac_repair_kubernetes() {
        let mut channels = ThoughtChannels::default();
        let tok = crate::tokenizer::BpeTokenizer::default_4k();
        let registry = LanguageGateRegistry::new(&tok);

        let result = generate_iac_with_self_repair(
            &registry,
            &mut channels,
            "create a kubernetes deployment",
            3,
        );
        assert!(result.contains("apiVersion: apps/v1") || result.contains("Deployment"));
    }
}
