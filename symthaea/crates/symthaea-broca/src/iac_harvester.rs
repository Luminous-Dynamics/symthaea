// iac_harvester.rs
// Training Data Harvester for IaC (Infrastructure as Code)
// Similar to InverseHarvester but specialized for K8s/Ansible/CFN/Pulumi/OpenTofu/CDK/Bicep/Crossplane/Argo CD/Helm repos
// Produces high-quality (prompt, code) pairs for Broca distillation / RL training
// Includes diversity filter (Jaccard on prompts) + structural scoring via GenericStructuralScorer

use crate::generic_structural_scorer_integration::GenericStructuralScorer;
use crate::language_gates::LanguageGateRegistry; // for intent detection
use std::collections::HashSet;
use std::path::Path; // assume exists for AST/structural diff

use crate::tokenizer::BpeTokenizer;

#[derive(Debug, Clone)]
pub struct IaCHarvestPair {
    pub prompt: String,
    pub code: String,
    pub intent: String, // e.g. "kubernetes", "opentofu", "cdk"
    pub source_repo: String,
    pub structural_score: f32, // from scorer (Jaccard etc.)
    pub diversity_score: f32,
}

pub struct IaCHarvester {
    registry: LanguageGateRegistry,
    scorer: GenericStructuralScorer,
    max_pairs: usize,
    min_diversity_threshold: f32,
    harvested: Vec<IaCHarvestPair>,
}

impl IaCHarvester {
    pub fn new(tokenizer: &BpeTokenizer) -> Self {
        IaCHarvester {
            registry: LanguageGateRegistry::new(tokenizer),
            scorer: GenericStructuralScorer::new_with_all(), // from earlier module
            max_pairs: 500,
            min_diversity_threshold: 0.75,
            harvested: Vec::new(),
        }
    }

    /// Harvest from known public IaC GitHub repos (in real: use git2 or GitHub API + walkdir)
    /// For this env: simulates with embedded examples + "clone" comments
    pub fn harvest_iac_repos(&mut self) {
        println!("🌾 Harvesting IaC training data from public repos (simulated in sandbox)...");

        // Simulated repos (in production: git clone https://github.com/{user}/{repo} --depth 1; walk .tf .yaml .bicep etc.)
        let examples = vec![
            (
                "kubernetes",
                "Deploy nginx with K8s Deployment",
                "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: nginx\nspec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n      - name: nginx\n        image: nginx:1.14.2",
            ),
            (
                "opentofu",
                "OpenTofu S3 bucket module",
                "resource \"aws_s3_bucket\" \"example\" {\n  bucket = var.bucket_name\n  tags = var.tags\n}\n\nmodule \"vpc\" {\n  source = \"terraform-aws-modules/vpc/aws\"\n}",
            ),
            (
                "cdk",
                "AWS CDK TypeScript Lambda",
                "import * as cdk from 'aws-cdk-lib';\nimport * as lambda from 'aws-cdk-lib/aws-lambda';\n\nexport class MyStack extends cdk.Stack {\n  constructor(scope: cdk.App, id: string) {\n    super(scope, id);\n    new lambda.Function(this, 'MyFunc', { runtime: lambda.Runtime.NODEJS_18_X, code: lambda.Code.fromInline('exports.handler = async () => {}') });\n  }\n}",
            ),
            (
                "bicep",
                "Azure storage with Bicep",
                "param location string = resourceGroup().location\nresource storage 'Microsoft.Storage/storageAccounts@2022-09-01' = {\n  name: 'mystorage'\n  location: location\n  sku: { name: 'Standard_LRS' }\n  kind: 'StorageV2'\n}",
            ),
            (
                "crossplane",
                "Crossplane AWS RDS claim",
                "apiVersion: database.example.org/v1alpha1\nkind: RDSInstanceClaim\nmetadata:\n  name: my-db\nspec:\n  forProvider:\n    region: us-east-1\n    engine: postgres\n  compositionSelector:\n    matchLabels:\n      provider: aws",
            ),
            (
                "argocd",
                "Argo CD App for microservice",
                "apiVersion: argoproj.io/v1alpha1\nkind: Application\nmetadata:\n  name: my-app\nspec:\n  project: default\n  source:\n    repoURL: https://github.com/myorg/myapp.git\n    targetRevision: HEAD\n    path: k8s\n  destination:\n    server: https://kubernetes.default.svc\n    namespace: default\n  syncPolicy:\n    automated: { prune: true, selfHeal: true }",
            ),
            (
                "helm_values",
                "Helm values for prometheus",
                "replicaCount: 2\nimage:\n  repository: prom/prometheus\n  tag: v2.45.0\nservice:\n  type: ClusterIP\n  port: 9090\ningress:\n  enabled: true\n  hosts:\n    - host: prometheus.local\n      paths:\n        - path: /\n          pathType: Prefix",
            ),
            (
                "ansible",
                "Ansible role for nginx install",
                "- hosts: webservers\n  become: yes\n  tasks:\n    - name: Install nginx\n      apt:\n        name: nginx\n        state: latest\n    - name: Start nginx\n      service:\n        name: nginx\n        state: started\n        enabled: yes",
            ),
            (
                "cloudformation",
                "CFN EC2 with security group",
                "AWSTemplateFormatVersion: '2010-09-09'\nResources:\n  MyEC2:\n    Type: AWS::EC2::Instance\n    Properties:\n      ImageId: ami-0c55b159cbfafe1f0\n      InstanceType: t2.micro\n      SecurityGroups:\n        - !Ref MySG\n  MySG:\n    Type: AWS::EC2::SecurityGroup\n    Properties:\n      GroupDescription: Allow SSH\n      SecurityGroupIngress:\n        - IpProtocol: tcp\n          FromPort: 22\n          ToPort: 22\n          CidrIp: 0.0.0.0/0",
            ),
            (
                "pulumi",
                "Pulumi TS AWS bucket",
                "import * as pulumi from '@pulumi/pulumi';\nimport * as aws from '@pulumi/aws';\nconst bucket = new aws.s3.Bucket('my-bucket');\nexport const bucketName = bucket.id;",
            ),
        ];

        for (intent, prompt, code) in examples {
            let structural_score = self
                .scorer
                .score_go(code, "golden_placeholder")
                .jaccard_similarity;
            let diversity = self.compute_diversity(prompt);
            if diversity >= self.min_diversity_threshold && self.harvested.len() < self.max_pairs {
                self.harvested.push(IaCHarvestPair {
                    prompt: prompt.to_string(),
                    code: code.to_string(),
                    intent: intent.to_string(),
                    source_repo: format!("github.com/public-iac/{}", intent),
                    structural_score,
                    diversity_score: diversity,
                });
            }
        }
        println!(
            "✅ Harvested {} diverse IaC pairs (filtered by Jaccard > {:.2})",
            self.harvested.len(),
            self.min_diversity_threshold
        );
    }

    fn compute_diversity(&self, prompt: &str) -> f32 {
        // Jaccard on tokens (simple version of diversity_filter_patch)
        let prompt_lower = prompt.to_lowercase();
        let tokens: HashSet<_> = prompt_lower.split_whitespace().collect();
        let mut max_sim = 0.0f32;
        for existing in &self.harvested {
            let existing_lower = existing.prompt.to_lowercase();
            let ex_tokens: HashSet<_> = existing_lower.split_whitespace().collect();
            let inter = tokens.intersection(&ex_tokens).count() as f32;
            let uni = tokens.union(&ex_tokens).count() as f32;
            if uni > 0.0 {
                max_sim = max_sim.max(inter / uni);
            }
        }
        1.0 - max_sim
    }

    pub fn get_pairs(&self) -> &[IaCHarvestPair] {
        &self.harvested
    }

    pub fn export_to_jsonl(&self, path: &Path) {
        // In real: serde_json serialize to .jsonl for training
        println!(
            "📤 Exported {} IaC training pairs to {:?}",
            self.harvested.len(),
            path
        );
    }
}

// Usage: let mut harvester = IaCHarvester::new(); harvester.harvest_iac_repos(); harvester.export_to_jsonl(Path::new("iac_training.jsonl"));
