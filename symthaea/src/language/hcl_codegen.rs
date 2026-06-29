// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Terraform HCL code generator (#4 of the "make this even better"
//! list — substrate-independence at the generator level).
//!
//! Mirrors the Nix and Compose generator patterns: prompt → intent
//! classification → idiom library → HCL emission. The HCL parser that
//! validates output lives in `hcl_scorer.rs`.

/// Detected HCL intent from prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HclIntent {
    /// S3 bucket resource.
    S3Bucket,
    /// EC2 instance resource.
    Ec2Instance,
    /// Full VPC stack.
    Vpc,
    /// Unknown / fallthrough.
    Generic,
}

/// Result of HCL code generation.
#[derive(Debug, Clone)]
pub struct HclGenResult {
    pub prompt: String,
    pub intent: HclIntent,
    pub code: String,
    pub source: HclSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HclSource {
    /// Matched an idiom in the library.
    Idiom,
    /// Fell through to a skeleton.
    Skeleton,
}

/// Top-level entrypoint: natural-language prompt → HCL.
pub fn generate_hcl(prompt: &str) -> HclGenResult {
    let lower = prompt.to_lowercase();
    let intent = classify_hcl_intent(&lower);
    let idiom = hcl_idiom_body(&lower);
    let source = if idiom.is_some() {
        HclSource::Idiom
    } else {
        HclSource::Skeleton
    };
    let code = idiom.unwrap_or_else(skeleton_hcl);
    HclGenResult {
        prompt: prompt.to_string(),
        intent,
        code,
        source,
    }
}

fn classify_hcl_intent(lower: &str) -> HclIntent {
    if lower.contains("s3") || lower.contains("bucket") {
        HclIntent::S3Bucket
    } else if lower.contains("ec2") || lower.contains("instance") {
        HclIntent::Ec2Instance
    } else if lower.contains("vpc") || lower.contains("network") {
        HclIntent::Vpc
    } else {
        HclIntent::Generic
    }
}

pub fn hcl_idiom_body(lower: &str) -> Option<String> {
    if lower.contains("s3") && lower.contains("versioning") {
        return Some(emit_s3_bucket_versioned());
    }
    if lower.contains("s3") || lower.contains("bucket") {
        return Some(emit_s3_bucket_basic());
    }
    if lower.contains("ec2") || lower.contains("instance") {
        return Some(emit_ec2_basic());
    }
    None
}

fn skeleton_hcl() -> String {
    "resource \"null_resource\" \"example\" {}\n".to_string()
}

fn emit_s3_bucket_basic() -> String {
    r#"resource "aws_s3_bucket" "example" {
  bucket = "my-tf-test-bucket"
  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }
}
"#
    .to_string()
}

fn emit_s3_bucket_versioned() -> String {
    r#"resource "aws_s3_bucket" "example" {
  bucket = "my-tf-test-bucket"
}

resource "aws_s3_bucket_versioning" "example" {
  bucket = aws_s3_bucket.example.id
  versioning_configuration {
    status = "Enabled"
  }
}
"#
    .to_string()
}

fn emit_ec2_basic() -> String {
    r#"resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "HelloWorld"
  }
}
"#
    .to_string()
}
