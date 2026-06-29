// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Golden-reference Terraform HCL snippets for distillation.

pub fn hcl_golden_for(prompt: &str) -> Option<&'static str> {
    match prompt.to_lowercase().as_str() {
        "create a basic s3 bucket" => Some(S3_BASIC),
        "s3 bucket with versioning" => Some(S3_VERSIONED),
        "standard t2.micro ec2 instance" => Some(EC2_BASIC),
        _ => None,
    }
}

pub const HCL_HARVEST_PROMPTS: &[&str] = &[
    "create a basic s3 bucket",
    "s3 bucket with versioning",
    "standard t2.micro ec2 instance",
];

const S3_BASIC: &str = r#"resource "aws_s3_bucket" "example" {
  bucket = "my-tf-test-bucket"
  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }
}
"#;

const S3_VERSIONED: &str = r#"resource "aws_s3_bucket" "example" {
  bucket = "my-tf-test-bucket"
}

resource "aws_s3_bucket_versioning" "example" {
  bucket = aws_s3_bucket.example.id
  versioning_configuration {
    status = "Enabled"
  }
}
"#;

const EC2_BASIC: &str = r#"resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "HelloWorld"
  }
}
"#;
