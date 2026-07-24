// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Aggregate authenticated studies into a private portfolio and suppressed public report.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationEvidenceBundle, CalibrationStudyPortfolioPolicy, audit_calibration_study_portfolio,
    build_calibration_study_portfolio,
};

#[derive(Debug)]
struct Arguments {
    portfolio_id: String,
    bundles: Vec<PathBuf>,
    write_private: PathBuf,
    write_public: PathBuf,
    require_accepted: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments()?;
    if lexical_absolute(&arguments.write_private)? == lexical_absolute(&arguments.write_public)? {
        return Err(invalid_input("private portfolio and public report paths must differ").into());
    }
    let bundles = arguments
        .bundles
        .iter()
        .map(|path| read_json::<CalibrationEvidenceBundle>(path))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let portfolio = build_calibration_study_portfolio(
        arguments.portfolio_id,
        &bundles,
        CalibrationStudyPortfolioPolicy::release(),
    )?;
    let audit = audit_calibration_study_portfolio(&portfolio);
    if !audit.valid() {
        return Err(invalid_input(format!(
            "portfolio failed audit with {} issues",
            audit.issues.len()
        ))
        .into());
    }
    atomic_json(&arguments.write_private, &portfolio)?;
    atomic_json(&arguments.write_public, &portfolio.public_report)?;
    if arguments.require_accepted && !portfolio.decision.accepted {
        return Err(invalid_input("portfolio did not pass the release policy").into());
    }
    Ok(())
}

fn parse_arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut portfolio_id = None;
    let mut bundles = Vec::new();
    let mut write_private = None;
    let mut write_public = None;
    let mut require_accepted = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--portfolio-id" => portfolio_id = Some(next_value(&mut arguments, &argument)?),
            "--bundle" => bundles.push(PathBuf::from(next_value(&mut arguments, &argument)?)),
            "--write-private" => {
                write_private = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
            }
            "--write-public" => {
                write_public = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
            }
            "--require-accepted" => require_accepted = true,
            "--help" | "-h" => {
                println!(
                    "usage: evidence_study_portfolio --portfolio-id ID --bundle PATH --bundle PATH [...] --write-private PATH --write-public PATH [--require-accepted]"
                );
                std::process::exit(0);
            }
            other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
        }
    }
    if bundles.is_empty() {
        return Err(invalid_input("at least one --bundle is required").into());
    }
    Ok(Arguments {
        portfolio_id: portfolio_id.ok_or_else(|| invalid_input("--portfolio-id is required"))?,
        bundles,
        write_private: write_private.ok_or_else(|| invalid_input("--write-private is required"))?,
        write_public: write_public.ok_or_else(|| invalid_input("--write-public is required"))?,
        require_accepted,
    })
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")).into())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn atomic_json(path: &Path, value: &impl serde::Serialize) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        error
    })?;
    Ok(())
}

fn lexical_absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    use std::path::Component;

    let source = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in source.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
