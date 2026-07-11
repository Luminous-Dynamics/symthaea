// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SPARC dataset loading: per-galaxy rotation-curve mass models (`*_rotmod.dat`)
//! joined with the galaxy metadata table (`SPARC_Lelli2016c.mrt`).
//!
//! Data source: <https://astroweb.cwru.edu/SPARC/> — fetch with
//! `scripts/download_sparc.sh` (target: `data/benchmarks/sparc/`).
//!
//! Format notes:
//! - Rotmod files are whitespace-delimited with `#` comment headers, columns:
//!   Rad (kpc), Vobs, errV, Vgas, Vdisk, Vbul (km/s), SBdisk, SBbul (L/pc²).
//!   Velocity columns are baryonic contributions at mass-to-light ratio Υ = 1.
//!   **Vgas can be negative** (central gas depressions) — downstream code must
//!   use sign-preserving quadrature, never naive squaring.
//! - The `.mrt` galaxy table is a CDS/AAS machine-readable table. **Wart,
//!   verified against the published file 2026-07-07: its data section does
//!   NOT honor the byte ranges declared in its own "Byte-by-byte Description"
//!   header** (rows are formatted wider than declared). The rows are, however,
//!   fully 0-filled — no blank fields — and every numeric column precedes the
//!   single free-text column (Ref, last). So we parse the description header
//!   only for the column *order*, then extract data fields as ordered
//!   whitespace tokens. A column shift would fail the numeric parses or the
//!   quality-range assertion in the integration test.

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// One radius sample of a galaxy rotation curve (SPARC rotmod row).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RotationPoint {
    /// Galactocentric radius [kpc]
    pub r_kpc: f64,
    /// Observed rotation velocity [km/s]
    pub v_obs: f64,
    /// Uncertainty on `v_obs` [km/s]
    pub e_v_obs: f64,
    /// Gas contribution at Υ=1 [km/s]; negative values encode central depressions
    pub v_gas: f64,
    /// Stellar disk contribution at Υ=1 [km/s]
    pub v_disk: f64,
    /// Bulge contribution at Υ=1 [km/s]
    pub v_bul: f64,
    /// Disk surface brightness [L/pc²]
    pub sb_disk: f64,
    /// Bulge surface brightness [L/pc²]
    pub sb_bul: f64,
}

/// A SPARC galaxy: metadata from the `.mrt` table + rotation curve points.
#[derive(Debug, Clone, Serialize)]
pub struct Galaxy {
    pub name: String,
    /// Distance [Mpc]
    pub distance_mpc: f64,
    /// Inclination [deg]
    pub inclination_deg: f64,
    /// Total [3.6] luminosity [10⁹ L☉]
    pub luminosity_3p6: f64,
    /// Effective surface brightness at [3.6] [L☉/pc²] — used for the
    /// low-surface-brightness extrapolation holdout
    pub sb_eff: f64,
    /// Total HI mass [10⁹ M☉] — used for the gas-fraction feature
    pub mhi_e9msun: f64,
    /// SPARC quality flag: 1 (high), 2 (medium), 3 (low)
    pub quality: u8,
    pub points: Vec<RotationPoint>,
}

/// Galaxy-level metadata parsed from `SPARC_Lelli2016c.mrt`.
#[derive(Debug, Clone, PartialEq)]
pub struct GalaxyMeta {
    pub distance_mpc: f64,
    pub inclination_deg: f64,
    pub luminosity_3p6: f64,
    pub sb_eff: f64,
    pub mhi_e9msun: f64,
    pub quality: u8,
}

/// Parse one `*_rotmod.dat` file. Returns the rotation points and the
/// distance from the `# Distance = X Mpc` header if present.
///
/// Fault-tolerant: the first three columns (radius, Vobs, errV) are
/// essential; remaining columns default to 0.0 if absent. Lines that fail to
/// parse the essential columns are skipped.
pub fn parse_rotmod(content: &str) -> (Vec<RotationPoint>, Option<f64>) {
    let mut points = Vec::new();
    let mut header_distance = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            // e.g. "# Distance = 3.16 Mpc"
            if header_distance.is_none() && comment.contains("Distance") {
                if let Some(rhs) = comment.split('=').nth(1) {
                    header_distance = rhs.split_whitespace().next().and_then(|t| t.parse().ok());
                }
            }
            continue;
        }
        let cols: Vec<f64> = line
            .split_whitespace()
            .map_while(|t| t.parse::<f64>().ok())
            .collect();
        if cols.len() < 3 {
            continue; // essential columns missing — skip line
        }
        let get = |i: usize| cols.get(i).copied().unwrap_or(0.0);
        points.push(RotationPoint {
            r_kpc: cols[0],
            v_obs: cols[1],
            e_v_obs: cols[2],
            v_gas: get(3),
            v_disk: get(4),
            v_bul: get(5),
            sb_disk: get(6),
            sb_bul: get(7),
        });
    }
    (points, header_distance)
}

/// True for CDS format specifiers: A11, I2, F6.2, E9.3, …
fn is_format_token(tok: &str) -> bool {
    let mut chars = tok.chars();
    matches!(chars.next(), Some('A' | 'I' | 'F' | 'E'))
        && chars.next().is_some_and(|c| c.is_ascii_digit())
}

/// Parse one line of the "Byte-by-byte Description" section into its column
/// label. Returns None for non-description lines.
///
/// A description line looks like `  14- 19 F6.2   Mpc  D  Distance` (byte
/// spec, format, units, label, explanation); the byte spec may also be a
/// single number (`  98 I1 ...`). We keep only the label — see the module
/// docs for why the declared byte ranges cannot be trusted for this file.
fn parse_description_label(line: &str) -> Option<String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let fmt_idx = tokens.iter().position(|t| is_format_token(t))?;
    if fmt_idx == 0 || fmt_idx + 2 >= tokens.len() {
        return None;
    }
    let bytes_spec: String = tokens[..fmt_idx].concat();
    if bytes_spec.is_empty() || !bytes_spec.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return None;
    }
    Some(tokens[fmt_idx + 2].to_string())
}

/// Parse the `SPARC_Lelli2016c.mrt` galaxy table into name → metadata.
///
/// Column order comes from the table's own byte-by-byte description header;
/// data rows (all lines after the last `----` separator) are read as ordered
/// whitespace tokens. Rows that fail to parse are skipped.
pub fn parse_mrt_metadata(content: &str) -> Result<HashMap<String, GalaxyMeta>, String> {
    let lines: Vec<&str> = content.lines().collect();

    let labels: Vec<String> = lines
        .iter()
        .filter_map(|l| parse_description_label(l))
        .collect();
    let col = |label: &str| -> Result<usize, String> {
        labels
            .iter()
            .position(|l| l == label)
            .ok_or_else(|| format!("MRT table missing column '{label}' (available: {labels:?})"))
    };
    let c_galaxy = col("Galaxy")?;
    let c_dist = col("D")?;
    let c_inc = col("Inc")?;
    let c_lum = col("L[3.6]")?;
    let c_sbeff = col("SBeff")?;
    let c_mhi = col("MHI")?;
    let c_q = col("Q")?;
    // All extracted columns must precede the free-text Ref column (if any),
    // otherwise token indices are unreliable.
    let numeric_cols = [c_dist, c_inc, c_lum, c_sbeff, c_mhi, c_q];
    if let Ok(c_ref) = col("Ref.") {
        if numeric_cols.iter().any(|&c| c > c_ref) {
            return Err("MRT numeric column appears after free-text Ref. column".to_string());
        }
    }

    // Data section: everything after the last dashed separator line.
    let last_dash = lines
        .iter()
        .rposition(|l| {
            let t = l.trim();
            t.len() >= 10 && t.chars().all(|c| c == '-')
        })
        .ok_or("MRT table has no dashed separator lines — not a CDS table?")?;

    let mut meta = HashMap::new();
    for line in &lines[last_dash + 1..] {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        // Ref (trailing text) may be absent; all needed columns must exist.
        let needed = 1 + *numeric_cols.iter().max().unwrap_or(&0).max(&c_galaxy);
        if tokens.len() < needed {
            continue;
        }
        let parsed = (|| -> Option<(String, GalaxyMeta)> {
            Some((
                tokens[c_galaxy].to_string(),
                GalaxyMeta {
                    distance_mpc: tokens[c_dist].parse().ok()?,
                    inclination_deg: tokens[c_inc].parse().ok()?,
                    luminosity_3p6: tokens[c_lum].parse().ok()?,
                    sb_eff: tokens[c_sbeff].parse().ok()?,
                    mhi_e9msun: tokens[c_mhi].parse().ok()?,
                    quality: tokens[c_q].parse().ok()?,
                },
            ))
        })();
        if let Some((name, m)) = parsed {
            meta.insert(name, m);
        }
    }
    if meta.is_empty() {
        return Err("MRT table parsed to zero galaxies".to_string());
    }
    Ok(meta)
}

/// Recursively collect `*_rotmod.dat` files under `dir` (the zip may extract
/// flat or into a subdirectory).
fn collect_rotmod_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            collect_rotmod_files(&path, out)?;
        } else if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_rotmod.dat"))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Load the full SPARC sample from a data directory containing
/// `SPARC_Lelli2016c.mrt` and the extracted rotmod files.
///
/// Galaxies without a metadata-table entry are skipped (no inclination or
/// luminosity → cannot quality-cut or compute stellar mass). Returns galaxies
/// sorted by name for determinism.
pub fn load_sparc(dir: &Path) -> Result<Vec<Galaxy>, String> {
    let mrt_path = dir.join("SPARC_Lelli2016c.mrt");
    let mrt = fs::read_to_string(&mrt_path).map_err(|e| {
        format!(
            "cannot read {}: {e} — run scripts/download_sparc.sh",
            mrt_path.display()
        )
    })?;
    let meta = parse_mrt_metadata(&mrt)?;

    let mut rotmod_files = Vec::new();
    collect_rotmod_files(dir, &mut rotmod_files)?;
    if rotmod_files.is_empty() {
        return Err(format!(
            "no *_rotmod.dat files under {} — run scripts/download_sparc.sh",
            dir.display()
        ));
    }

    let mut galaxies = Vec::new();
    for path in &rotmod_files {
        let Some(name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix("_rotmod.dat"))
        else {
            continue;
        };
        let content =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let (points, _header_distance) = parse_rotmod(&content);
        if points.is_empty() {
            continue;
        }
        let Some(m) = meta.get(name) else {
            continue;
        };
        galaxies.push(Galaxy {
            name: name.to_string(),
            distance_mpc: m.distance_mpc,
            inclination_deg: m.inclination_deg,
            luminosity_3p6: m.luminosity_3p6,
            sb_eff: m.sb_eff,
            mhi_e9msun: m.mhi_e9msun,
            quality: m.quality,
            points,
        });
    }
    galaxies.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(galaxies)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROTMOD_SAMPLE: &str = "\
# Distance = 3.16 Mpc
# Rad\tVobs\terrV\tVgas\tVdisk\tVbul\tSBdisk\tSBbul
# kpc\tkm/s\tkm/s\tkm/s\tkm/s\tkm/s\tL/pc^2\tL/pc^2
 0.16   14.2   3.2   -1.9   9.6   0.0   1163.9   0.0
 0.33   25.4   2.5    2.4  17.5   0.0    881.9   0.0
 not a data line
 0.49   35.7   2.1    5.1  24.1   0.0    668.6   0.0
";

    #[test]
    fn rotmod_parses_points_and_header_distance() {
        let (points, dist) = parse_rotmod(ROTMOD_SAMPLE);
        assert_eq!(points.len(), 3);
        assert_eq!(dist, Some(3.16));
        assert_eq!(points[0].r_kpc, 0.16);
        assert_eq!(points[0].v_obs, 14.2);
        assert_eq!(points[0].e_v_obs, 3.2);
        // Negative Vgas must survive parsing untouched
        assert_eq!(points[0].v_gas, -1.9);
        assert_eq!(points[1].sb_disk, 881.9);
        assert_eq!(points[2].v_disk, 24.1);
    }

    #[test]
    fn rotmod_tolerates_short_rows() {
        let (points, _) = parse_rotmod("1.0 50.0 2.0\n2.0 60.0\n");
        assert_eq!(points.len(), 1); // second row lacks essential errV
        assert_eq!(points[0].v_gas, 0.0); // optional columns default
    }

    #[test]
    fn description_label_variants() {
        assert_eq!(
            parse_description_label("   1- 11 A11    ---     Galaxy    Galaxy name"),
            Some("Galaxy".to_string())
        );
        assert_eq!(
            parse_description_label("  35- 41 F7.3   10+9solLum L[3.6]  Luminosity"),
            Some("L[3.6]".to_string())
        );
        assert_eq!(
            parse_description_label("  98 I1     ---     Q         Quality flag"),
            Some("Q".to_string())
        );
        assert_eq!(parse_description_label("Note (1): distances"), None);
        assert_eq!(parse_description_label("----------------"), None);
    }

    /// Mirrors the real table's structure: description declares column order;
    /// data rows are whitespace-separated (their byte alignment deliberately
    /// does NOT match the declared ranges, as in the published file).
    const MRT_SAMPLE: &str = "\
Title: test table
================================================================================
Byte-by-byte Description of file: Table1.mrt
--------------------------------------------------------------------------------
   Bytes Format Units         Label   Explanations
--------------------------------------------------------------------------------
   1- 11 A11    ---           Galaxy  Galaxy Name
  12- 13 I2     ---           T       Hubble Type
  14- 19 F6.2   Mpc           D       Distance
  20- 24 F5.2   Mpc         e_D       Mean error on D
  25- 26 I2     ---         f_D       Distance Method
  27- 30 F4.1   deg           Inc     Inclination
  31- 34 F4.1   deg         e_Inc     Mean error on Inc
  35- 41 F7.3   10+9solLum    L[3.6]  Total Luminosity at [3.6]
  42- 48 F7.3   10+9solLum  e_L[3.6]  Mean error on L[3.6]
  49- 53 F5.2   kpc           Reff    Effective Radius
  54- 61 F8.2   solLum/pc2    SBeff   Effective Surface Brightness
  62- 66 F5.2   kpc           Rdisk   Disk Scale Length
  67- 74 F8.2   solLum/pc2    SBdisk  Disk Central Surface Brightness
  75- 81 F7.3   10+9solMass   MHI     Total HI mass
  82- 86 F5.2   kpc           RHI     HI radius
  87- 91 F5.1   km/s          Vflat   Flat Rotation Velocity
  92- 96 F5.1   km/s        e_Vflat   Mean error on Vflat
  97- 99 I3     ---           Q       Quality Flag
 100-113 A14    ---           Ref.    References
--------------------------------------------------------------------------------
Note (3):
 1 = High, 2 = Medium, 3 = Low
--------------------------------------------------------------------------------
       CamB 10   3.36  0.26  2 65.0  5.0   0.075   0.003  1.21     7.89  0.47    66.20   0.012  1.21   0.0   0.0   2           Bm03
     D631-7 10   7.72  0.18  2 59.0  3.0   0.196   0.009  1.22    20.93  0.70   115.04   0.290  0.00  57.7   2.7   1      Tr09,dB01
    NoRefGx 10   4.04  0.20  2 64.0  3.0   0.053   0.002  0.65    19.99  0.37    71.26   0.275  4.96  47.0   1.0   2
";

    #[test]
    fn mrt_metadata_parses_by_declared_column_order() {
        let meta = parse_mrt_metadata(MRT_SAMPLE).unwrap();
        assert_eq!(meta.len(), 3);
        let camb = &meta["CamB"];
        assert_eq!(camb.distance_mpc, 3.36);
        assert_eq!(camb.inclination_deg, 65.0);
        assert_eq!(camb.luminosity_3p6, 0.075);
        assert_eq!(camb.sb_eff, 7.89);
        assert_eq!(camb.mhi_e9msun, 0.012);
        assert_eq!(camb.quality, 2);
        assert_eq!(meta["D631-7"].quality, 1);
        // Trailing Ref column absent → still parses (all needed cols precede it)
        assert_eq!(meta["NoRefGx"].mhi_e9msun, 0.275);
    }

    #[test]
    fn mrt_missing_column_is_reported() {
        let bad = "\
--------------------------------------------------------------------------------
   1- 11 A11    ---        Galaxy    Galaxy name
--------------------------------------------------------------------------------
NGC0024
";
        let err = parse_mrt_metadata(bad).unwrap_err();
        assert!(
            err.contains("missing column 'D'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[ignore = "requires SPARC data: run scripts/download_sparc.sh"]
    fn loads_full_sparc_sample() {
        let galaxies =
            load_sparc(&crate::test_support::sparc_data_dir()).expect("load_sparc failed");
        // SPARC has 175 galaxies; allow slack for metadata joins
        assert!(
            (160..=180).contains(&galaxies.len()),
            "expected ~175 galaxies, got {}",
            galaxies.len()
        );
        for g in &galaxies {
            assert!(
                g.points.len() >= 4,
                "{} has only {} points",
                g.name,
                g.points.len()
            );
            assert!(g.distance_mpc > 0.0, "{} has non-positive distance", g.name);
            assert!(
                (1..=3).contains(&g.quality),
                "{} has quality {}",
                g.name,
                g.quality
            );
            assert!(
                g.points.iter().all(|p| p.r_kpc > 0.0),
                "{} has r<=0",
                g.name
            );
        }
        // Spot-check a famous galaxy is present with sane values
        let n3198 = galaxies
            .iter()
            .find(|g| g.name == "NGC3198")
            .expect("NGC3198 missing");
        assert!(
            (12.0..=15.0).contains(&n3198.distance_mpc),
            "NGC3198 D={}",
            n3198.distance_mpc
        );
        assert!(n3198.quality == 1, "NGC3198 Q={}", n3198.quality);
    }
}
