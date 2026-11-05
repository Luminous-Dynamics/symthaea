//! Deterministic valid ZIP fixtures used to diversify the coverage-guided corpus.
//!
//! The generator uses the pinned zip-rs writer only to create ordinary valid examples. It is
//! not an oracle for parser correctness; the production REA reader and the independently
//! configured zip-rs/zlib-rs reader still evaluate each generated seed separately.

use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

fn archive(
    files: &[(&str, &[u8], CompressionMethod)],
    directories: &[&str],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for directory in directories {
        writer.add_directory(*directory, SimpleFileOptions::default())?;
    }
    for (name, bytes, method) in files {
        let options = SimpleFileOptions::default().compression_method(*method);
        writer.start_file(*name, options)?;
        writer.write_all(bytes)?;
    }
    Ok(writer.finish()?.into_inner())
}

fn save_both(
    name: &str,
    bytes: &[u8],
    destinations: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    for directory in destinations {
        fs::create_dir_all(directory)?;
        fs::write(directory.join(format!("{name}.zip")), bytes)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let destinations = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if destinations.len() != 2 {
        return Err("usage: zip_seed_corpus <production-corpus-dir> <differential-corpus-dir>".into());
    }

    let stored = archive(
        &[("stored.txt", b"REA deterministic Stored fixture\n", CompressionMethod::Stored)],
        &[],
    )?;
    save_both("stored-single", &stored, &destinations)?;

    let deflated_payload = b"Symthaea independent Deflate fixture. ".repeat(128);
    let deflated = archive(
        &[("deflated.txt", &deflated_payload, CompressionMethod::Deflated)],
        &[],
    )?;
    save_both("deflate-repeated", &deflated, &destinations)?;

    let mixed_payload = b"mixed Stored and Deflate content ".repeat(32);
    let mixed = archive(
        &[
            ("first.bin", &(0u8..=127).collect::<Vec<_>>(), CompressionMethod::Stored),
            ("second.txt", &mixed_payload, CompressionMethod::Deflated),
        ],
        &[],
    )?;
    save_both("mixed-methods", &mixed, &destinations)?;

    let empty = archive(
        &[("empty.bin", b"", CompressionMethod::Stored)],
        &[],
    )?;
    save_both("zero-length", &empty, &destinations)?;

    let nested_payload = b"nested directory child".repeat(8);
    let nested = archive(
        &[("folder/child.txt", &nested_payload, CompressionMethod::Deflated)],
        &["folder/"],
    )?;
    save_both("directory-and-child", &nested, &destinations)?;

    let unicode_payload = b"UTF-8 member path bytes";
    let unicode = archive(
        &[("資料/évidence.txt", unicode_payload, CompressionMethod::Deflated)],
        &[],
    )?;
    save_both("utf8-member-name", &unicode, &destinations)?;

    let eocd_payload = b"payload includes these bytes: PK\x05\x06 but not an EOCD";
    let false_eocd = archive(
        &[("false-eocd.txt", eocd_payload, CompressionMethod::Stored)],
        &[],
    )?;
    save_both("eocd-signature-in-payload", &false_eocd, &destinations)?;

    Ok(())
}
