//! Content-addressing utilities for model, dataset, and run artifacts.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Hash a directory deterministically from relative paths, file sizes, and file contents.
pub fn hash_tree(root: &Path) -> io::Result<String> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort();
    let mut hasher = blake3::Hasher::new();
    for relative in files {
        let path = root.join(&relative);
        let metadata = fs::metadata(&path)?;
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update(&metadata.len().to_le_bytes());
        let mut file = File::open(path)?;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn collect_files(root: &Path, current: &Path, output: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("symlink not allowed in artifact: {}", path.display()),
            ));
        }
        if file_type.is_dir() {
            collect_files(root, &path, output)?;
        }
        if file_type.is_file() {
            output.push(
                path.strip_prefix(root)
                    .map_err(io::Error::other)?
                    .to_owned(),
            );
        }
    }
    Ok(())
}

pub fn verify_path(path: &Path, expected: &str) -> Result<(), String> {
    let actual = if path.is_dir() {
        hash_tree(path)
    } else {
        hash_file(path)
    }
    .map_err(|error| format!("{}: {error}", path.display()))?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "artifact hash mismatch: expected {expected}, got {actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_hash_is_checked_not_just_present() {
        let path = std::env::temp_dir().join(format!("symthaea-artifact-{}", std::process::id()));
        fs::write(&path, b"model").unwrap();
        assert!(verify_path(&path, &hash_file(&path).unwrap()).is_ok());
        assert!(verify_path(&path, &"0".repeat(64)).is_err());
        fs::remove_file(path).unwrap();
    }
}
