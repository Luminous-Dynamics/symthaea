use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
};

use anyhow::{bail, Context, Result};

pub fn validate_relative_path(name: &str, value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.trim().is_empty() || path.is_absolute() {
        bail!("{name} must be a non-empty relative path");
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            bail!("{name} contains a disallowed path component: {value}");
        }
    }
    Ok(())
}

pub fn canonical_closed_root(root: &Path) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("inspect archive root {}", root.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("archive root must not be a symlink: {}", root.display());
    }
    if !metadata.is_dir() {
        bail!("archive root is not a directory: {}", root.display());
    }
    fs::canonicalize(root).with_context(|| format!("canonicalize archive root {}", root.display()))
}

pub fn closed_relative_file(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_relative_path("archive object path", relative)?;
    let canonical_root = canonical_closed_root(root)?;

    let mut current = root.to_path_buf();
    for component in Path::new(relative).components() {
        let Component::Normal(name) = component else {
            unreachable!("relative path was validated above")
        };
        current.push(name);
        let metadata = fs::symlink_metadata(&current)
            .with_context(|| format!("inspect archive object component {}", current.display()))?;
        if metadata.file_type().is_symlink() {
            bail!("archive objects must not traverse symlinks: {}", current.display());
        }
    }

    let metadata = fs::symlink_metadata(&current)
        .with_context(|| format!("inspect archive object {}", current.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "archive object is not a regular non-symlink file: {}",
            current.display()
        );
    }

    let canonical_object = fs::canonicalize(&current)
        .with_context(|| format!("canonicalize archive object {}", current.display()))?;
    if !canonical_object.starts_with(&canonical_root) {
        bail!(
            "archive object escapes canonical archive root: object={} root={}",
            canonical_object.display(),
            canonical_root.display()
        );
    }
    Ok(current)
}

pub fn require_external_directory(repo_root: &Path, dir: &Path) -> Result<()> {
    let canonical_repo = canonical_non_symlink_directory(repo_root, "source checkout")?;
    let canonical_dir = canonical_non_symlink_directory(dir, "qualification archive")?;
    if canonical_dir.starts_with(&canonical_repo) {
        bail!(
            "qualification evidence/archive directory must live outside the target source checkout: {}",
            dir.display()
        );
    }
    Ok(())
}

pub fn require_external_new_file(repo_root: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        bail!("refusing to overwrite existing output file: {}", out.display());
    }
    let canonical_repo = canonical_non_symlink_directory(repo_root, "source checkout")?;
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let canonical_parent = canonical_non_symlink_directory(parent, "promotion output parent")?;
    if canonical_parent.starts_with(&canonical_repo) {
        bail!(
            "promotion authorization output must live outside the target source checkout: {}",
            out.display()
        );
    }
    Ok(())
}

/// Create a new qualification artifact without a check-then-write race.
///
/// `create_new(true)` is the authoritative no-overwrite guard. The preceding
/// path-policy check exists to enforce source-checkout separation, not to provide
/// mutual exclusion. Bytes are flushed to the file before success is returned.
pub fn create_new_external_file(repo_root: &Path, out: &Path, bytes: &[u8]) -> Result<()> {
    require_external_new_file(repo_root, out)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .with_context(|| format!("create new qualification output {}", out.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write qualification output {}", out.display()))?;
    file.sync_all()
        .with_context(|| format!("sync qualification output {}", out.display()))?;
    Ok(())
}

fn canonical_non_symlink_directory(path: &Path, role: &str) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect {role} {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("{role} must not be a symlink: {}", path.display());
    }
    if !metadata.is_dir() {
        bail!("{role} is not a directory: {}", path.display());
    }
    fs::canonicalize(path).with_context(|| format!("canonicalize {role} {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs, process,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("symthaea-{label}-{}-{nonce}", process::id()))
    }

    #[test]
    fn rejects_parent_and_absolute_paths() {
        assert!(validate_relative_path("test", "../escape").is_err());
        assert!(validate_relative_path("test", "/absolute").is_err());
        assert!(validate_relative_path("test", "safe/object.json").is_ok());
    }

    #[test]
    fn create_new_external_file_never_replaces_existing_bytes() {
        let repo = unique_dir("source-checkout");
        let outside = unique_dir("promotion-out");
        fs::create_dir_all(&repo).expect("create repo");
        fs::create_dir_all(&outside).expect("create outside");
        let out = outside.join("authorization.json");

        create_new_external_file(&repo, &out, b"first").expect("create first artifact");
        assert_eq!(fs::read(&out).expect("read first artifact"), b"first");

        assert!(create_new_external_file(&repo, &out, b"second").is_err());
        assert_eq!(fs::read(&out).expect("read preserved artifact"), b"first");

        let _ = fs::remove_dir_all(&repo);
        let _ = fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = unique_dir("closed-root");
        let outside = unique_dir("outside");
        fs::create_dir_all(&root).expect("create root");
        fs::create_dir_all(&outside).expect("create outside");
        fs::write(outside.join("secret.bin"), b"secret").expect("write outside file");
        symlink(&outside, root.join("linked")).expect("create symlink");

        let result = closed_relative_file(&root, "linked/secret.bin");
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_root() {
        use std::os::unix::fs::symlink;

        let target = unique_dir("target-root");
        let link = unique_dir("root-link");
        fs::create_dir_all(&target).expect("create target");
        symlink(&target, &link).expect("create root symlink");
        assert!(canonical_closed_root(&link).is_err());

        let _ = fs::remove_file(&link);
        let _ = fs::remove_dir_all(&target);
    }
}
