use std::{
    fs,
    path::{Path, PathBuf},
    process::{self, Command},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};

use crate::{
    interoception_archive_fs::{create_new_external_file, require_external_new_file},
    interoception_github_live::{self, PromotionAuthorizationEnvelope},
    symthaea_interoception::QualificationEvidenceBundle,
};

/// Final CLI-facing authorization boundary.
///
/// The inner live verifier remains responsible for capsule, archived-gate,
/// promotion-time local-command, and exact GitHub-attempt verification. This
/// outer guard adds two properties that are intentionally difficult to retrofit
/// into historical archive code:
///
/// 1. the promotion checkout toolchain/target identity must exactly equal the
///    identity declared by the evidence capsule; and
/// 2. the only user-visible authorization artifact is created atomically with
///    create-new semantics.
pub fn authorize_promotion_strict(
    bundle_path: &Path,
    repo_root: &Path,
    evidence_root: &Path,
    local_fmt_dir: &Path,
    local_test_dir: &Path,
    local_clippy_dir: &Path,
    workspace_ci_dir: &Path,
    showroom_dir: &Path,
    out_path: &Path,
) -> Result<PromotionAuthorizationEnvelope> {
    require_external_new_file(repo_root, out_path)?;
    verify_toolchain_identity(bundle_path, repo_root)?;

    let scratch_dir = create_private_scratch_dir(out_path)?;
    let scratch_out = scratch_dir.join("provisional-envelope.json");
    let result = interoception_github_live::authorize_promotion_live(
        bundle_path,
        repo_root,
        evidence_root,
        local_fmt_dir,
        local_test_dir,
        local_clippy_dir,
        workspace_ci_dir,
        showroom_dir,
        &scratch_out,
    );

    let envelope = match result {
        Ok(envelope) => envelope,
        Err(error) => {
            let _ = fs::remove_dir_all(&scratch_dir);
            return Err(error);
        }
    };

    // Never trust or copy the provisional file. Serialize the independently
    // returned in-memory envelope and create the durable artifact with O_EXCL-like
    // create-new semantics. A concurrent writer can only make this operation fail;
    // it cannot replace an already-created authorization artifact.
    let bytes = serde_json::to_vec(&envelope).context("serialize strict promotion envelope")?;
    let write_result = create_new_external_file(repo_root, out_path, &bytes);
    let cleanup_result = fs::remove_dir_all(&scratch_dir)
        .with_context(|| format!("remove promotion scratch directory {}", scratch_dir.display()));

    write_result?;
    cleanup_result?;
    Ok(envelope)
}

fn verify_toolchain_identity(bundle_path: &Path, repo_root: &Path) -> Result<()> {
    let bytes = fs::read(bundle_path)
        .with_context(|| format!("read qualification bundle {}", bundle_path.display()))?;
    let bundle: QualificationEvidenceBundle =
        serde_json::from_slice(&bytes).context("parse qualification bundle for toolchain parity")?;

    let rustc_vv = command_text(repo_root, "rustc", &["-vV"])?;
    let cargo_vv = command_text(repo_root, "cargo", &["-Vv"])?;
    let target_triple = rustc_vv
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .context("rustc -vV did not report host target triple")?
        .trim()
        .to_string();
    let architecture = std::env::consts::ARCH.to_string();

    if rustc_vv != bundle.evidence.rustc_vv {
        bail!("promotion-time rustc -vV identity differs from evidence capsule");
    }
    if cargo_vv != bundle.evidence.cargo_vv {
        bail!("promotion-time cargo -Vv identity differs from evidence capsule");
    }
    if target_triple != bundle.evidence.target_triple {
        bail!(
            "promotion-time target triple {} differs from evidence capsule {}",
            target_triple,
            bundle.evidence.target_triple
        );
    }
    if architecture != bundle.evidence.architecture {
        bail!(
            "promotion-time architecture {} differs from evidence capsule {}",
            architecture,
            bundle.evidence.architecture
        );
    }
    Ok(())
}

fn create_private_scratch_dir(out_path: &Path) -> Result<PathBuf> {
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("clock before unix epoch while creating promotion scratch directory")?
        .as_nanos();
    let scratch = parent.join(format!(
        ".interoception-promotion-{}-{nonce}",
        process::id()
    ));

    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        builder
            .create(&scratch)
            .with_context(|| format!("create private promotion scratch directory {}", scratch.display()))?;
    }
    #[cfg(not(unix))]
    {
        fs::create_dir(&scratch)
            .with_context(|| format!("create promotion scratch directory {}", scratch.display()))?;
    }

    Ok(scratch)
}

fn command_text(repo_root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run {program} {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "command failed: {program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("toolchain identity command emitted non-UTF-8 output")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_name_is_not_the_final_output_path() {
        let out = Path::new("/tmp/final-authorization.json");
        let parent = out.parent().expect("parent");
        assert_ne!(
            parent.join(format!(".interoception-promotion-{}-0", process::id())),
            out
        );
    }
}
