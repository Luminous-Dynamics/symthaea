use std::process::Command;

fn run_git(args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("cannot execute git {args:?}: {error}"));
    if !output.status.success() {
        panic!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("git {args:?} returned non-UTF-8 output: {error}"))
        .trim()
        .to_owned()
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn main() {
    println!("cargo:rerun-if-env-changed=SBR_GENERATOR_SOURCE_REVISION");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    let checked_out_revision = run_git(&["rev-parse", "HEAD"]);
    if !valid_revision(&checked_out_revision) {
        panic!("checked-out generator revision is not a full lowercase Git SHA");
    }

    let dirty = run_git(&["status", "--porcelain", "--untracked-files=all"]);
    let worktree_clean = dirty.is_empty();
    if !worktree_clean {
        println!(
            "cargo:warning=dirty worktree; unit tests may run, but qualified task generation will be disabled"
        );
    }

    if let Ok(declared_revision) = std::env::var("SBR_GENERATOR_SOURCE_REVISION") {
        if declared_revision != checked_out_revision {
            panic!(
                "SBR_GENERATOR_SOURCE_REVISION ({declared_revision}) does not match git HEAD ({checked_out_revision})"
            );
        }
    }

    println!("cargo:rustc-env=SBR_GENERATOR_SOURCE_REVISION={checked_out_revision}");
    println!(
        "cargo:rustc-env=SBR_GENERATOR_WORKTREE_CLEAN={}",
        if worktree_clean { "true" } else { "false" }
    );
}
