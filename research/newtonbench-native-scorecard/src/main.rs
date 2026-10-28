use newtonbench_native_scorecard::{audit, ArtifactInput};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("newtonbench-native-scorecard: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(
            concat!(
                "usage: newtonbench-native-scorecard MANIFEST.json ",
                "VANILLA_TRIALS_DIR CODE_ASSISTED_TRIALS_DIR"
            )
            .into(),
        );
    }

    let manifest = fs::read(&args[0])?;
    let vanilla_root = Path::new(&args[1]);
    let tool_root = Path::new(&args[2]);
    let vanilla = collect_trial_artifacts(vanilla_root)?;
    let tool = collect_trial_artifacts(tool_root)?;
    let report = audit(&manifest, vanilla, tool).map_err(io::Error::other)?;
    if !report.verify_receipt() {
        return Err("generated scorecard receipt did not verify".into());
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn collect_trial_artifacts(root: &Path) -> io::Result<Vec<ArtifactInput>> {
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "trial input root must be a real directory, not a symlink: {}",
                root.display()
            ),
        ));
    }
    let mut paths = Vec::new();
    collect_recursively(root, &mut paths)?;
    paths.sort();
    let mut artifacts = Vec::with_capacity(paths.len());
    for path in paths {
        let relative: PathBuf = path
            .strip_prefix(root)
            .map_err(|error| io::Error::other(error.to_string()))?
            .to_owned();
        let relative_path = relative.to_string_lossy().replace('\\', "/");
        artifacts.push(ArtifactInput {
            path: relative_path,
            raw_bytes: fs::read(path)?,
        });
    }
    Ok(artifacts)
}

fn collect_recursively(current: &Path, found: &mut Vec<PathBuf>) -> io::Result<()> {
    let mut entries = fs::read_dir(current)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("refusing symlink inside trial artifact directory: {}", path.display()),
            ));
        }
        if file_type.is_dir() {
            collect_recursively(&path, found)?;
        } else if file_type.is_file() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("trial") && path.extension().is_some_and(|ext| ext == "json") {
                found.push(path);
            }
        }
    }
    Ok(())
}
