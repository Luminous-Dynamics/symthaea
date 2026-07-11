use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub fn generate_manifest(root: &Path, files: &[&str]) -> anyhow::Result<()> {
    let mut manifest = BTreeMap::new();
    for file_path in files {
        let full_path = root.join(file_path);
        if full_path.exists() {
            let mut file = File::open(&full_path)?;
            let mut hasher = Sha256::new();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            hasher.update(&buffer);
            let hash = format!("{:x}", hasher.finalize());
            manifest.insert(file_path.to_string(), hash);
        }
    }

    let json = serde_json::to_string_pretty(&manifest)?;
    let mut output = File::create(root.join("manifest.json"))?;
    output.write_all(json.as_bytes())?;
    Ok(())
}
