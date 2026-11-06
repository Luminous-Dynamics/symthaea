//! Unix-only implementation used by the public CEB target-matrix fixture.

use std::os::unix::fs::MetadataExt;

pub fn platform_id() -> &'static str {
    let _metadata_method = MetadataExt::len;
    "unix"
}
