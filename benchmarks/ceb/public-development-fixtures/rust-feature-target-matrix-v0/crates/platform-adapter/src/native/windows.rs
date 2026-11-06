//! Windows-only implementation used by the public CEB target-matrix fixture.

use std::os::windows::ffi::OsStrExt;

pub fn platform_id() -> &'static str {
    let _wide_string_method = OsStrExt::encode_wide;
    "windows"
}
