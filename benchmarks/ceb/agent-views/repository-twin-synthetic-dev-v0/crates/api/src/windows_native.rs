//! Windows-only adapter compiled only with the `windows-native` feature.

pub fn os_version() -> u32 {
    // SAFETY: GetVersion has no pointer arguments and reads OS version metadata.
    unsafe { windows_sys::Win32::System::SystemInformation::GetVersion() }
}
