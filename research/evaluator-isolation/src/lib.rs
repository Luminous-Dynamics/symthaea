//! Fail-closed Linux intake for candidate-written benchmark artifacts.
//!
//! The caller supplies evaluator-owned exchange and private directories.
//! Candidate filenames are treated as untrusted single path components.
//! The source file is opened beneath the exchange directory using openat2,
//! then copied from that same open descriptor into a newly-created evaluator-
//! private file. SHA-256 is computed over the exact bytes written to the
//! private copy. The evaluator must parse/score only that private copy.
//!
//! There is intentionally no openat()/canonicalize() fallback when openat2
//! is unavailable. ENOSYS and unsupported kernel/filesystem behavior fail
//! closed. This library does not establish process isolation; see #9437/#9621.

#[cfg(not(target_os = "linux"))]
compile_error!("symthaea-evaluator-isolation requires Linux openat2");

use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
const RESOLVE_NO_SYMLINKS: u64 = 0x04;
const RESOLVE_BENEATH: u64 = 0x08;
const COPY_BUFFER_BYTES: usize = 64 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedArtifact {
    pub private_copy: PathBuf,
    pub sha256: [u8; 32],
    pub bytes: u64,
    pub source_device: u64,
    pub source_inode: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntakeError {
    InvalidDirectoryPath,
    InvalidFileName,
    OpenAt2Unavailable,
    OpenAt2(io::ErrorKind),
    Io(io::ErrorKind),
    SourceNotRegular,
    SourceHasMultipleLinks { links: u64 },
    SourceTooLarge { size: u64, maximum: u64 },
    SourceChangedDuringCopy,
    CopyLengthMismatch { expected: u64, actual: u64 },
    ArtifactTooLarge { actual: u64, maximum: u64 },
    SizeOverflow,
}

impl fmt::Display for IntakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDirectoryPath => write!(f, "intake directory must be an absolute path"),
            Self::InvalidFileName => write!(f, "artifact name must be one normal path component"),
            Self::OpenAt2Unavailable => write!(
                f,
                "openat2 is unavailable; refusing an insecure path-resolution fallback"
            ),
            Self::OpenAt2(kind) => write!(f, "openat2 failed: {kind:?}"),
            Self::Io(kind) => write!(f, "artifact intake I/O failed: {kind:?}"),
            Self::SourceNotRegular => write!(f, "candidate artifact is not a regular file"),
            Self::SourceHasMultipleLinks { links } => {
                write!(f, "candidate artifact has link count {links}; expected exactly one")
            }
            Self::SourceTooLarge { size, maximum } => {
                write!(f, "candidate artifact size {size} exceeds maximum {maximum}")
            }
            Self::SourceChangedDuringCopy => {
                write!(f, "candidate artifact metadata changed during private intake")
            }
            Self::CopyLengthMismatch { expected, actual } => write!(
                f,
                "candidate artifact copy length {actual} differs from initial size {expected}"
            ),
            Self::ArtifactTooLarge { actual, maximum } => {
                write!(f, "candidate artifact grew to {actual} bytes; maximum is {maximum}")
            }
            Self::SizeOverflow => write!(f, "candidate artifact length overflowed"),
        }
    }
}

impl std::error::Error for IntakeError {}

/// Import one candidate artifact into an evaluator-owned private directory.
///
/// Directory paths are fixed by evaluator configuration, not candidate data.
/// Source and destination names are checked as single components. The copied
/// file is created with O_EXCL and mode 0600. The caller must parse only the
/// returned private_copy and bind sha256 into its evaluator receipt.
pub fn import_candidate_artifact(
    exchange_directory: &Path,
    candidate_file_name: &str,
    evaluator_private_directory: &Path,
    private_file_name: &str,
    maximum_bytes: u64,
) -> Result<ImportedArtifact, IntakeError> {
    let source_name = single_component(candidate_file_name)?;
    let destination_name = single_component(private_file_name)?;

    let exchange_dir = open_absolute_directory(exchange_directory)?;
    let private_dir = open_absolute_directory(evaluator_private_directory)?;

    let source_fd = openat2_beneath(
        exchange_dir.as_raw_fd(),
        &source_name,
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        0,
        RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    )?;
    // SAFETY: openat2 returned a new owned file descriptor.
    let mut source = unsafe { File::from_raw_fd(source_fd) };

    let before = source.metadata().map_err(map_io_error)?;
    if !before.file_type().is_file() {
        return Err(IntakeError::SourceNotRegular);
    }
    if before.nlink() != 1 {
        return Err(IntakeError::SourceHasMultipleLinks { links: before.nlink() });
    }
    if before.len() > maximum_bytes {
        return Err(IntakeError::SourceTooLarge {
            size: before.len(),
            maximum: maximum_bytes,
        });
    }

    let destination_fd = openat2_beneath(
        private_dir.as_raw_fd(),
        &destination_name,
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        0o600,
        RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    )?;
    // SAFETY: openat2 returned a new owned file descriptor.
    let mut destination = unsafe { File::from_raw_fd(destination_fd) };

    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    let mut copied = 0_u64;

    let copy_result = (|| -> Result<(), IntakeError> {
        loop {
            let read = source.read(&mut buffer).map_err(map_io_error)?;
            if read == 0 {
                break;
            }
            copied = copied
                .checked_add(u64::try_from(read).map_err(|_| IntakeError::SizeOverflow)?)
                .ok_or(IntakeError::SizeOverflow)?;
            if copied > maximum_bytes {
                return Err(IntakeError::ArtifactTooLarge {
                    actual: copied,
                    maximum: maximum_bytes,
                });
            }
            destination
                .write_all(&buffer[..read])
                .map_err(map_io_error)?;
            hasher.update(&buffer[..read]);
        }

        if copied != before.len() {
            return Err(IntakeError::CopyLengthMismatch {
                expected: before.len(),
                actual: copied,
            });
        }

        let after = source.metadata().map_err(map_open_error)?;
        if before.dev() != after.dev()
            || before.ino() != after.ino()
            || before.nlink() != after.nlink()
            || before.len() != after.len()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
        {
            return Err(IntakeError::SourceChangedDuringCopy);
        }

        destination.sync_all().map_err(map_io_error)?;
        Ok(())
    })();

    if let Err(error) = copy_result {
        drop(destination);
        let _ = fs::remove_file(evaluator_private_directory.join(private_file_name));
        return Err(error);
    }

    Ok(ImportedArtifact {
        private_copy: evaluator_private_directory.join(private_file_name),
        sha256: hasher.finalize().into(),
        bytes: copied,
        source_device: before.dev(),
        source_inode: before.ino(),
    })
}

fn single_component(value: &str) -> Result<CString, IntakeError> {
    let path = Path::new(value);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) if !value.is_empty() => {
            CString::new(value.as_bytes()).map_err(|_| IntakeError::InvalidFileName)
        }
        _ => Err(IntakeError::InvalidFileName),
    }
}

fn open_absolute_directory(path: &Path) -> Result<File, IntakeError> {
    if !path.is_absolute() {
        return Err(IntakeError::InvalidDirectoryPath);
    }
    let raw = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| IntakeError::InvalidDirectoryPath)?;
    let fd = openat2_raw(
        libc::AT_FDCWD,
        &raw,
        libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC,
        0,
        RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    )?;
    // SAFETY: openat2 returned a new owned directory descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn openat2_beneath(
    directory_fd: i32,
    name: &CString,
    flags: i32,
    mode: u32,
    resolve: u64,
) -> Result<i32, IntakeError> {
    openat2_raw(directory_fd, name, flags, mode, resolve)
}

fn openat2_raw(
    directory_fd: i32,
    path: &CString,
    flags: i32,
    mode: u32,
    resolve: u64,
) -> Result<i32, IntakeError> {
    let how = OpenHow {
        flags: u64::try_from(flags).map_err(|_| IntakeError::SizeOverflow)?,
        mode: u64::from(mode),
        resolve,
    };
    // SAFETY: the kernel reads a repr(C) open_how and NUL-terminated path
    // only for the duration of the syscall.
    let result = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            directory_fd,
            path.as_ptr(),
            &how as *const OpenHow,
            std::mem::size_of::<OpenHow>(),
        )
    };
    if result >= 0 {
        return i32::try_from(result).map_err(|_| IntakeError::SizeOverflow);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ENOSYS) {
        return Err(IntakeError::OpenAt2Unavailable);
    }
    Err(IntakeError::OpenAt2(error.kind()))
}

fn map_io_error(error: io::Error) -> IntakeError {
    IntakeError::Io(error.kind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{hard_link, symlink};

    #[test]
    fn imports_regular_candidate_file_and_hashes_exact_private_copy() {
        let root = tempfile::tempdir().unwrap();
        let exchange = root.path().join("exchange");
        let private = root.path().join("private");
        fs::create_dir(&exchange).unwrap();
        fs::create_dir(&private).unwrap();
        fs::write(exchange.join("prediction.bin"), b"candidate-prediction").unwrap();

        let imported = import_candidate_artifact(
            &exchange,
            "prediction.bin",
            &private,
            "run-1.prediction",
            1_024,
        )
        .unwrap();
        let copied = fs::read(imported.private_copy).unwrap();
        assert_eq!(copied, b"candidate-prediction");
        assert_eq!(imported.bytes, u64::try_from(copied.len()).unwrap());
        assert_eq!(imported.sha256, Sha256::digest(&copied).into());
    }

    #[test]
    fn rejects_path_traversal_and_multi_component_names() {
        for name in ["../oracle.bin", "/etc/passwd", "subdir/file", ".", ".."] {
            assert!(matches!(single_component(name), Err(IntakeError::InvalidFileName)));
        }
    }

    #[test]
    fn rejects_symlink_sources() {
        let root = tempfile::tempdir().unwrap();
        let exchange = root.path().join("exchange");
        let private = root.path().join("private");
        fs::create_dir(&exchange).unwrap();
        fs::create_dir(&private).unwrap();
        fs::write(root.path().join("secret"), b"secret").unwrap();
        symlink(root.path().join("secret"), exchange.join("prediction.bin")).unwrap();

        assert!(import_candidate_artifact(
            &exchange, "prediction.bin", &private, "copy.bin", 1_024
        ).is_err());
    }

    #[test]
    fn rejects_directories_and_oversized_files() {
        let root = tempfile::tempdir().unwrap();
        let exchange = root.path().join("exchange");
        let private = root.path().join("private");
        fs::create_dir(&exchange).unwrap();
        fs::create_dir(&private).unwrap();
        fs::create_dir(exchange.join("directory")).unwrap();
        fs::write(exchange.join("large.bin"), vec![0_u8; 32]).unwrap();

        assert_eq!(
            import_candidate_artifact(&exchange, "directory", &private, "copy-dir", 64),
            Err(IntakeError::SourceNotRegular)
        );
        assert!(matches!(
            import_candidate_artifact(&exchange, "large.bin", &private, "copy-large", 8),
            Err(IntakeError::SourceTooLarge { size: 32, maximum: 8 })
        ));
    }

    #[test]
    fn rejects_hard_link_sources_and_existing_private_destination() {
        let root = tempfile::tempdir().unwrap();
        let exchange = root.path().join("exchange");
        let private = root.path().join("private");
        fs::create_dir(&exchange).unwrap();
        fs::create_dir(&private).unwrap();
        fs::write(exchange.join("original.bin"), b"data").unwrap();
        hard_link(exchange.join("original.bin"), exchange.join("prediction.bin")).unwrap();

        assert!(matches!(
            import_candidate_artifact(&exchange, "prediction.bin", &private, "copy.bin", 1_024),
            Err(IntakeError::SourceHasMultipleLinks { .. })
        ));

        fs::write(private.join("existing.bin"), b"keep").unwrap();
        assert!(import_candidate_artifact(
            &exchange, "original.bin", &private, "existing.bin", 1_024
        ).is_err());
        assert_eq!(fs::read(private.join("existing.bin")).unwrap(), b"keep");
    }
}
