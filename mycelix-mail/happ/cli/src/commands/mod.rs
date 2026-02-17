pub mod did;
pub mod export;
pub mod inbox;
/// Command implementations for Mycelix Mail CLI
///
/// This module contains all command handlers that implement the CLI functionality.
/// Each command is in its own submodule for maintainability.
pub mod init;
pub mod read;
pub mod search;
pub mod send;
pub mod status;
pub mod sync;
pub mod trust;
