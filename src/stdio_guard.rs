//! File-descriptor level stdout silencing for native calls that print.
//!
//! `semantic-sidecar-protocol-v1.md` § Stdout purity: a single stray byte on stdout
//! desyncs the consumer's line reader. C libraries write to file descriptor 1 directly,
//! bypassing any language-level stdout object, so the redirect has to happen at the
//! descriptor level: `dup(1)` to save, `dup2(2, 1)` so fd 1 points at stderr, run the
//! native call, then restore fd 1 from the saved descriptor.
//!
//! The restore lives in a [`Drop`] impl, which is the `finally`-equivalent the contract
//! requires: a load that fails or panics still leaves fd 1 correct.
//!
//! Rust's `std::io::Stdout` writes straight to fd 1 rather than caching a duplicate, so
//! restoring the descriptor is the whole restore — nothing has to be reattached.
//!
//! **Single-threaded only.** The guard mutates a process-global descriptor, so it must be
//! held only while no other thread writes to stdout. Startup, where every guarded call
//! lives, satisfies this.

use std::io::Write;

/// Runs `body` with file descriptor 1 pointing at stderr, restoring it afterwards.
///
/// Wrap every native call that may print: model load, backend probe, and the first-start
/// micro-benchmark. The value `body` returns is passed through untouched.
pub fn guarded<T>(body: impl FnOnce() -> T) -> T {
    let _guard = SilencedStdout::new();
    body()
}

/// An active fd-1 redirect that restores the real stdout when dropped.
///
/// Nesting is safe: an inner guard saves whatever fd 1 currently points at and restores
/// exactly that, so the outer guard still owns the real stdout.
#[derive(Debug)]
pub struct SilencedStdout {
    saved: Option<i32>,
}

impl SilencedStdout {
    /// Saves fd 1 and points it at fd 2 for the lifetime of the returned guard.
    ///
    /// A failing `dup`/`dup2` yields an inert guard rather than an error: silencing is a
    /// hygiene measure, and refusing to start because a descriptor could not be duplicated
    /// would trade a possible stray byte for a certain outage.
    pub fn new() -> Self {
        let _ = std::io::stdout().flush();
        let saved = sys::dup(sys::STDOUT);
        if saved < 0 {
            return Self { saved: None };
        }
        if sys::dup2(sys::STDERR, sys::STDOUT) < 0 {
            sys::close(saved);
            return Self { saved: None };
        }
        Self { saved: Some(saved) }
    }

    /// Whether the guard actually holds a redirect.
    pub fn is_active(&self) -> bool {
        self.saved.is_some()
    }
}

impl Default for SilencedStdout {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SilencedStdout {
    fn drop(&mut self) {
        let Some(saved) = self.saved.take() else {
            return;
        };
        let _ = std::io::stdout().flush();
        sys::dup2(saved, sys::STDOUT);
        sys::close(saved);
    }
}

#[cfg(unix)]
mod sys {
    pub const STDOUT: i32 = 1;
    pub const STDERR: i32 = 2;

    pub fn dup(fd: i32) -> i32 {
        unsafe { libc::dup(fd) }
    }

    pub fn dup2(source: i32, target: i32) -> i32 {
        unsafe { libc::dup2(source, target) }
    }

    pub fn close(fd: i32) {
        unsafe {
            libc::close(fd);
        }
    }
}

#[cfg(windows)]
mod sys {
    pub const STDOUT: i32 = 1;
    pub const STDERR: i32 = 2;

    pub fn dup(fd: i32) -> i32 {
        unsafe { libc::dup(fd) }
    }

    pub fn dup2(source: i32, target: i32) -> i32 {
        unsafe { libc::dup2(source, target) }
    }

    pub fn close(fd: i32) {
        unsafe {
            libc::close(fd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_passes_the_body_result_through() {
        assert_eq!(guarded(|| 7 * 6), 42);
    }

    #[test]
    fn a_guard_reports_itself_active_and_restores_on_drop() {
        let guard = SilencedStdout::new();
        assert!(guard.is_active());
        drop(guard);
        assert!(SilencedStdout::new().is_active());
    }

    #[test]
    fn guards_nest_without_losing_the_real_stdout() {
        let outer = SilencedStdout::new();
        {
            let inner = SilencedStdout::new();
            assert!(inner.is_active());
        }
        assert!(outer.is_active());
        drop(outer);
        println!();
    }

    #[test]
    fn a_raw_write_to_descriptor_one_lands_on_stderr_while_guarded() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let captured = std::fs::File::create(file.path()).expect("create");
        let captured_fd = raw_fd(&captured);

        let saved_stderr = sys::dup(sys::STDERR);
        assert!(saved_stderr >= 0);
        assert!(sys::dup2(captured_fd, sys::STDERR) >= 0);
        {
            let guard = SilencedStdout::new();
            assert!(guard.is_active());
            write_raw(sys::STDOUT, b"native loader chatter\n");
        }
        sys::dup2(saved_stderr, sys::STDERR);
        sys::close(saved_stderr);

        let landed = std::fs::read_to_string(file.path()).expect("read capture");
        assert!(landed.contains("native loader chatter"), "{landed:?}");
    }

    #[cfg(unix)]
    fn raw_fd(file: &std::fs::File) -> i32 {
        std::os::fd::AsRawFd::as_raw_fd(file)
    }

    #[cfg(windows)]
    fn raw_fd(file: &std::fs::File) -> i32 {
        unsafe {
            libc::open_osfhandle(
                std::os::windows::io::AsRawHandle::as_raw_handle(file) as isize,
                0,
            )
        }
    }

    fn write_raw(fd: i32, bytes: &[u8]) {
        #[cfg(unix)]
        unsafe {
            libc::write(fd, bytes.as_ptr().cast(), bytes.len());
        }
        #[cfg(windows)]
        unsafe {
            libc::write(fd, bytes.as_ptr().cast(), bytes.len() as u32);
        }
    }

    #[test]
    fn a_panicking_body_still_restores_the_descriptor() {
        let result = std::panic::catch_unwind(|| guarded(|| panic!("native load blew up")));
        assert!(result.is_err());
        assert!(SilencedStdout::new().is_active());
    }
}
