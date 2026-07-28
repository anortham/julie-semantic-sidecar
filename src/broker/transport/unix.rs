use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Listener(UnixListener);

impl Listener {
    pub fn accept(&self) -> io::Result<Connection> {
        self.0.accept().map(|(stream, _)| Connection(stream))
    }
}

pub struct Connection(UnixStream);

impl Read for Connection {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.0.read(buffer)
    }
}

impl Write for Connection {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

pub fn bind(path: &Path, endpoint_bound: &AtomicBool) -> io::Result<Listener> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Unix broker endpoint must be absolute",
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "broker endpoint has no parent")
    })?;
    std::fs::create_dir_all(parent)?;
    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    let listener = UnixListener::bind(path)?;
    endpoint_bound.store(true, Ordering::Release);
    if let Err(err) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        endpoint_bound.store(false, Ordering::Release);
        let _ = std::fs::remove_file(path);
        return Err(err);
    }
    Ok(Listener(listener))
}
