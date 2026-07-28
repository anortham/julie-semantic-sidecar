use fs4::FileExt;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

pub struct ServiceLease {
    _file: File,
}

pub struct AcceleratorLease {
    _file: File,
}

impl ServiceLease {
    pub fn try_acquire(path: &Path) -> io::Result<Option<Self>> {
        try_acquire(path).map(|lease| lease.map(|file| Self { _file: file }))
    }
}

impl AcceleratorLease {
    pub fn try_acquire(path: &Path) -> io::Result<Option<Self>> {
        try_acquire(path).map(|lease| lease.map(|file| Self { _file: file }))
    }
}

fn try_acquire(path: &Path) -> io::Result<Option<File>> {
    secure_parent(path)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    match FileExt::try_lock(&file) {
        Ok(()) => Ok(Some(file)),
        Err(fs4::TryLockError::WouldBlock) => Ok(None),
        Err(fs4::TryLockError::Error(err)) => Err(err),
    }
}

fn secure_parent(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "lock path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
