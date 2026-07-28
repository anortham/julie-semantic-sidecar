use std::ffi::c_void;
use std::io::{self, Read, Write};
use std::mem::size_of;
use std::os::windows::io::{AsRawHandle, RawHandle};
use std::pin::Pin;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, ERROR_IO_PENDING, ERROR_NOT_FOUND, ERROR_OPERATION_ABORTED,
    ERROR_PIPE_CONNECTED, GENERIC_ALL, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::{
    AddAccessAllowedAce, GetLengthSid, GetTokenInformation, InitializeAcl,
    InitializeSecurityDescriptor, SetSecurityDescriptorDacl, TokenUser, ACCESS_ALLOWED_ACE, ACL,
    ACL_REVISION, SECURITY_ATTRIBUTES, SECURITY_DESCRIPTOR, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    ReadFile, WriteFile, FILE_FLAG_OVERLAPPED, PIPE_ACCESS_DUPLEX,
};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, PIPE_READMODE_BYTE,
    PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess, OpenProcessToken};
use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};

const PIPE_BUFFER_SIZE: u32 = 64 * 1024;
const PIPE_PREFIX: &str = r"\\.\pipe\";

#[derive(Clone)]
pub struct Listener {
    inner: Arc<ListenerInner>,
}

struct ListenerInner {
    endpoint: Vec<u16>,
    first: Mutex<Option<OwnedPipe>>,
    pending: Mutex<Vec<usize>>,
    cancel_requested: AtomicBool,
}

impl Listener {
    pub fn bind(endpoint: &str, endpoint_bound: &AtomicBool) -> io::Result<Self> {
        validate_endpoint(endpoint)?;
        let endpoint: Vec<u16> = endpoint.encode_utf16().chain([0]).collect();
        let first = create_instance(&endpoint)?;
        endpoint_bound.store(true, Ordering::Release);
        Ok(Self {
            inner: Arc::new(ListenerInner {
                endpoint,
                first: Mutex::new(Some(first)),
                pending: Mutex::new(Vec::new()),
                cancel_requested: AtomicBool::new(false),
            }),
        })
    }

    pub fn accept(&self) -> io::Result<Connection> {
        let pipe = self
            .inner
            .first
            .lock()
            .map_err(poisoned)?
            .take()
            .map_or_else(|| create_instance(&self.inner.endpoint), Ok)?;
        let handle = pipe.handle;
        let _registration = PendingAccept::register(&self.inner.pending, handle)?;
        let mut operation = PendingIo::new(handle)?;
        let connected = unsafe { ConnectNamedPipe(handle, operation.overlapped()) };
        let connect_error = (connected == 0).then(|| unsafe { GetLastError() });
        if self.inner.cancel_requested.swap(false, Ordering::AcqRel) {
            operation.cancel();
        }
        if connected == 0 {
            let error = connect_error.expect("failed ConnectNamedPipe must capture last-error");
            match error {
                ERROR_IO_PENDING => {
                    operation.complete()?;
                }
                ERROR_PIPE_CONNECTED => operation.finish_without_wait(),
                _ => {
                    operation.finish_without_wait();
                    return Err(io_error(error));
                }
            }
        } else {
            operation.finish_without_wait();
        }
        Ok(Connection {
            inner: Arc::new(PipeHandle {
                pipe,
                cancel_requested: AtomicBool::new(false),
            }),
        })
    }

    pub fn cancel_pending(&self) -> io::Result<()> {
        self.inner.cancel_requested.store(true, Ordering::Release);
        let handles = self.inner.pending.lock().map_err(poisoned)?;
        for handle in handles.iter().copied() {
            cancel_handle(handle as HANDLE)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Connection {
    inner: Arc<PipeHandle>,
}

impl Connection {
    pub fn cancel_io(&self) -> io::Result<()> {
        self.inner.cancel_requested.store(true, Ordering::Release);
        cancel_handle(self.inner.pipe.handle)
    }
}

impl Read for Connection {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let size = buffer.len().min(u32::MAX as usize) as u32;
        let mut immediate_bytes = 0;
        let mut operation = PendingIo::new(self.inner.pipe.handle)?;
        let started = unsafe {
            ReadFile(
                self.inner.pipe.handle,
                buffer.as_mut_ptr(),
                size,
                &mut immediate_bytes,
                operation.overlapped(),
            )
        };
        let start_error = (started == 0).then(|| unsafe { GetLastError() });
        if self.inner.cancel_requested.swap(false, Ordering::AcqRel) {
            operation.cancel();
        }
        finish_io(started, start_error, immediate_bytes, &mut operation)
    }
}

impl Write for Connection {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let size = buffer.len().min(u32::MAX as usize) as u32;
        let mut immediate_bytes = 0;
        let mut operation = PendingIo::new(self.inner.pipe.handle)?;
        let started = unsafe {
            WriteFile(
                self.inner.pipe.handle,
                buffer.as_ptr(),
                size,
                &mut immediate_bytes,
                operation.overlapped(),
            )
        };
        let start_error = (started == 0).then(|| unsafe { GetLastError() });
        if self.inner.cancel_requested.swap(false, Ordering::AcqRel) {
            operation.cancel();
        }
        finish_io(started, start_error, immediate_bytes, &mut operation)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl AsRawHandle for Connection {
    fn as_raw_handle(&self) -> RawHandle {
        self.inner.pipe.handle.cast()
    }
}

fn finish_io(
    started: i32,
    start_error: Option<u32>,
    immediate_bytes: u32,
    operation: &mut PendingIo,
) -> io::Result<usize> {
    if started != 0 {
        operation.finish_without_wait();
        return Ok(immediate_bytes as usize);
    }
    let error = start_error.expect("failed overlapped I/O must capture last-error");
    if error != ERROR_IO_PENDING {
        operation.finish_without_wait();
        return Err(io_error(error));
    }
    operation.complete().map(|bytes| bytes as usize)
}

fn validate_endpoint(endpoint: &str) -> io::Result<()> {
    if !endpoint.starts_with(PIPE_PREFIX) || endpoint.len() == PIPE_PREFIX.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            r"Windows broker endpoint must use the full \\.\pipe\<name> form",
        ));
    }
    if endpoint.encode_utf16().any(|unit| unit == 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Windows broker endpoint contains NUL",
        ));
    }
    Ok(())
}

fn create_instance(endpoint: &[u16]) -> io::Result<OwnedPipe> {
    let mut security = PipeSecurity::current_user()?;
    let attributes = security.attributes();
    let handle = unsafe {
        CreateNamedPipeW(
            endpoint.as_ptr(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            PIPE_UNLIMITED_INSTANCES,
            PIPE_BUFFER_SIZE,
            PIPE_BUFFER_SIZE,
            0,
            &attributes,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        Err(last_error())
    } else {
        Ok(OwnedPipe { handle })
    }
}

struct PipeSecurity {
    descriptor: Box<SECURITY_DESCRIPTOR>,
    _acl: Vec<usize>,
}

impl PipeSecurity {
    fn current_user() -> io::Result<Self> {
        let token_user = TokenUserBuffer::current()?;
        let sid = token_user.sid();
        let sid_length = unsafe { GetLengthSid(sid) };
        if sid_length == 0 {
            return Err(last_error());
        }
        let acl_bytes = size_of::<ACL>() + size_of::<ACCESS_ALLOWED_ACE>() - size_of::<u32>()
            + sid_length as usize;
        let words = acl_bytes.div_ceil(size_of::<usize>());
        let mut acl = vec![0_usize; words];
        let acl_ptr = acl.as_mut_ptr().cast::<ACL>();
        if unsafe { InitializeAcl(acl_ptr, acl_bytes as u32, ACL_REVISION) } == 0 {
            return Err(last_error());
        }
        if unsafe { AddAccessAllowedAce(acl_ptr, ACL_REVISION, GENERIC_ALL, sid) } == 0 {
            return Err(last_error());
        }
        let mut descriptor = Box::<SECURITY_DESCRIPTOR>::default();
        if unsafe {
            InitializeSecurityDescriptor(
                descriptor.as_mut() as *mut SECURITY_DESCRIPTOR as *mut c_void,
                1,
            )
        } == 0
        {
            return Err(last_error());
        }
        if unsafe {
            SetSecurityDescriptorDacl(
                descriptor.as_mut() as *mut SECURITY_DESCRIPTOR as *mut c_void,
                1,
                acl_ptr,
                0,
            )
        } == 0
        {
            return Err(last_error());
        }
        Ok(Self {
            descriptor,
            _acl: acl,
        })
    }

    fn attributes(&mut self) -> SECURITY_ATTRIBUTES {
        SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: self.descriptor.as_mut() as *mut SECURITY_DESCRIPTOR
                as *mut c_void,
            bInheritHandle: 0,
        }
    }
}

struct TokenUserBuffer {
    token: HANDLE,
    words: Vec<usize>,
}

impl TokenUserBuffer {
    fn current() -> io::Result<Self> {
        let mut token = null_mut();
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(last_error());
        }
        let mut needed = 0;
        unsafe {
            GetTokenInformation(token, TokenUser, null_mut(), 0, &mut needed);
        }
        if needed == 0 {
            unsafe {
                CloseHandle(token);
            }
            return Err(last_error());
        }
        let mut words = vec![0_usize; (needed as usize).div_ceil(size_of::<usize>())];
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                words.as_mut_ptr().cast(),
                needed,
                &mut needed,
            )
        } == 0
        {
            let error = last_error();
            unsafe {
                CloseHandle(token);
            }
            return Err(error);
        }
        Ok(Self { token, words })
    }

    fn sid(&self) -> *mut c_void {
        unsafe { (*(self.words.as_ptr().cast::<TOKEN_USER>())).User.Sid }
    }
}

impl Drop for TokenUserBuffer {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.token);
        }
    }
}

struct PipeHandle {
    pipe: OwnedPipe,
    cancel_requested: AtomicBool,
}

unsafe impl Send for PipeHandle {}
unsafe impl Sync for PipeHandle {}

impl Drop for PipeHandle {
    fn drop(&mut self) {
        unsafe {
            DisconnectNamedPipe(self.pipe.handle);
        }
    }
}

struct OwnedPipe {
    handle: HANDLE,
}

unsafe impl Send for OwnedPipe {}

impl Drop for OwnedPipe {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

struct PendingAccept<'a> {
    pending: &'a Mutex<Vec<usize>>,
    handle: usize,
}

impl<'a> PendingAccept<'a> {
    fn register(pending: &'a Mutex<Vec<usize>>, handle: HANDLE) -> io::Result<Self> {
        pending.lock().map_err(poisoned)?.push(handle as usize);
        Ok(Self {
            pending,
            handle: handle as usize,
        })
    }
}

impl Drop for PendingAccept<'_> {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(index) = pending.iter().position(|handle| *handle == self.handle) {
                pending.swap_remove(index);
            }
        }
    }
}

struct PendingIo {
    handle: HANDLE,
    state: Pin<Box<OverlappedState>>,
    pending: bool,
}

impl PendingIo {
    fn new(handle: HANDLE) -> io::Result<Self> {
        let event = unsafe { CreateEventW(null(), 1, 0, null()) };
        if event.is_null() {
            return Err(last_error());
        }
        let mut overlapped = OVERLAPPED::default();
        overlapped.hEvent = event;
        Ok(Self {
            handle,
            state: Box::pin(OverlappedState { overlapped, event }),
            pending: true,
        })
    }

    fn overlapped(&mut self) -> *mut OVERLAPPED {
        &mut self.state.as_mut().get_mut().overlapped
    }

    fn complete(&mut self) -> io::Result<u32> {
        let mut bytes = 0;
        let overlapped = self.overlapped();
        let result = unsafe { GetOverlappedResult(self.handle, overlapped, &mut bytes, 1) };
        self.pending = false;
        if result == 0 {
            Err(last_error())
        } else {
            Ok(bytes)
        }
    }

    fn finish_without_wait(&mut self) {
        self.pending = false;
    }

    fn cancel(&self) {
        unsafe {
            CancelIoEx(self.handle, &self.state.as_ref().get_ref().overlapped);
        }
    }
}

impl Drop for PendingIo {
    fn drop(&mut self) {
        if !self.pending {
            return;
        }
        unsafe {
            let overlapped = &mut self.state.as_mut().get_mut().overlapped;
            CancelIoEx(self.handle, overlapped);
            let mut bytes = 0;
            GetOverlappedResult(self.handle, overlapped, &mut bytes, 1);
        }
        self.pending = false;
    }
}

struct OverlappedState {
    overlapped: OVERLAPPED,
    event: HANDLE,
}

impl Drop for OverlappedState {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.event);
        }
    }
}

fn cancel_handle(handle: HANDLE) -> io::Result<()> {
    let cancelled = unsafe { CancelIoEx(handle, null()) };
    if cancelled != 0 {
        return Ok(());
    }
    let error = unsafe { GetLastError() };
    if error == ERROR_NOT_FOUND {
        Ok(())
    } else {
        Err(io_error(error))
    }
}

fn last_error() -> io::Error {
    io_error(unsafe { GetLastError() })
}

fn io_error(error: u32) -> io::Error {
    if error == ERROR_OPERATION_ABORTED {
        io::Error::new(
            io::ErrorKind::Interrupted,
            "overlapped pipe operation cancelled",
        )
    } else {
        io::Error::from_raw_os_error(error as i32)
    }
}

fn poisoned<T>(_error: std::sync::PoisonError<T>) -> io::Error {
    io::Error::other("Windows broker transport lock poisoned")
}
