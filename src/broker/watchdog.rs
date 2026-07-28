use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const REQUEST_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(60);

pub struct OwnerWatchdog {
    _thread: thread::JoinHandle<()>,
}

impl OwnerWatchdog {
    pub fn start<R>(
        mut owner: R,
        endpoint: Option<PathBuf>,
        endpoint_bound: Arc<AtomicBool>,
    ) -> io::Result<Self>
    where
        R: Read + Send + 'static,
    {
        let thread = thread::Builder::new()
            .name("semantic-broker-owner-watchdog".to_string())
            .spawn(move || {
                let mut byte = [0_u8; 1];
                loop {
                    match owner.read(&mut byte) {
                        Ok(0) | Err(_) => {
                            if endpoint_bound.load(Ordering::Acquire) {
                                if let Some(path) = endpoint.as_deref() {
                                    let _ = std::fs::remove_file(path);
                                }
                            }
                            std::process::exit(0);
                        }
                        Ok(_) => {}
                    }
                }
            })?;
        Ok(Self { _thread: thread })
    }
}

pub struct RequestWatchdog {
    active_since: Arc<Mutex<Option<Instant>>>,
    _thread: thread::JoinHandle<()>,
}

pub struct ActiveRequest {
    active_since: Arc<Mutex<Option<Instant>>>,
}

impl RequestWatchdog {
    pub fn start() -> io::Result<Self> {
        let active_since = Arc::new(Mutex::new(None::<Instant>));
        let watched = Arc::clone(&active_since);
        let thread = thread::Builder::new()
            .name("semantic-broker-request-watchdog".to_string())
            .spawn(move || loop {
                thread::sleep(Duration::from_millis(100));
                let started = *watched.lock().expect("request watchdog mutex poisoned");
                if started.is_some_and(|started| started.elapsed() >= REQUEST_WATCHDOG_TIMEOUT) {
                    std::process::abort();
                }
            })?;
        Ok(Self {
            active_since,
            _thread: thread,
        })
    }

    pub fn arm(&self) -> ActiveRequest {
        *self
            .active_since
            .lock()
            .expect("request watchdog mutex poisoned") = Some(Instant::now());
        ActiveRequest {
            active_since: Arc::clone(&self.active_since),
        }
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        *self
            .active_since
            .lock()
            .expect("request watchdog mutex poisoned") = None;
    }
}
