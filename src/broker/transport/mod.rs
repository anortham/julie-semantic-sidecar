#[cfg(unix)]
pub mod unix;
#[cfg(windows)]
pub mod windows;

#[cfg(unix)]
pub use unix::{Connection, Listener};
#[cfg(windows)]
pub use windows::{Connection, Listener};

use super::BrokerEndpoint;
use std::io;
use std::sync::atomic::AtomicBool;

pub fn bind(endpoint: &BrokerEndpoint, endpoint_bound: &AtomicBool) -> io::Result<Listener> {
    match endpoint {
        #[cfg(unix)]
        BrokerEndpoint::Unix(path) => unix::bind(path, endpoint_bound),
        #[cfg(windows)]
        BrokerEndpoint::Windows(path) => windows::Listener::bind(path, endpoint_bound),
    }
}
