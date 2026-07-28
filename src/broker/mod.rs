pub mod engine;
pub mod lease;
pub mod queue;
pub mod transport;
pub mod watchdog;

use crate::engine_trait::EmbedEngine;
use crate::health::Limits;
use crate::protocol::{
    internal_error_reply, process_line, read_request, FramedRequest, ProtocolReply,
};
use lease::{AcceleratorLease, ServiceLease};
use queue::{BrokerQueue, Dequeued, QueueError, RequestClass};
use serde_json::Value;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use watchdog::{OwnerWatchdog, RequestWatchdog};

const QUEUE_CAPACITY: usize = 64;
const QUEUE_DEADLINE: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrokerEndpoint {
    #[cfg(unix)]
    Unix(PathBuf),
    #[cfg(windows)]
    Windows(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokerConfig {
    pub model_id: String,
    pub endpoint: BrokerEndpoint,
    pub service_lock: PathBuf,
    pub accelerator_lock: PathBuf,
}

struct BrokerRequest {
    line: Vec<u8>,
    reply: SyncSender<io::Result<Option<ProtocolReply>>>,
}

pub fn serve(config: BrokerConfig) -> io::Result<()> {
    serve_with_loader(config, engine::load)
}

pub fn serve_with_loader<E, F>(config: BrokerConfig, loader: F) -> io::Result<()>
where
    E: EmbedEngine + 'static,
    F: FnOnce(&BrokerConfig, Option<AcceleratorLease>) -> io::Result<E>,
{
    let Some(_service_lease) = ServiceLease::try_acquire(&config.service_lock)? else {
        return Ok(());
    };
    let endpoint_bound = Arc::new(AtomicBool::new(false));
    let _owner_watchdog = OwnerWatchdog::start(
        io::stdin(),
        endpoint_path(&config.endpoint),
        Arc::clone(&endpoint_bound),
    )?;
    let accelerator_lease = AcceleratorLease::try_acquire(&config.accelerator_lock)?;
    let engine = loader(&config, accelerator_lease)?;
    let queue = Arc::new(BrokerQueue::new(QUEUE_CAPACITY));
    bind_and_accept(
        &config.endpoint,
        Arc::clone(&endpoint_bound),
        Arc::clone(&queue),
    )?;
    let watchdog = RequestWatchdog::start()?;

    loop {
        let request = match queue.dequeue() {
            Dequeued::Ready(request) => request,
            Dequeued::Expired(request) => {
                let reply = failure_reply(
                    &request.line,
                    "BrokerRequestExpired",
                    "broker request expired",
                )
                .map(Some);
                let _ = request.reply.send(reply);
                continue;
            }
        };
        let _active = watchdog.arm();
        let reply = process_line(&request.line, &engine, Limits::default());
        let _ = request.reply.send(reply);
    }
}

fn bind_and_accept(
    endpoint: &BrokerEndpoint,
    endpoint_bound: Arc<AtomicBool>,
    queue: Arc<BrokerQueue<BrokerRequest>>,
) -> io::Result<()> {
    let listener = transport::bind(endpoint, &endpoint_bound)?;
    thread::Builder::new()
        .name("semantic-broker-accept".to_string())
        .spawn(move || loop {
            match listener.accept() {
                Ok(stream) => spawn_connection(stream, Arc::clone(&queue)),
                Err(err) => {
                    eprintln!("julie-semantic-sidecar: broker accept failed: {err}");
                }
            }
        })?;
    Ok(())
}

fn spawn_connection(stream: transport::Connection, queue: Arc<BrokerQueue<BrokerRequest>>) {
    if let Err(err) = thread::Builder::new()
        .name("semantic-broker-connection".to_string())
        .spawn(move || {
            if let Err(err) = handle_connection(stream, queue) {
                eprintln!("julie-semantic-sidecar: broker connection failed: {err}");
            }
        })
    {
        eprintln!("julie-semantic-sidecar: could not start broker connection: {err}");
    }
}

fn handle_connection(
    stream: transport::Connection,
    queue: Arc<BrokerQueue<BrokerRequest>>,
) -> io::Result<()> {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    loop {
        match read_request(&mut reader, &mut line, Limits::default())? {
            FramedRequest::Eof => return Ok(()),
            FramedRequest::Rejected(reply) => {
                write_reply(reader.get_mut(), &reply)?;
                continue;
            }
            FramedRequest::Line => {}
        }
        let class = request_class(&line);
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let request = BrokerRequest {
            line: line.clone(),
            reply: reply_tx,
        };
        if queue.try_enqueue(class, request, Instant::now() + QUEUE_DEADLINE)
            == Err(QueueError::Full)
        {
            let reply = failure_reply(&line, "BrokerQueueFull", "broker queue is full")?;
            write_reply(reader.get_mut(), &reply)?;
            continue;
        }
        let reply = reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "broker scheduler stopped"))??;
        if let Some(reply) = reply {
            write_reply(reader.get_mut(), &reply)?;
            if reply.stop_connection {
                return Ok(());
            }
        }
    }
}

fn write_reply(writer: &mut impl Write, reply: &ProtocolReply) -> io::Result<()> {
    writer.write_all(reply.line.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn failure_reply(line: &[u8], kind: &str, message: &str) -> io::Result<ProtocolReply> {
    internal_error_reply(&request_id(line), kind, message)
}

fn request_id(line: &[u8]) -> String {
    serde_json::from_slice::<Value>(line)
        .ok()
        .and_then(|request| {
            request
                .get("request_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default()
}

fn request_class(line: &[u8]) -> RequestClass {
    serde_json::from_slice::<Value>(line)
        .ok()
        .and_then(|request| {
            request
                .get("method")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .filter(|method| method == "embed_batch")
        .map_or(RequestClass::Interactive, |_| RequestClass::Batch)
}

fn endpoint_path(endpoint: &BrokerEndpoint) -> Option<PathBuf> {
    match endpoint {
        #[cfg(unix)]
        BrokerEndpoint::Unix(path) => Some(path.clone()),
        #[cfg(windows)]
        BrokerEndpoint::Windows(_) => None,
    }
}
