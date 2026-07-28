use crate::broker::lease::AcceleratorLease;
use crate::broker::BrokerConfig;
use crate::engine::LlamaEngine;
use crate::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role, UnreadyEngine};
use serde_json::Value;
use std::io;

pub enum LoadedEngine {
    Ready(LlamaEngine),
    Unready(UnreadyEngine),
}

pub struct BrokerEngine<E> {
    inner: E,
    _accelerator_lease: Option<AcceleratorLease>,
}

impl<E> BrokerEngine<E> {
    pub fn new(inner: E, accelerator_lease: Option<AcceleratorLease>) -> Self {
        Self {
            inner,
            _accelerator_lease: accelerator_lease,
        }
    }

    pub fn accelerator_lease_held(&self) -> bool {
        self._accelerator_lease.is_some()
    }
}

impl<E: EmbedEngine> EmbedEngine for BrokerEngine<E> {
    fn health_facts(&self) -> Result<Value, EngineError> {
        self.inner.health_facts()
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<EmbedOutput, EngineError> {
        self.inner.embed(texts, role)
    }
}

impl EmbedEngine for LoadedEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        match self {
            Self::Ready(engine) => engine.health_facts(),
            Self::Unready(engine) => engine.health_facts(),
        }
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<EmbedOutput, EngineError> {
        match self {
            Self::Ready(engine) => engine.embed(texts, role),
            Self::Unready(engine) => engine.embed(texts, role),
        }
    }
}

pub fn load(
    config: &BrokerConfig,
    accelerator_lease: Option<AcceleratorLease>,
) -> io::Result<BrokerEngine<LoadedEngine>> {
    let pin = crate::manifest::by_id(&config.model_id).ok_or_else(|| {
        io::Error::other(format!(
            "unknown model id '{}'; run `prepare --model`",
            config.model_id
        ))
    })?;
    let cache_dir = crate::prepare::cache_dir().map_err(|err| {
        io::Error::other(format!("cannot resolve the model cache: {}", err.message()))
    })?;
    if let Err(err) = crate::prepare::clean_stale_partials(&cache_dir) {
        eprintln!("julie-semantic-sidecar: could not clean stale partial downloads: {err}");
    }

    match LlamaEngine::load(pin, &cache_dir) {
        Ok(engine) => Ok(BrokerEngine::new(
            LoadedEngine::Ready(engine),
            accelerator_lease,
        )),
        Err(err) if err.kind == "ModelNotPrepared" => {
            eprintln!("julie-semantic-sidecar: {err}");
            Ok(BrokerEngine::new(
                LoadedEngine::Unready(UnreadyEngine::new(crate::health::MODEL_NOT_PREPARED)),
                accelerator_lease,
            ))
        }
        Err(err) => Err(io::Error::other(format!(
            "cannot load model '{}' from {}: {err}",
            config.model_id,
            cache_dir.display()
        ))),
    }
}
