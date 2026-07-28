use crate::broker::lease::AcceleratorLease;
use crate::broker::BrokerConfig;
use crate::engine::{BackendPolicy, LlamaEngine};
use crate::engine_trait::{
    EmbedEngine, EmbedOutput, EngineError, EngineFailureClass, Role, UnreadyEngine,
};
use serde_json::Value;
use std::cell::RefCell;
use std::io;

pub enum LoadedEngine {
    Ready(LlamaEngine),
    Unready(UnreadyEngine),
}

type EngineLoader<E> = Box<dyn FnMut(BackendPolicy) -> Result<E, EngineError>>;

struct BrokerEngineState<E> {
    inner: Option<E>,
    accelerator_lease: Option<AcceleratorLease>,
    loader: Option<EngineLoader<E>>,
    degraded_reason: Option<&'static str>,
    terminal_failure: Option<EngineError>,
}

pub struct BrokerEngine<E> {
    state: RefCell<BrokerEngineState<E>>,
}

impl<E> BrokerEngine<E> {
    pub fn new(inner: E, accelerator_lease: Option<AcceleratorLease>) -> Self {
        Self {
            state: RefCell::new(BrokerEngineState {
                inner: Some(inner),
                accelerator_lease,
                loader: None,
                degraded_reason: None,
                terminal_failure: None,
            }),
        }
    }

    pub fn accelerator_lease_held(&self) -> bool {
        self.state.borrow().accelerator_lease.is_some()
    }
}

impl<E: EmbedEngine + 'static> BrokerEngine<E> {
    pub fn load_with<F>(
        mut accelerator_lease: Option<AcceleratorLease>,
        mut loader: F,
    ) -> Result<Self, EngineError>
    where
        F: FnMut(BackendPolicy) -> Result<E, EngineError> + 'static,
    {
        let policy = if accelerator_lease.is_some() {
            BackendPolicy::Auto
        } else {
            BackendPolicy::CpuOnly
        };
        let inner = loader(policy)?;
        if policy == BackendPolicy::CpuOnly && inner.is_accelerated() {
            return Err(EngineError::new(
                "BackendPolicy",
                "CpuOnly loader resolved an accelerated backend",
            ));
        }
        if !inner.is_accelerated() {
            accelerator_lease = None;
        }
        Ok(Self {
            state: RefCell::new(BrokerEngineState {
                inner: Some(inner),
                accelerator_lease,
                loader: Some(Box::new(loader)),
                degraded_reason: None,
                terminal_failure: None,
            }),
        })
    }
}

impl<E: EmbedEngine> EmbedEngine for BrokerEngine<E> {
    fn health_facts(&self) -> Result<Value, EngineError> {
        let state = self.state.borrow();
        if let Some(err) = &state.terminal_failure {
            return Err(err.clone());
        }
        let mut health = state
            .inner
            .as_ref()
            .expect("broker engine invariant: live engine or terminal failure")
            .health_facts()?;
        crate::health::apply_broker_runtime_facts(
            &mut health,
            state.accelerator_lease.is_some(),
            state.degraded_reason,
        );
        Ok(health)
    }

    fn is_accelerated(&self) -> bool {
        self.state
            .borrow()
            .inner
            .as_ref()
            .is_some_and(EmbedEngine::is_accelerated)
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<EmbedOutput, EngineError> {
        let mut state = self.state.borrow_mut();
        if let Some(err) = &state.terminal_failure {
            return Err(err.clone());
        }
        let inner = state
            .inner
            .as_ref()
            .expect("broker engine invariant: live engine or terminal failure");
        let first = inner.embed(texts, role);
        let recover = first.as_ref().is_err_and(|err| {
            err.failure_class == EngineFailureClass::ResourceExhausted
                && inner.is_accelerated()
                && state.accelerator_lease.is_some()
                && state.loader.is_some()
        });
        if !recover {
            return first;
        }

        drop(state.inner.take());
        drop(state.accelerator_lease.take());
        state.degraded_reason = Some(crate::health::ACCELERATOR_RESOURCE_EXHAUSTED);
        let loaded = state
            .loader
            .as_mut()
            .expect("recoverable broker engine has a loader")(
            BackendPolicy::CpuOnly
        );
        let cpu = match loaded {
            Ok(engine) if !engine.is_accelerated() => engine,
            Ok(_) => {
                let err = EngineError::new(
                    "BackendPolicy",
                    "CpuOnly loader resolved an accelerated backend",
                );
                state.terminal_failure = Some(err.clone());
                return Err(err);
            }
            Err(err) => {
                state.terminal_failure = Some(err.clone());
                return Err(err);
            }
        };
        state.inner = Some(cpu);
        state
            .inner
            .as_ref()
            .expect("CPU engine was installed")
            .embed(texts, role)
    }
}

impl EmbedEngine for LoadedEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        match self {
            Self::Ready(engine) => engine.health_facts(),
            Self::Unready(engine) => engine.health_facts(),
        }
    }

    fn is_accelerated(&self) -> bool {
        match self {
            Self::Ready(engine) => engine.is_accelerated(),
            Self::Unready(engine) => engine.is_accelerated(),
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

    let load_cache_dir = cache_dir.clone();
    let loader = move |policy| match LlamaEngine::load_with_policy(pin, &load_cache_dir, policy) {
        Ok(engine) => Ok(LoadedEngine::Ready(engine)),
        Err(err) if err.kind == "ModelNotPrepared" => {
            eprintln!("julie-semantic-sidecar: {err}");
            Ok(LoadedEngine::Unready(UnreadyEngine::new(
                crate::health::MODEL_NOT_PREPARED,
            )))
        }
        Err(err) => Err(err),
    };
    BrokerEngine::load_with(accelerator_lease, loader).map_err(|err| {
        io::Error::other(format!(
            "cannot load model '{}' from {}: {err}",
            config.model_id,
            cache_dir.display()
        ))
    })
}
